"""The humanization pipeline.

    mask code/tables -> parse blocks -> chunk on headings
      -> per chunk: model rewrite -> structural validation -> regex clean-up
                    -> stylometric score -> retry with targeted feedback
      -> reassemble -> restore frozen blocks -> lift the title out

Two properties are worth calling out, because they are the difference between
this and a single "make it sound human" prompt:

* The retry is *directed*. A failed chunk is not simply rewritten again; the
  measured signals that failed are turned into imperative instructions and fed
  back in. Blind retries wander, directed retries converge.
* Editing strength varies per chunk. Uniformly humanized text has its own
  fingerprint, so the intensity profile is chosen from a hash of the chunk.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.config import Settings
from app.html_bridge import html_to_markdown, looks_like_html, markdown_to_html
from app.llm import PRICING, ModelRefusal, Rewriter, Usage
from app.prompts import builder
from app.rules.postprocess import postprocess
from app.schemas import (
    AdversarialReport,
    ChunkTrace,
    HumanizeRequest,
    HumanizeResponse,
    MetricReport,
    UsageReport,
)
from app.scoring import metrics
from app.structure import (
    Chunk,
    MaskResult,
    MASK_RE,
    chunk_blocks,
    extract_title,
    mask_verbatim,
    parse_blocks,
    validate_rewrite,
)

if TYPE_CHECKING:
    from app.adversarial import AdversarialResult

log = logging.getLogger(__name__)

# Structural damage we refuse to ship. A dropped link or a mangled heading tree
# costs more than a slightly higher AI score.
SERIOUS_ISSUES = {"skeleton", "masked", "preserve_term", "urls"}

HEADING1_RE = re.compile(r"^\s{0,3}#\s+(.+?)\s*#*\s*$")


def to_report(m: metrics.Measurements, s: metrics.Score) -> MetricReport:
    return MetricReport(
        ai_score=s.value,
        verdict=s.verdict,
        words=m.words,
        sentences=m.sentences,
        mean_sentence_length=round(m.mean_sentence_length, 2),
        sentence_length_cv=round(m.sentence_length_cv, 3),
        short_sentences_per_150w=round(m.short_sentences_per_150w, 2),
        monotone_runs=m.monotone_runs,
        mattr=round(m.mattr, 3),
        hapax_ratio=round(m.hapax_ratio, 3),
        em_dashes_per_1k=round(m.em_dashes_per_1k, 2),
        semicolons_per_1k=round(m.semicolons_per_1k, 2),
        tell_density_per_1k=round(m.tell_density_per_1k, 2),
        opening_transition_rate=round(m.opening_transition_rate, 3),
        contraction_rate=round(m.contraction_rate, 2),
        paragraph_length_cv=round(m.paragraph_length_cv, 3),
        passive_rate=round(m.passive_rate, 3),
        penalties=s.penalties,
    )


@dataclass
class ChunkResult:
    chunk: Chunk
    markdown: str
    trace: ChunkTrace
    usage: Usage
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Candidate:
    markdown: str
    score: float
    attempt: int
    intensity: str
    acceptable: bool
    number_issue: str | None = None


class Pipeline:
    def __init__(self, settings: Settings, rewriter: Rewriter | None = None):
        self.settings = settings
        self.rewriter = rewriter or Rewriter(settings)

    # --- per chunk --------------------------------------------------------

    async def _rewrite_chunk(
        self,
        chunk: Chunk,
        req: HumanizeRequest,
        pack: metrics.LanguagePack,
        system_blocks: list[dict],
        mask: MaskResult,
        target: float,
        max_attempts: int,
        strength: str = "standard",
        texture: bool = False,
        pass_index: int = 0,
    ) -> ChunkResult:
        original = chunk.markdown
        # Only the frozen tokens that actually occur in this chunk may be
        # required back; validating against the whole document's set would flag
        # every chunk for the blocks it never contained.
        present = {m.group(0) for m in MASK_RE.finditer(original)}
        chunk_mask = MaskResult(
            text="", store={t: v for t, v in mask.store.items() if t in present}
        )
        frozen = list(chunk_mask.store)

        _, base_score, _ = metrics.analyse(original, req.language)
        usage = Usage()
        warnings: list[str] = []
        feedback: list[str] = []
        best: _Candidate | None = None
        attempts_made = 0

        for attempt in range(max_attempts):
            attempts_made = attempt + 1
            intensity = builder.pick_intensity(original, chunk.index, attempt, strength)
            user_message = builder.build_user_message(
                chunk_markdown=original,
                language=req.language,
                intensity=intensity,
                context_before=chunk.context_before,
                tone=req.tone,
                preserve_terms=req.preserve_terms,
                rewrite_headings=req.rewrite_headings,
                feedback=feedback or None,
                attempt=attempt,
                frozen_tokens=frozen,
                strength=strength,
                texture=texture,
            )

            try:
                completion = await self.rewriter.complete(
                    system_blocks, user_message, model=req.model, effort=req.effort
                )
            except ModelRefusal as exc:
                warnings.append(f"chunk {chunk.index}: {exc}")
                break

            usage.add(completion.usage)
            candidate_md = completion.text
            if not candidate_md.strip():
                feedback = ["Your previous response was empty. Return the rewritten Markdown."]
                continue

            issues = validate_rewrite(
                original, candidate_md, req.preserve_terms, mask=chunk_mask
            )
            serious = [i for i in issues if i.kind in SERIOUS_ISSUES]
            number_issue = next((i for i in issues if i.kind == "numbers"), None)

            # A per-request flag wins over the env default in both directions.
            do_postprocess = (
                req.postprocess
                if req.postprocess is not None
                else self.settings.enable_postprocess
            )
            if do_postprocess:
                candidate_md, _ = postprocess(
                    candidate_md,
                    pack,
                    seed=f"{chunk.index}:{attempt}",
                    protect_headings=not req.rewrite_headings,
                )

            measured, cand_score, _ = metrics.analyse(candidate_md, req.language)

            candidate = _Candidate(
                markdown=candidate_md,
                score=cand_score.value,
                attempt=attempt,
                intensity=intensity,
                acceptable=not serious,
                number_issue=number_issue.detail if number_issue else None,
            )
            if best is None or _better(candidate, best):
                best = candidate

            if serious:
                feedback = [
                    "Your previous rewrite broke the structure. " + i.detail for i in serious
                ] + [
                    "Reproduce the heading levels, paragraph count, list item count, link "
                    "URLs and frozen placeholders exactly as they appear in the input."
                ]
                continue

            if cand_score.value <= target:
                break

            feedback = metrics.metric_feedback(cand_score, measured)

        if best is None or not best.acceptable:
            if best is None:
                warnings.append(
                    f"chunk {chunk.index}: no usable rewrite produced; original text kept."
                )
            else:
                warnings.append(
                    f"chunk {chunk.index}: every rewrite damaged the structure; "
                    "original text kept."
                )
            return ChunkResult(
                chunk=chunk,
                markdown=original,
                trace=ChunkTrace(
                    index=chunk.index,
                    heading=chunk.heading,
                    attempts=attempts_made,
                    ai_score_before=base_score.value,
                    ai_score_after=base_score.value,
                    accepted_attempt=-1,
                    intensity="none",
                    pass_=pass_index,
                ),
                usage=usage,
                warnings=warnings,
            )

        if best.score > target:
            warnings.append(
                f"chunk {chunk.index}: best score {best.score} is above the target {target} "
                f"after {attempts_made} attempt(s)."
            )
        if best.number_issue:
            # Not a rejection — a number may have been spelled out legitimately —
            # but a stat-heavy article deserves a spot-check pointer.
            warnings.append(
                f"chunk {chunk.index}: numbers may have changed ({best.number_issue}); "
                "verify the facts in this section."
            )

        return ChunkResult(
            chunk=chunk,
            markdown=best.markdown,
            trace=ChunkTrace(
                index=chunk.index,
                heading=chunk.heading,
                attempts=attempts_made,
                ai_score_before=base_score.value,
                ai_score_after=best.score,
                accepted_attempt=best.attempt,
                intensity=best.intensity,
                pass_=pass_index,
            ),
            usage=usage,
            warnings=warnings,
        )

    # --- whole document ---------------------------------------------------

    async def _run_pass(
        self,
        source: str,
        req: HumanizeRequest,
        pack: metrics.LanguagePack,
        target: float,
        max_attempts: int,
        strength: str,
        texture: bool,
        pass_index: int,
    ) -> tuple[str, list[ChunkResult]]:
        """One full mask -> chunk -> rewrite -> reassemble pass over the document."""
        mask = mask_verbatim(source)
        blocks = parse_blocks(mask.text)
        if not blocks:
            raise ValueError("no content found in `text`")

        chunks = chunk_blocks(blocks, self.settings.chunk_target_words, self.settings.chunk_max_words)
        system_blocks = builder.build_system_blocks(pack, req.language, texture=texture)

        # The first chunk runs alone so it writes the prompt cache; the rest then
        # read from it instead of each paying for the rule set.
        results: list[ChunkResult] = []
        if chunks:
            results.append(
                await self._rewrite_chunk(
                    chunks[0], req, pack, system_blocks, mask, target, max_attempts,
                    strength=strength, texture=texture, pass_index=pass_index,
                )
            )
        if len(chunks) > 1:
            sem = asyncio.Semaphore(max(1, self.settings.concurrency))

            async def guarded(c: Chunk) -> ChunkResult:
                async with sem:
                    return await self._rewrite_chunk(
                        c, req, pack, system_blocks, mask, target, max_attempts,
                        strength=strength, texture=texture, pass_index=pass_index,
                    )

            results.extend(await asyncio.gather(*(guarded(c) for c in chunks[1:])))

        results.sort(key=lambda r: r.chunk.index)
        rebuilt = "\n\n".join(r.markdown.strip() for r in results if r.markdown.strip())
        rebuilt = mask.restore(rebuilt)
        return rebuilt, results

    async def run(self, req: HumanizeRequest) -> HumanizeResponse:
        settings = self.settings
        warnings_pre: list[str] = []
        strength = (req.strength or settings.strength or "standard").lower()
        if strength not in {"standard", "aggressive", "max"}:
            strength = "standard"
        passes = self._resolve_passes(req, strength)

        base_target = (
            req.target_ai_score if req.target_ai_score is not None else settings.target_ai_score
        )
        # Aggressive/max push the proxy target lower, which makes the retry loop
        # try harder. Floored so it stays reachable.
        target = max(3.0, base_target - 5.0) if strength in {"aggressive", "max"} else base_target
        max_attempts = req.max_attempts or settings.max_attempts
        model = req.model or settings.model
        pack = metrics.pack_for(req.language)

        # --- input format -------------------------------------------------
        # CMS payloads arrive as HTML, often with double-encoded newlines. Both
        # are converted to Markdown here and converted back at the end, because
        # every downstream stage — block parsing, sentence splitting, the
        # detector — reads raw tags as prose and gets the structure wrong.
        source = req.text.strip()
        output_format = (req.format or "auto").lower()
        html_doc = None
        if output_format == "html" or (output_format == "auto" and looks_like_html(source)):
            html_doc = html_to_markdown(source)
            source = html_doc.markdown
            output_format = "html"
            if html_doc.had_literal_newlines:
                warnings_pre.append(
                    "input contained literal \\n sequences rather than real newlines "
                    "(double-encoded JSON); they were converted."
                )
            if not source.strip():
                raise ValueError("no text content found after parsing the HTML input")
        else:
            output_format = "markdown"

        injected_title = False
        # A title supplied out-of-band still needs humanizing, so it rides
        # through the pipeline as an H1 and is lifted back out at the end.
        if req.title and req.title.strip():
            probe_blocks = parse_blocks(mask_verbatim(source).text)
            existing, _ = extract_title(probe_blocks)
            if not existing or existing.strip().lower() != req.title.strip().lower():
                source = f"# {req.title.strip()}\n\n{source}"
                injected_title = True

        measured_before, score_before, _ = metrics.analyse(
            mask_verbatim(source).text, req.language
        )

        usage = Usage()
        warnings: list[str] = list(warnings_pre)
        trajectory: list[float] = []
        last_results: list[ChunkResult] = []
        current = source

        for p in range(passes):
            # The first pass cleans; later passes add human texture rather than
            # sand the text flatter by re-running the cleanup prompt.
            texture = p > 0
            current, results = await self._run_pass(
                current, req, pack, target, max_attempts,
                strength=strength, texture=texture, pass_index=p,
            )
            last_results = results
            for r in results:
                usage.add(r.usage)
                warnings.extend(r.warnings)
            _, pass_score, _ = metrics.analyse(mask_verbatim(current).text, req.language)
            trajectory.append(pass_score.value)

        # --- adversarial pass ---------------------------------------------
        # Everything above optimises a hand-crafted surface proxy. This is the
        # stage that optimises a real detector, and it is the only one that
        # moves trained classifiers. Sentence-for-sentence substitution inside
        # paragraphs, so it cannot damage the structure the passes above built.
        adversarial = None
        if self._adversarial_enabled(req):
            adversarial = await self._run_adversarial(current, req, model)
            if adversarial:
                current = adversarial.markdown
                usage.add(adversarial.usage)
                warnings.extend(adversarial.warnings)
                if not adversarial.neural:
                    warnings.append(
                        "adversarial pass ran against the stylometric proxy, not a neural "
                        "detector. Set GUIDANCE_DETECTOR=local for this stage to affect "
                        "trained classifiers such as GPTZero or Copyleaks."
                    )

        if model not in PRICING:
            warnings.append(
                f"no published price on file for '{model}'; estimated_cost_usd is billed at "
                "Opus rates and may be wrong."
            )

        title, content = self._split_title(current, req, injected_title)
        measured_after, score_after, _ = metrics.analyse(content, req.language)
        if output_format == "html":
            # Scored as Markdown above (prose, not tags), then rendered back to
            # the format the caller sent.
            content = markdown_to_html(content, html_doc.frozen if html_doc else {})

        return HumanizeResponse(
            title=title,
            content=content,
            language=req.language,
            metrics=to_report(measured_after, score_after),
            metrics_before=to_report(measured_before, score_before),
            target_ai_score=target,
            target_met=score_after.value <= target,
            strength=strength,
            format=output_format,
            passes_run=passes,
            score_trajectory=[round(v, 2) for v in trajectory],
            adversarial=(
                AdversarialReport(
                    detector=adversarial.detector,
                    neural=adversarial.neural,
                    ai_probability_before=round(adversarial.probability_before, 4),
                    ai_probability_after=round(adversarial.probability_after, 4),
                    rounds_run=adversarial.rounds_run,
                    substitutions=adversarial.substitutions,
                    trajectory=adversarial.trajectory,
                )
                if adversarial
                else None
            ),
            chunks=[r.trace for r in last_results],
            usage=UsageReport(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens,
                api_calls=usage.api_calls,
                estimated_cost_usd=usage.cost_usd(model),
            ),
            model=model,
            warnings=warnings,
        )

    def _adversarial_enabled(self, req: HumanizeRequest) -> bool:
        if req.adversarial is not None:
            return req.adversarial
        return self.settings.enable_adversarial

    async def _run_adversarial(
        self, markdown: str, req: HumanizeRequest, model: str
    ) -> "AdversarialResult | None":
        from app.adversarial import AdversarialRefiner
        from app.detector import get_detector

        s = self.settings
        try:
            detector = get_detector(s)
        except Exception as exc:  # pragma: no cover - defensive
            log.error("guidance detector unavailable, skipping adversarial pass: %s", exc)
            return None

        refiner = AdversarialRefiner(
            detector=detector,
            rewriter=self.rewriter,
            rounds=req.adversarial_rounds or s.adversarial_rounds,
            candidates=s.adversarial_candidates,
            sentences_per_round=s.adversarial_sentences_per_round,
            target_probability=(
                req.adversarial_target
                if req.adversarial_target is not None
                else s.adversarial_target_probability
            ),
            min_sentence_probability=s.adversarial_min_sentence_probability,
            model=req.model,
            effort=req.effort,
        )
        return await refiner.refine(markdown, req.language)

    def _resolve_passes(self, req: HumanizeRequest, strength: str) -> int:
        """How many full passes to run. Explicit request wins, else per strength."""
        if req.passes:
            return max(1, min(3, req.passes))
        if self.settings.passes:
            return max(1, min(3, self.settings.passes))
        return 2 if strength == "max" else 1

    def _split_title(
        self, rebuilt: str, req: HumanizeRequest, injected_title: bool
    ) -> tuple[str, str]:
        """Lift the article title out of the body.

        A leading H1 is the article title, so it is returned in `title` and
        removed from `content` — most CMSes want those separate. A document that
        opens at H2 has no article title, so nothing is removed and the title is
        derived without touching the body.
        """
        blocks = parse_blocks(mask_verbatim(rebuilt).text)
        if blocks and blocks[0].kind == "heading" and blocks[0].level == 1:
            # Cut the H1 out of the raw text rather than re-rendering blocks,
            # so nothing else in the body is reformatted on the way through.
            lines = rebuilt.split("\n")
            for i, line in enumerate(lines):
                if HEADING1_RE.match(line):
                    title = HEADING1_RE.match(line).group(1).strip()
                    body = "\n".join(lines[i + 1 :]).lstrip("\n")
                    return title, body.strip() + "\n"

        if injected_title and req.title:
            return req.title.strip(), rebuilt.strip() + "\n"

        derived, _ = extract_title(blocks)
        if derived:
            return derived.strip(), rebuilt.strip() + "\n"
        if req.title:
            return req.title.strip(), rebuilt.strip() + "\n"
        first = next((b.text for b in blocks if b.text), "Untitled")
        return first[:120].strip(), rebuilt.strip() + "\n"


def _better(candidate: _Candidate, incumbent: _Candidate) -> bool:
    """Structural integrity outranks the score; among equals, lower score wins."""
    if candidate.acceptable != incumbent.acceptable:
        return candidate.acceptable
    return candidate.score < incumbent.score
