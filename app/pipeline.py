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

from app.config import Settings
from app.llm import PRICING, ModelRefusal, Rewriter, Usage
from app.prompts import builder
from app.rules.postprocess import postprocess
from app.schemas import (
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
            intensity = builder.pick_intensity(original, chunk.index, attempt)
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
                ),
                usage=usage,
                warnings=warnings,
            )

        if best.score > target:
            warnings.append(
                f"chunk {chunk.index}: best score {best.score} is above the target {target} "
                f"after {attempts_made} attempt(s)."
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
            ),
            usage=usage,
            warnings=warnings,
        )

    # --- whole document ---------------------------------------------------

    async def run(self, req: HumanizeRequest) -> HumanizeResponse:
        settings = self.settings
        target = req.target_ai_score if req.target_ai_score is not None else settings.target_ai_score
        max_attempts = req.max_attempts or settings.max_attempts
        model = req.model or settings.model
        pack = metrics.pack_for(req.language)

        source = req.text.strip()
        injected_title = False
        # A title supplied out-of-band still needs humanizing, so it rides
        # through the pipeline as an H1 and is lifted back out at the end.
        if req.title and req.title.strip():
            probe_blocks = parse_blocks(mask_verbatim(source).text)
            existing, _ = extract_title(probe_blocks)
            if not existing or existing.strip().lower() != req.title.strip().lower():
                source = f"# {req.title.strip()}\n\n{source}"
                injected_title = True

        mask = mask_verbatim(source)
        blocks = parse_blocks(mask.text)
        if not blocks:
            raise ValueError("no content found in `text`")

        measured_before, score_before, _ = metrics.analyse(mask.text, req.language)
        chunks = chunk_blocks(blocks, settings.chunk_target_words, settings.chunk_max_words)
        system_blocks = builder.build_system_blocks(pack, req.language)

        # The first chunk runs alone so it writes the prompt cache; the rest
        # then read from it instead of each paying for the rule set.
        results: list[ChunkResult] = []
        if chunks:
            results.append(
                await self._rewrite_chunk(
                    chunks[0], req, pack, system_blocks, mask, target, max_attempts
                )
            )
        if len(chunks) > 1:
            sem = asyncio.Semaphore(max(1, settings.concurrency))

            async def guarded(c: Chunk) -> ChunkResult:
                async with sem:
                    return await self._rewrite_chunk(
                        c, req, pack, system_blocks, mask, target, max_attempts
                    )

            results.extend(await asyncio.gather(*(guarded(c) for c in chunks[1:])))

        results.sort(key=lambda r: r.chunk.index)

        usage = Usage()
        warnings: list[str] = []
        for r in results:
            usage.add(r.usage)
            warnings.extend(r.warnings)

        if model not in PRICING:
            warnings.append(
                f"no published price on file for '{model}'; estimated_cost_usd is billed at "
                "Opus rates and may be wrong."
            )

        rebuilt = "\n\n".join(r.markdown.strip() for r in results if r.markdown.strip())
        rebuilt = mask.restore(rebuilt)

        title, content = self._split_title(rebuilt, req, injected_title)

        measured_after, score_after, _ = metrics.analyse(content, req.language)

        return HumanizeResponse(
            title=title,
            content=content,
            language=req.language,
            metrics=to_report(measured_after, score_after),
            metrics_before=to_report(measured_before, score_before),
            target_ai_score=target,
            target_met=score_after.value <= target,
            chunks=[r.trace for r in results],
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
