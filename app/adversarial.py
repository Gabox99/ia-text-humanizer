"""Adversarial paraphrasing: detector-guided sentence substitution.

Implements the attack shape that the literature actually reports working against
trained classifiers, adapted to use Claude as the paraphraser:

    Adversarial Paraphrasing (arXiv:2506.07001) — score the text with a guidance
    detector, find the sentences carrying the machine signal, generate several
    paraphrase candidates for each, and keep the candidate that most reduces the
    detector score. Evasion transfers to detectors never used during the attack.

    SICO (arXiv:2305.10847) — the same greedy-substitution-against-a-proxy idea
    at word and sentence level, reported to drop GPTZero AUC 0.779 -> 0.184.

    Recursive paraphrasing (arXiv:2303.11156) — repeating the pass compounds the
    effect; the gain per round decays, so we hill-climb with a rollback instead
    of running a fixed number of rounds blind.

Why this is safe for structure: the pass only ever swaps one sentence for
another *inside a paragraph or blockquote*. Headings, list items, tables and
code are never touched, so unlike a free rewrite this stage cannot corrupt the
document skeleton. That is a property of the construction, not of the prompt.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from app.detector.base import GuidanceDetector
from app.llm import ModelRefusal, Rewriter, Usage
from app.scoring.metrics import split_sentences
from app.structure import Block, MASK_RE, blocks_to_markdown, parse_blocks

log = logging.getLogger(__name__)

# Only prose blocks are eligible. A heading is too short to carry much signal
# and is often SEO-locked; list items are structurally fragile.
_ELIGIBLE_KINDS = {"paragraph", "quote"}

_CANDIDATE_RE = re.compile(r"^\s*\[(\d+)([a-z])\]\s*(.+?)\s*$", re.MULTILINE)
_NUMBER_RE = re.compile(r"\d[\d.,]*")


@dataclass
class Target:
    """A sentence selected for substitution."""

    uid: int
    block_index: int
    sentence_index: int
    text: str
    ai_probability: float


@dataclass
class AdversarialResult:
    markdown: str
    rounds_run: int = 0
    substitutions: int = 0
    probability_before: float = 0.0
    probability_after: float = 0.0
    trajectory: list[float] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    warnings: list[str] = field(default_factory=list)
    detector: str = ""
    neural: bool = False


def _numbers(text: str) -> set[str]:
    return {m.strip(".,").replace(",", "") for m in _NUMBER_RE.findall(text)} - {""}


def _sentences_of(block: Block) -> list[str]:
    """Sentence split on the block's raw text, so inline Markdown is retained."""
    raw = " ".join(line.strip() for line in block.raw.split("\n")).strip()
    return split_sentences(raw)


def _rebuild_block(block: Block, sentences: list[str]) -> Block:
    joined = " ".join(s.strip() for s in sentences if s.strip())
    return Block(kind=block.kind, raw=joined, level=block.level, text=joined)


CANDIDATE_SYSTEM = """You rewrite individual sentences. You will be given numbered sentences \
from an article, each of which a machine-text classifier has flagged as reading like \
model output. For each one, produce alternative phrasings that a specific human writer \
would plausibly have written instead.

Rules, all mandatory:

1. PRESERVE MEANING EXACTLY. Every fact, number, name, date and logical relationship in \
the sentence must survive unchanged. Never alter a figure. Never add information.
2. ONE SENTENCE IN, ONE SENTENCE OUT. Do not split a sentence into two, do not merge, do \
not produce a fragment that cannot stand where the original stood.
3. KEEP INLINE MARKDOWN. Bold, italics, inline code and link syntax like [text](url) must \
be reproduced intact, URLs character for character.
4. MAKE THE ALTERNATIVES GENUINELY DIFFERENT FROM EACH OTHER. Vary length, clause order, \
register and word choice across them. Two near-identical options waste the slot.
5. WRITE LIKE A PERSON, NOT LIKE AN EDITOR SMOOTHING TEXT. Some alternatives should be \
blunt and short. Some should run long and slightly loose. Contractions where natural. \
Starting with And, But or So is fine. A little asymmetry is the point.
6. STAY IN THE SAME LANGUAGE as the input.

# Output format

For sentence 1, output exactly:

[1a] first alternative
[1b] second alternative
[1c] third alternative

Then sentence 2 as [2a], [2b], [2c], and so on. One alternative per line, nothing else \
on the line. No commentary, no preamble, no explanation, no blank-line padding."""


def build_candidate_prompt(targets: list[Target], count: int, language: str) -> str:
    lines = [
        f"Language: {language}",
        f"Produce {count} alternatives for each sentence below "
        f"(letters a through {chr(ord('a') + count - 1)}).",
        "",
        "Sentences to rewrite:",
    ]
    for t in targets:
        lines.append(f"{t.uid}. {t.text}")
    return "\n".join(lines)


def parse_candidates(raw: str, targets: list[Target]) -> dict[int, list[str]]:
    valid_uids = {t.uid for t in targets}
    out: dict[int, list[str]] = {t.uid: [] for t in targets}
    for match in _CANDIDATE_RE.finditer(raw):
        uid = int(match.group(1))
        text = match.group(3).strip()
        if uid in valid_uids and text:
            out[uid].append(text)
    return out


def acceptable_candidate(original: str, candidate: str) -> bool:
    """Reject candidates that break facts or clearly went off the rails."""
    if not candidate or candidate == original:
        return False
    # A number that appeared in the original must still be there, and no new
    # number may be invented. This is the fact-safety gate.
    if _numbers(original) != _numbers(candidate):
        return False
    ow, cw = len(original.split()), len(candidate.split())
    if ow and not (0.5 * ow <= cw <= 1.9 * ow):
        return False
    # The model occasionally answers with commentary instead of a sentence.
    if candidate.lower().startswith(("here", "sure", "alternative", "option", "rewritten")):
        return False
    # Links must survive intact.
    if set(re.findall(r"https?://[^\s\)\]]+", original)) - set(
        re.findall(r"https?://[^\s\)\]]+", candidate)
    ):
        return False
    return True


class AdversarialRefiner:
    """Detector-guided sentence substitution over an already-rewritten document."""

    def __init__(
        self,
        detector: GuidanceDetector,
        rewriter: Rewriter,
        *,
        rounds: int = 3,
        candidates: int = 4,
        sentences_per_round: int = 12,
        target_probability: float = 0.25,
        min_sentence_probability: float = 0.35,
        model: str | None = None,
        effort: str | None = None,
    ):
        self.detector = detector
        self.rewriter = rewriter
        self.rounds = rounds
        self.candidates = candidates
        self.sentences_per_round = sentences_per_round
        self.target_probability = target_probability
        self.min_sentence_probability = min_sentence_probability
        self.model = model
        self.effort = effort

    # --- indexing ---------------------------------------------------------

    def _index(self, blocks: list[Block]) -> tuple[dict[int, list[str]], list[Target]]:
        """Map eligible blocks to their sentences, and flatten into targets."""
        per_block: dict[int, list[str]] = {}
        flat: list[Target] = []
        uid = 1
        for bi, block in enumerate(blocks):
            if block.kind not in _ELIGIBLE_KINDS or MASK_RE.search(block.raw):
                continue
            sentences = _sentences_of(block)
            if not sentences:
                continue
            per_block[bi] = sentences
            for si, sentence in enumerate(sentences):
                flat.append(
                    Target(
                        uid=uid,
                        block_index=bi,
                        sentence_index=si,
                        text=sentence,
                        ai_probability=0.0,
                    )
                )
                uid += 1
        return per_block, flat

    # --- the loop ---------------------------------------------------------

    async def refine(self, markdown: str, language: str) -> AdversarialResult:
        result = AdversarialResult(markdown=markdown, detector=self.detector.name,
                                   neural=self.detector.neural)

        loop = asyncio.get_running_loop()

        def score_doc(text: str) -> float:
            return self.detector.score_document(text)

        def score_batch(items: list[str]) -> list[float]:
            return self.detector.score_sentences(items)

        current = markdown
        # Detector inference is synchronous CPU work; keep it off the event loop
        # so concurrent requests are not blocked by one document's scoring.
        baseline = await loop.run_in_executor(None, score_doc, current)
        result.probability_before = baseline
        result.trajectory.append(round(baseline, 4))
        best_prob = baseline

        if baseline <= self.target_probability:
            result.probability_after = baseline
            return result

        for round_index in range(self.rounds):
            blocks = parse_blocks(current)
            per_block, flat = self._index(blocks)
            if not flat:
                result.warnings.append("adversarial pass: no eligible prose sentences found")
                break

            probs = await loop.run_in_executor(None, score_batch, [t.text for t in flat])
            for target, p in zip(flat, probs):
                target.ai_probability = p

            targets = [
                t
                for t in sorted(flat, key=lambda x: x.ai_probability, reverse=True)
                if t.ai_probability >= self.min_sentence_probability
            ][: self.sentences_per_round]

            if not targets:
                log.info("adversarial: no sentence above the substitution threshold; stopping")
                break

            try:
                completion = await self.rewriter.complete(
                    [{"type": "text", "text": CANDIDATE_SYSTEM,
                      "cache_control": {"type": "ephemeral"}}],
                    build_candidate_prompt(targets, self.candidates, language),
                    model=self.model,
                    effort=self.effort,
                )
            except ModelRefusal as exc:
                result.warnings.append(f"adversarial pass: {exc}")
                break
            result.usage.add(completion.usage)

            by_uid = parse_candidates(completion.text, targets)
            # Score every candidate for every target in one batch.
            flat_candidates: list[str] = []
            provenance: list[tuple[Target, str]] = []
            for target in targets:
                for cand in by_uid.get(target.uid, []):
                    if acceptable_candidate(target.text, cand):
                        flat_candidates.append(cand)
                        provenance.append((target, cand))

            if not flat_candidates:
                result.warnings.append(
                    f"adversarial round {round_index + 1}: no usable candidates returned"
                )
                break

            cand_probs = await loop.run_in_executor(None, score_batch, flat_candidates)

            # Greedy pick per target: lowest-scoring candidate that also beats
            # the original. A candidate that scores worse is not an improvement.
            chosen: dict[int, tuple[Target, str, float]] = {}
            for (target, cand), prob in zip(provenance, cand_probs):
                if prob >= target.ai_probability:
                    continue
                incumbent = chosen.get(target.uid)
                if incumbent is None or prob < incumbent[2]:
                    chosen[target.uid] = (target, cand, prob)

            if not chosen:
                log.info("adversarial round %d: no candidate improved on its original",
                         round_index + 1)
                break

            trial_blocks = list(blocks)
            edited: dict[int, list[str]] = {bi: list(s) for bi, s in per_block.items()}
            for target, cand, _prob in chosen.values():
                edited[target.block_index][target.sentence_index] = cand
            for bi, sentences in edited.items():
                trial_blocks[bi] = _rebuild_block(blocks[bi], sentences)

            trial = blocks_to_markdown(trial_blocks)
            trial_prob = await loop.run_in_executor(None, score_doc, trial)
            result.rounds_run = round_index + 1
            result.trajectory.append(round(trial_prob, 4))

            # Hill climb with rollback. Sentence-level gains do not always add
            # up at document level, and accepting a worse document because the
            # parts looked better is how these loops drift.
            if trial_prob < best_prob:
                current = trial
                best_prob = trial_prob
                result.substitutions += len(chosen)
            else:
                log.info(
                    "adversarial round %d: document score did not improve "
                    "(%.3f -> %.3f); reverting",
                    round_index + 1, best_prob, trial_prob,
                )
                break

            if best_prob <= self.target_probability:
                break

        result.markdown = current
        result.probability_after = best_prob
        return result
