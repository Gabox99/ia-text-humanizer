"""Adversarial-pass tests with a scripted detector, so they run offline.

The fake detector assigns a probability from a word list rather than a model,
which makes the greedy selection, the rollback and the fact-safety gate all
deterministically checkable. Whether a real detector actually drops is measured
separately by scripts/detector_check.py, which needs the weights.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adversarial import (  # noqa: E402
    AdversarialRefiner,
    acceptable_candidate,
    parse_candidates,
    Target,
)
from app.detector.base import GuidanceDetector  # noqa: E402
from app.llm import Completion, Usage  # noqa: E402

DOC = """The system delivers robust synergy across the organization.

Teams shipped 42 features last quarter. The framework is comprehensive.

See https://example.com/x for details.
"""

# Words that make the fake detector suspicious, and their "human" replacements.
_BAD = ("robust", "synergy", "comprehensive", "framework", "delivers")


class ScriptedDetector(GuidanceDetector):
    """Probability = fraction of flagged words. Deterministic and inspectable."""

    name = "scripted"
    neural = True

    def __init__(self):
        self.doc_calls = 0
        self.sentence_calls = 0

    def _p(self, text: str) -> float:
        words = re.findall(r"[a-zA-Z']+", text.lower())
        if not words:
            return 0.0
        hits = sum(1 for w in words if w in _BAD)
        return min(1.0, hits / max(1, len(words)) * 6)

    def score_document(self, text: str) -> float:
        self.doc_calls += 1
        return self._p(text)

    def score_sentences(self, sentences: list[str]) -> list[float]:
        self.sentence_calls += 1
        return [self._p(s) for s in sentences]


class CandidateRewriter:
    """Returns a clean alternative plus a deliberately bad one."""

    def __init__(self, mode: str = "good"):
        self.mode = mode
        self.calls = 0
        self.prompts: list[str] = []

    async def complete(self, system_blocks, user_message, model=None, effort=None):
        self.calls += 1
        self.prompts.append(user_message)
        lines = []
        for m in re.finditer(r"^(\d+)\. (.+)$", user_message, re.MULTILINE):
            uid, sent = int(m.group(1)), m.group(2).strip()
            clean = sent
            for bad in _BAD:
                clean = re.sub(bad, "plain", clean, flags=re.IGNORECASE)
            if self.mode == "good":
                lines.append(f"[{uid}a] {clean}")
                # A candidate that mangles a number must be rejected outright.
                lines.append(f"[{uid}b] {clean.replace('42', '99')}")
            elif self.mode == "worse":
                lines.append(f"[{uid}a] {sent} robust synergy comprehensive")
            elif self.mode == "empty":
                pass
        return Completion(text="\n".join(lines), usage=Usage(api_calls=1))


def _refine(rewriter, detector, **kw):
    defaults = dict(rounds=3, candidates=2, sentences_per_round=6,
                    target_probability=0.05, min_sentence_probability=0.05)
    defaults.update(kw)
    r = AdversarialRefiner(detector, rewriter, **defaults)
    return asyncio.run(r.refine(DOC, "en-US"))


def test_probability_drops_and_structure_is_preserved():
    d = ScriptedDetector()
    res = _refine(CandidateRewriter("good"), d)
    assert res.probability_after < res.probability_before, res.trajectory
    assert res.substitutions > 0
    assert res.neural is True
    # Paragraph count and the link both survive: the pass only swaps sentences.
    assert len([p for p in res.markdown.split("\n\n") if p.strip()]) == 3
    assert "https://example.com/x" in res.markdown


def test_number_mangling_candidate_is_rejected():
    res = _refine(CandidateRewriter("good"), ScriptedDetector())
    assert "42" in res.markdown, "the real figure was lost"
    assert "99" not in res.markdown, "a candidate that changed a number was accepted"


def test_rollback_when_document_does_not_improve():
    d = ScriptedDetector()
    res = _refine(CandidateRewriter("worse"), d)
    # Every candidate is worse, so nothing may be substituted.
    assert res.substitutions == 0
    assert res.markdown.strip() == DOC.strip()
    assert res.probability_after == res.probability_before


def test_no_candidates_is_handled():
    res = _refine(CandidateRewriter("empty"), ScriptedDetector())
    assert res.substitutions == 0
    assert any("no usable candidates" in w for w in res.warnings)


def test_early_exit_when_already_below_target():
    # A document with none of the flagged words scores 0, so the pass must not
    # spend a single LLM call on it.
    clean = "The team shipped 42 features last quarter. People used them.\n"
    d = ScriptedDetector()
    assert d.score_document(clean) == 0.0
    rewriter = CandidateRewriter("good")
    r = AdversarialRefiner(d, rewriter, rounds=3, candidates=2,
                           target_probability=0.25, min_sentence_probability=0.05)
    res = asyncio.run(r.refine(clean, "en-US"))
    assert res.rounds_run == 0
    assert rewriter.calls == 0, "no LLM call should be made when already under target"
    assert res.markdown == clean


def test_acceptable_candidate_gate():
    orig = "Teams shipped 42 features last quarter."
    assert acceptable_candidate(orig, "The team put out 42 features last quarter.")
    assert not acceptable_candidate(orig, orig)                      # identical
    assert not acceptable_candidate(orig, "Teams shipped 43 features.")  # number changed
    assert not acceptable_candidate(orig, "Teams shipped.")          # too short
    assert not acceptable_candidate(orig, "Here is a rewritten version of it 42")
    assert not acceptable_candidate(
        "See https://example.com/x now.", "See the page now."
    )  # link dropped


def test_candidate_parsing():
    targets = [Target(1, 0, 0, "a", 0.9), Target(2, 0, 1, "b", 0.9)]
    raw = "[1a] first\n[1b] second\n[2a] third\n[9a] ignored\nnoise line"
    got = parse_candidates(raw, targets)
    assert got[1] == ["first", "second"]
    assert got[2] == ["third"]
    assert 9 not in got


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print("failures:", failures)
