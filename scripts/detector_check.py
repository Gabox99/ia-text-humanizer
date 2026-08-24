#!/usr/bin/env python
"""Validate a guidance detector before trusting it to steer the rewrite.

This is the most important diagnostic in the project, and it exists because of
a concrete failure. An earlier candidate guidance detector
(`fakespot-ai/roberta-base-ai-text-detection-v1`) scored a real pipeline output
at 0.003 — confidently human — while GPTZero and Copyleaks both scored the same
text at 100% AI. Optimising against that detector cannot work: it already
believes the text is human, so it offers no gradient to climb.

A guidance detector is only useful if it *agrees with the detector that gates
your work*. This script measures that agreement on text you supply.

Usage
-----
    # Sanity check on built-in samples
    python scripts/detector_check.py

    # The check that matters: text a real detector flagged, and its verdict
    python scripts/detector_check.py --file flagged.md --expect ai
    python scripts/detector_check.py --file my_own_writing.md --expect human

    # Compare candidate models on the same input
    python scripts/detector_check.py --file flagged.md --expect ai \\
        --models desklib/ai-text-detector-v1.01 \\
                 fakespot-ai/roberta-base-ai-text-detection-v1

Read the AGREE / DISAGREE column, not the raw number. A model that disagrees
with your target detector is the wrong guide no matter how good its benchmark
scores look.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.detector.local_model import LocalNeuralDetector  # noqa: E402
from app.scoring.metrics import split_sentences, strip_markdown  # noqa: E402

SAMPLE_AI = """In today's rapidly evolving business landscape, data analytics has become a
pivotal component of organizational success. Companies across various industries are
leveraging robust analytical frameworks to unlock unprecedented insights. Furthermore, the
seamless integration of these tools facilitates comprehensive decision-making processes.
It is important to note that a holistic approach delivers significant advantages. Moreover,
organizations that harness these capabilities can streamline operations effectively."""

SAMPLE_HUMAN = """Three years ago I watched a company spend $400,000 on a data warehouse
nobody used. Not because the tech was bad. It wasn't. The problem was simpler and dumber
than that. Nobody had asked the sales team what they actually needed. What kills projects is
that the person paying for the dashboard and the person who has to open it every Monday
morning have never had a real conversation. So you get a beautiful thing that answers
questions nobody asked."""


def verdict(prob: float) -> str:
    if prob >= 0.75:
        return "ai"
    if prob <= 0.25:
        return "human"
    return "borderline"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="Text to score. Omit to use the built-in samples.")
    ap.add_argument(
        "--expect",
        choices=["ai", "human"],
        help="What your real detector said about --file. Enables the agreement check.",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["desklib/ai-text-detector-v1.01"],
        help="Guidance detector model ids to compare.",
    )
    ap.add_argument("--sentences", action="store_true", help="Show per-sentence attribution.")
    args = ap.parse_args()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8-sig")
        cases = [("your file", text, args.expect)]
    else:
        cases = [("built-in AI sample", SAMPLE_AI, "ai"),
                 ("built-in human sample", SAMPLE_HUMAN, "human")]

    words = len(strip_markdown(cases[0][1]).split())
    print(f"\ninput: {words} words\n")

    exit_code = 0
    for model_id in args.models:
        print(f"===== {model_id}")
        try:
            det = LocalNeuralDetector(model_id)
            t0 = time.time()
            det.warmup()
            print(f"  loaded in {time.time() - t0:.1f}s  "
                  f"(head={det._head}, max_len={det._max_length})")
        except Exception as exc:
            print(f"  FAILED to load: {exc}\n")
            exit_code = 1
            continue

        for label, text, expected in cases:
            t0 = time.time()
            prob = det.score_document(text)
            got = verdict(prob)
            line = f"  {label:22} p(AI)={prob:.4f}  -> {got:10} ({time.time() - t0:.2f}s)"
            if expected:
                agree = got == expected or (got == "borderline" and expected == "ai")
                line += f"  expected={expected}  {'AGREE' if agree else 'DISAGREE'}"
                if not agree:
                    exit_code = 1
            print(line)

        if args.sentences and cases:
            sents = split_sentences(strip_markdown(cases[0][1]))
            probs = det.score_sentences(sents)
            print("\n  highest-scoring sentences (these get rewritten first):")
            for s, p in sorted(zip(sents, probs), key=lambda x: -x[1])[:8]:
                print(f"    {p:.3f}  {s[:76]}")
        print()

    if args.expect and exit_code:
        print("At least one model DISAGREED with your detector on this text.")
        print("A disagreeing model is the wrong guidance detector: the adversarial")
        print("pass would optimise a target your real detector does not share.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
