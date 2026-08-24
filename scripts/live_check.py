#!/usr/bin/env python
"""End-to-end check against the real Anthropic API.

    python scripts/live_check.py                      # built-in 2000-word sample
    python scripts/live_check.py article.md            # your own file
    python scripts/live_check.py article.md --lang pt-BR

Prints the before/after scores, the per-section trace, cost, and writes the
result next to the input as `<name>.humanized.md` so you can paste it straight
into whichever AI detector gates your work. The internal score is a proxy —
that paste is the measurement that actually counts.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.schemas import HumanizeRequest  # noqa: E402

SAMPLE = """# Unlocking the Transformative Power of Predictive Maintenance

In today's rapidly evolving industrial landscape, predictive maintenance has emerged as a
pivotal component of operational excellence. Organizations across various sectors are
leveraging robust analytical frameworks to unlock unprecedented insights into equipment
health. Furthermore, the seamless integration of sensor telemetry with machine learning
models facilitates comprehensive decision-making processes that were previously
unattainable.

## Understanding the Fundamentals

It is important to note that predictive maintenance differs fundamentally from both
reactive and preventive approaches. Moreover, organizations that harness these
capabilities can streamline their maintenance operations significantly. Additionally, the
intricate nature of modern industrial equipment underscores the crucial role of
meticulous data collection strategies.

Traditional preventive maintenance relies on fixed schedules. Predictive maintenance, by
contrast, delves into real-time condition monitoring. Consequently, maintenance teams can
navigate complex operational demands with greater precision. This transformative shift
represents a game-changing opportunity for forward-thinking enterprises.

## Key Benefits of Implementation

Organizations implementing predictive maintenance typically observe substantial
improvements across multiple dimensions. Notably, unplanned downtime decreases
considerably. Similarly, maintenance costs decline as unnecessary interventions are
eliminated. Ultimately, equipment lifespan extends meaningfully.

- Reduced unplanned downtime through early anomaly detection
- Lower maintenance costs by eliminating unnecessary scheduled interventions
- Extended asset lifespan resulting from timely, targeted repairs
- Improved safety outcomes as failure modes are identified proactively

## The Importance of Data Quality

It is worth noting that data quality plays a crucial role in determining outcomes.
Therefore, organizations must ensure that their sensor infrastructure delivers reliable,
comprehensive telemetry. Nevertheless, many enterprises embark on predictive maintenance
initiatives without adequately addressing foundational data concerns.

Sensor placement requires meticulous planning. Sampling frequency must align with the
failure modes being monitored. Additionally, historical failure data is invaluable for
model training, yet many organizations lack sufficiently detailed records.

## Navigating Common Implementation Challenges

Organizations frequently encounter obstacles when implementing these systems. First and
foremost, cultural resistance often emerges among maintenance personnel accustomed to
established workflows. Furthermore, the technical complexity of integrating disparate
data sources presents significant hurdles.

Change management is essential. Maintenance technicians possess invaluable domain
expertise, and their buy-in is crucial for sustained success. Consequently, organizations
should foster collaborative relationships between data science teams and frontline
maintenance staff.

## Conclusion

In conclusion, predictive maintenance represents a transformative opportunity for
industrial organizations seeking to elevate their operational performance. By leveraging
robust analytical capabilities and fostering a data-driven culture, enterprises can
unlock substantial value. Ultimately, the organizations that thrive will be those that
navigate this transition with meticulous planning and unwavering commitment.
"""


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="Markdown file to humanize")
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--attempts", type=int, default=None)
    ap.add_argument("--tone", default=None)
    args = ap.parse_args()

    settings = get_settings()
    if not (settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")):
        print("ANTHROPIC_API_KEY is not set. Export it or put it in .env.", file=sys.stderr)
        return 2

    if args.file:
        src = Path(args.file)
        text = src.read_text(encoding="utf-8-sig")
        out_path = src.with_suffix(".humanized.md")
    else:
        text = SAMPLE
        out_path = Path("sample.humanized.md")

    req = HumanizeRequest(
        text=text,
        language=args.lang,
        target_ai_score=args.target,
        max_attempts=args.attempts,
        tone=args.tone,
    )

    started = time.monotonic()
    result = await Pipeline(settings).run(req)
    elapsed = time.monotonic() - started

    b, a = result.metrics_before, result.metrics
    print(f"\nmodel        {result.model}   effort={settings.effort}")
    print(f"elapsed      {elapsed:.1f}s   api calls {result.usage.api_calls}")
    print(f"cost         ${result.usage.estimated_cost_usd:.4f}"
          f"  (cache read {result.usage.cache_read_input_tokens} tok)")
    print(f"\nAI score     {b.ai_score:.1f} ({b.verdict})  ->  "
          f"{a.ai_score:.1f} ({a.verdict})   target {result.target_ai_score}"
          f"   {'MET' if result.target_met else 'NOT MET'}")
    print(f"words        {b.words} -> {a.words}")
    print(f"burstiness   {b.sentence_length_cv:.2f} -> {a.sentence_length_cv:.2f}"
          f"   (human band 0.45-0.75)")
    print(f"tells/1k     {b.tell_density_per_1k:.1f} -> {a.tell_density_per_1k:.1f}")
    print(f"em dash/1k   {b.em_dashes_per_1k:.1f} -> {a.em_dashes_per_1k:.1f}")
    print(f"transitions  {b.opening_transition_rate:.2f} -> {a.opening_transition_rate:.2f}")
    print(f"MATTR        {b.mattr:.2f} -> {a.mattr:.2f}")

    if a.penalties:
        print("\nremaining penalties:")
        for k, v in sorted(a.penalties.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<22} +{v:.1f}")

    print("\nper-section:")
    for t in result.chunks:
        head = (t.heading or "(no heading)")[:46]
        print(f"  [{t.index}] {head:<48} {t.intensity:<9} "
              f"{t.ai_score_before:5.1f} -> {t.ai_score_after:5.1f}  "
              f"({t.attempts} attempt(s))")

    if result.warnings:
        print("\nwarnings:")
        for w in result.warnings:
            print("  -", w)

    out_path.write_text(f"# {result.title}\n\n{result.content}", encoding="utf-8")
    print(f"\ntitle:   {result.title}")
    print(f"written: {out_path}")
    print("\nNext step: paste that file into your AI detector. The score above is a")
    print("stylometric proxy, not a detector verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
