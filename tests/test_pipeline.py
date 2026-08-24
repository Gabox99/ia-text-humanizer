"""Pipeline tests with a stubbed model, so they run without an API key."""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.llm import Completion, Usage  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.schemas import HumanizeRequest  # noqa: E402

ARTICLE = """# Unlocking the Transformative Power of Data Analytics

In today's rapidly evolving business landscape, data analytics has become a pivotal
component of organizational success. Companies are leveraging robust frameworks to
unlock unprecedented insights. Furthermore, the seamless integration of these tools
facilitates comprehensive decision-making processes.

## The Importance of a Holistic Approach

It is important to note that a holistic approach delivers significant advantages.
Moreover, organizations that harness these capabilities can streamline operations.
Additionally, the intricate nature of modern data ecosystems underscores the crucial
role of meticulous planning.

- Accuracy matters for every downstream consumer
- Scalability determines whether the system survives growth
- See https://example.com/report for the full dataset

```python
df = load("sales.csv")
```

| Metric | Value |
|--------|-------|
| Rows   | 1200  |

## Conclusion

In conclusion, leveraging data analytics represents a game-changing opportunity for
enterprises seeking to elevate their competitive positioning.
"""

SECTION_RE = re.compile(r"<section>\n(.*?)\n</section>", re.DOTALL)


class EchoRewriter:
    """Returns the input unchanged. Exercises validation and the retry loop."""

    def __init__(self):
        self.calls = 0
        self.seen_models: list[str | None] = []
        self.seen_efforts: list[str | None] = []

    async def complete(self, system_blocks, user_message, model=None, effort=None):
        self.calls += 1
        self.seen_models.append(model)
        self.seen_efforts.append(effort)
        section = SECTION_RE.search(user_message).group(1)
        return Completion(
            text=section,
            usage=Usage(input_tokens=1000, output_tokens=500, api_calls=1),
        )


class BreakingRewriter(EchoRewriter):
    """Drops a heading and a link, so every attempt must be rejected."""

    async def complete(self, system_blocks, user_message, model=None, effort=None):
        self.calls += 1
        section = SECTION_RE.search(user_message).group(1)
        damaged = "\n\n".join(
            p for p in section.split("\n\n") if not p.lstrip().startswith("#")
        )
        return Completion(
            text=damaged or "gutted",
            usage=Usage(input_tokens=10, output_tokens=5, api_calls=1),
        )


def _settings(**kw) -> Settings:
    base = dict(
        anthropic_api_key="test",
        max_attempts=2,
        chunk_target_words=60,
        chunk_max_words=120,
        target_ai_score=10.0,
        concurrency=2,
    )
    base.update(kw)
    return Settings(**base)


def run(rewriter, req: HumanizeRequest, **kw):
    settings = _settings(**kw)
    return asyncio.run(Pipeline(settings, rewriter).run(req))


def test_structure_and_title_preserved():
    r = EchoRewriter()
    out = run(r, HumanizeRequest(text=ARTICLE, language="en-US"))

    assert not out.content.lstrip().startswith("# "), "H1 should be lifted into `title`"
    assert out.title and "Data Analytics" in out.title
    # Heading levels and count survive even though the text is reworded.
    heads = [ln for ln in out.content.splitlines() if ln.startswith("#")]
    assert len(heads) == 2, heads
    assert all(h.startswith("## ") for h in heads), heads
    # Frozen blocks come back intact.
    assert 'df = load("sales.csv")' in out.content
    assert "```python" in out.content
    assert "| Rows   | 1200  |" in out.content
    assert "https://example.com/report" in out.content
    assert out.content.count("- ") >= 3


def test_retry_loop_uses_full_budget_when_target_unmet():
    r = EchoRewriter()
    out = run(r, HumanizeRequest(text=ARTICLE, language="en-US"), max_attempts=3)
    # The echo cannot reach the target, so at least one chunk burns its budget.
    assert max(t.attempts for t in out.chunks) == 3, [t.attempts for t in out.chunks]
    assert r.calls == sum(t.attempts for t in out.chunks)
    assert not out.target_met
    assert any("above the target" in w for w in out.warnings)


def test_locked_headings_are_untouched():
    r = EchoRewriter()
    out = run(
        r,
        HumanizeRequest(text=ARTICLE, language="en-US", rewrite_headings=False),
    )
    assert "## The Importance of a Holistic Approach" in out.content
    assert "## Conclusion" in out.content


def test_article_agreement_repaired():
    r = EchoRewriter()
    out = run(r, HumanizeRequest(text=ARTICLE, language="en-US"))
    import re as _re
    bad = _re.findall(r"a\s+[aeiou]\w+", out.content)
    assert not bad, f"broken a/an agreement: {bad}"


def test_structural_damage_falls_back_to_original():
    r = BreakingRewriter()
    out = run(r, HumanizeRequest(text=ARTICLE, language="en-US"))
    assert "## The Importance of a Holistic Approach" in out.content
    assert any("damaged the structure" in w for w in out.warnings)
    assert all(t.accepted_attempt == -1 for t in out.chunks)


def test_supplied_title_is_humanized_and_lifted():
    r = EchoRewriter()
    out = run(
        r,
        HumanizeRequest(
            text="## Section One\n\nSome body text that is long enough to matter here.\n",
            title="My Provided Title",
            language="en-US",
        ),
    )
    assert out.title == "My Provided Title"
    assert "My Provided Title" not in out.content
    assert "## Section One" in out.content


def test_metrics_and_usage_reported():
    r = EchoRewriter()
    out = run(r, HumanizeRequest(text=ARTICLE, language="en-US"))
    assert out.metrics_before.ai_score > 20
    assert out.metrics.words > 100
    assert out.usage.api_calls == r.calls
    assert out.usage.estimated_cost_usd > 0
    assert out.model == "claude-opus-5"


def test_model_override_per_request():
    r = EchoRewriter()
    out = run(
        r,
        HumanizeRequest(
            text=ARTICLE, language="en-US", model="claude-sonnet-5", effort="low"
        ),
    )
    assert out.model == "claude-sonnet-5", out.model
    assert set(r.seen_models) == {"claude-sonnet-5"}
    assert set(r.seen_efforts) == {"low"}
    # Sonnet is cheaper than the Opus default, and the cost must reflect that.
    baseline = run(EchoRewriter(), HumanizeRequest(text=ARTICLE, language="en-US"))
    assert out.usage.estimated_cost_usd < baseline.usage.estimated_cost_usd


def test_model_override_defaults_to_env():
    r = EchoRewriter()
    out = run(r, HumanizeRequest(text=ARTICLE, language="en-US"))
    assert out.model == "claude-opus-5"
    # None is passed through so the Rewriter falls back to its own settings.
    assert set(r.seen_models) == {None}


def test_unknown_model_warns_about_cost():
    r = EchoRewriter()
    out = run(
        r, HumanizeRequest(text=ARTICLE, language="en-US", model="claude-future-9")
    )
    assert out.model == "claude-future-9"
    assert any("no published price" in w for w in out.warnings)


def test_non_claude_model_rejected():
    import pydantic

    try:
        HumanizeRequest(text=ARTICLE, model="gpt-5")
    except pydantic.ValidationError as e:
        assert "Claude model id" in str(e)
    else:
        raise AssertionError("a non-Claude model id should be rejected")


def test_generic_language_still_runs():
    r = EchoRewriter()
    out = run(r, HumanizeRequest(text=ARTICLE, language="de-DE"))
    assert out.title
    assert "## Conclusion" in out.content


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
