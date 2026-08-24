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
SENTENCE_RE = re.compile(r"^(\d+)\. (.+)$", re.MULTILINE)


class EchoRewriter:
    """Returns the input unchanged. Exercises validation and the retry loop."""

    def __init__(self):
        self.calls = 0
        self.seen_models: list[str | None] = []
        self.seen_efforts: list[str | None] = []
        self.seen_systems: list[str] = []
        self.seen_messages: list[str] = []

    async def complete(self, system_blocks, user_message, model=None, effort=None):
        self.calls += 1
        self.seen_models.append(model)
        self.seen_efforts.append(effort)
        self.seen_systems.append(
            " ".join(b.get("text", "") for b in system_blocks) if system_blocks else ""
        )
        self.seen_messages.append(user_message)
        usage = Usage(input_tokens=1000, output_tokens=500, api_calls=1)

        # The adversarial pass uses a different prompt shape (numbered sentences
        # in, [1a]/[1b] candidates out). Echo each sentence back so the loop runs
        # its full mechanics and then finds no improvement.
        section = SECTION_RE.search(user_message)
        if section is None:
            lines = [
                f"[{m.group(1)}a] {m.group(2).strip()}"
                for m in SENTENCE_RE.finditer(user_message)
            ]
            return Completion(text="\n".join(lines), usage=usage)

        return Completion(text=section.group(1), usage=usage)


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
        enable_adversarial=False,
        guidance_detector="none",
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


def test_strength_max_runs_two_passes_with_texture():
    r = EchoRewriter()
    out = run(r, HumanizeRequest(text=ARTICLE, language="en-US", strength="max"))
    assert out.strength == "max"
    assert out.passes_run == 2
    assert len(out.score_trajectory) == 2
    # Chunks are reported for the last pass, all tagged pass index 1.
    assert all(getattr(c, "pass_", None) == 1 for c in out.chunks)
    # The second pass used the texture system prompt; the first did not.
    assert any("final human pass" in s for s in r.seen_systems), "texture prompt not used"
    assert any("already been humanized once" in m for m in r.seen_messages)
    # Structure still preserved after two passes.
    assert "## Conclusion" in out.content
    assert 'df = load("sales.csv")' in out.content


def test_aggressive_single_pass_lowers_target_and_adds_addendum():
    r = EchoRewriter()
    out = run(r, HumanizeRequest(text=ARTICLE, language="en-US", strength="aggressive"))
    assert out.strength == "aggressive"
    assert out.passes_run == 1
    assert out.target_ai_score <= 5.0, out.target_ai_score  # base 10 - 5
    assert any("AGGRESSIVE MODE" in m for m in r.seen_messages)


def test_explicit_passes_override_strength():
    r = EchoRewriter()
    out = run(r, HumanizeRequest(text=ARTICLE, language="en-US", strength="standard", passes=3))
    assert out.passes_run == 3
    assert len(out.score_trajectory) == 3


def test_standard_still_single_pass_no_addendum():
    r = EchoRewriter()
    out = run(r, HumanizeRequest(text=ARTICLE, language="en-US"))
    assert out.strength == "standard"
    assert out.passes_run == 1
    assert not any("AGGRESSIVE MODE" in m for m in r.seen_messages)
    assert not any("final human pass" in s for s in r.seen_systems)


def test_adversarial_pass_runs_and_reports():
    r = EchoRewriter()
    out = run(r, HumanizeRequest(text=ARTICLE, language="en-US"), enable_adversarial=True)
    assert out.adversarial is not None
    # With GUIDANCE_DETECTOR unset the pass falls back to the proxy, and the
    # response must say so rather than let a proxy number pass for a detector one.
    assert out.adversarial.neural is False
    assert out.adversarial.detector == "stylometric-proxy"
    assert any("stylometric proxy, not a neural detector" in w for w in out.warnings)
    # Structure survives the extra pass.
    assert "## Conclusion" in out.content
    assert 'df = load("sales.csv")' in out.content
    assert "https://example.com/report" in out.content


def test_adversarial_can_be_disabled_per_request():
    r = EchoRewriter()
    out = run(
        r,
        HumanizeRequest(text=ARTICLE, language="en-US", adversarial=False),
        enable_adversarial=True,
    )
    assert out.adversarial is None


HTML_ARTICLE = (
    "<p>In today's landscape, teams are leveraging robust frameworks that deliver "
    "comprehensive outcomes for every stakeholder involved.</p>\\n"
    "<h2>The Importance of a Holistic Approach</h2>\\n"
    "<p>It is important to note that a holistic approach delivers significant "
    'advantages. See <a href="https://example.com/report">the report</a> for detail.</p>\\n'
    "<ul><li>Accuracy matters downstream</li><li>Scalability decides survival</li></ul>\\n"
    "<h3>Did throughput reach 1200 units?</h3>\\n"
    "<p>Yes, the warehouse moved 1200 units last quarter across all lines.</p>"
)


def test_html_in_html_out_with_structure_intact():
    r = EchoRewriter()
    out = run(r, HumanizeRequest(text=HTML_ARTICLE, language="en-US"))

    assert out.format == "html", out.format
    # Comes back as HTML, not Markdown.
    assert "<p>" in out.content
    assert "##" not in out.content and "**" not in out.content
    # Heading levels preserved at the right depth.
    assert "<h2>" in out.content and "<h3>" in out.content
    # List and link survive the round trip.
    assert out.content.count("<li>") == 2
    assert 'href="https://example.com/report"' in out.content
    # Facts survive.
    assert "1200" in out.content
    # The double-encoded newlines were detected and reported.
    assert any("literal" in w for w in out.warnings)


def test_markdown_input_still_returns_markdown():
    r = EchoRewriter()
    out = run(r, HumanizeRequest(text=ARTICLE, language="en-US"))
    assert out.format == "markdown"
    assert "<p>" not in out.content
    assert "## Conclusion" in out.content


def test_format_can_be_forced_to_markdown_on_html_input():
    r = EchoRewriter()
    out = run(
        r, HumanizeRequest(text=HTML_ARTICLE, language="en-US", format="markdown")
    )
    assert out.format == "markdown"
    # Forced Markdown mode leaves the tags as literal text rather than parsing them.
    assert "<p>" in out.content


def test_number_change_warns_but_does_not_reject():
    # A rewriter that mangles a statistic should not be rejected (numbers may be
    # spelled out legitimately) but must surface a spot-check warning.
    class NumberBreaker(EchoRewriter):
        async def complete(self, system_blocks, user_message, model=None, effort=None):
            self.calls += 1
            section = SECTION_RE.search(user_message).group(1)
            return Completion(
                text=section.replace("1200", "1500"),
                usage=Usage(input_tokens=10, output_tokens=5, api_calls=1),
            )

    r = NumberBreaker()
    art = "# T\n\nThe warehouse held 1200 units across the floor last year, in total.\n"
    out = run(r, HumanizeRequest(text=art, language="en-US"))
    assert any("numbers may have changed" in w for w in out.warnings), out.warnings
    assert "1500" in out.content  # accepted despite the change


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
