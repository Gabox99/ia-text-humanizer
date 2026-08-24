"""HTML round-trip tests, using the shape a CMS actually emits."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.html_bridge import (  # noqa: E402
    html_to_markdown,
    looks_like_html,
    markdown_to_html,
    normalise_literal_newlines,
)

# Double-encoded newlines and HTML tags: exactly what came out of the n8n node.
RAW = (
    "<p>Humanoid robot records used to look like staged lab tricks.</p>\\n"
    "<h2>Inside Beijing's five-day robot meet</h2>\\n"
    "<p>The games run at the scale of a <strong>proper</strong> sports meet. "
    'See <a href="https://example.com/x">the report</a> for detail.</p>\\n'
    "<ul><li>First item</li><li>Second item</li></ul>\\n"
    "<h3>Did a robot beat Usain Bolt?</h3>\\n"
    "<p>Yes, 9.39 seconds against 9.58.</p>"
)


def test_detects_html():
    assert looks_like_html(RAW)
    assert looks_like_html("<p>hello</p>")
    assert not looks_like_html("# A Markdown Title\n\nSome prose here.")


def test_literal_newlines_normalised():
    fixed, changed = normalise_literal_newlines("a\\nb")
    assert changed and fixed == "a\nb"
    # Real newlines already present: leave the text alone.
    untouched, changed2 = normalise_literal_newlines("a\nb")
    assert not changed2 and untouched == "a\nb"


def test_html_to_markdown_structure():
    doc = html_to_markdown(RAW)
    md = doc.markdown
    assert doc.had_literal_newlines
    lines = [ln for ln in md.split("\n") if ln.strip()]
    assert "## Inside Beijing's five-day robot meet" in lines
    assert "### Did a robot beat Usain Bolt?" in lines
    assert "- First item" in lines
    assert "- Second item" in lines
    assert "**proper**" in md
    assert "[the report](https://example.com/x)" in md
    # Tags must not survive into the prose the model and detector will see.
    assert "<p>" not in md and "<h2>" not in md and "<li>" not in md


def test_round_trip_back_to_html():
    doc = html_to_markdown(RAW)
    out = markdown_to_html(doc.markdown, doc.frozen)
    assert "<h2>Inside Beijing's five-day robot meet</h2>" in out
    assert "<h3>Did a robot beat Usain Bolt?</h3>" in out
    assert "<ul>" in out and out.count("<li>") == 2
    assert "<strong>proper</strong>" in out
    assert '<a href="https://example.com/x">the report</a>' in out
    assert out.count("<p>") == 3
    assert "9.39" in out and "9.58" in out


def test_pre_and_table_are_frozen_verbatim():
    src = (
        "<p>Before.</p>"
        "<pre><code>x = 1\ny = 2</code></pre>"
        "<table><tr><td>a</td><td>1</td></tr></table>"
        "<p>After.</p>"
    )
    doc = html_to_markdown(src)
    assert len(doc.frozen) == 2, doc.frozen
    out = markdown_to_html(doc.markdown, doc.frozen)
    assert "x = 1\ny = 2" in out
    assert "<table>" in out and "<td>1</td>" in out
    assert "<p>Before.</p>" in out and "<p>After.</p>" in out


def test_script_content_is_dropped():
    doc = html_to_markdown("<p>Keep.</p><script>alert('no')</script>")
    assert "alert" not in doc.markdown
    assert "Keep." in doc.markdown


def test_entities_decoded_once():
    doc = html_to_markdown("<p>Bolt&#39;s time &amp; the record</p>")
    assert "Bolt's time & the record" in doc.markdown
    out = markdown_to_html(doc.markdown, doc.frozen)
    # Re-escaped on the way out so the HTML stays valid.
    assert "&amp;" in out


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
