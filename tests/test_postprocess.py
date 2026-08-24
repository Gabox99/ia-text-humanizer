import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rules.postprocess import postprocess  # noqa: E402
from app.scoring.metrics import analyse, pack_for  # noqa: E402

SRC = """## Leveraging Robust Frameworks

Furthermore, it is important to note that organizations must delve into a myriad of
options; this is crucial. Moreover, the seamless integration — which is transformative —
underscores the pivotal role of meticulous planning — and it is essential to ensure
compliance. In this article, we will explore these ideas. It is not a small thing.

- Use `npm install --save` in order to add the package
- See https://example.com/docs?a=1&b=2 and [the guide](https://x.io/g "Guide")
- Nested:
    - deeper item that should keep its indentation

@@FROZEN-CODE-0@@

| a | b |
|---|---|
| 1 | 2 |

In conclusion, this is a game-changer that will elevate outcomes.
"""


def run() -> tuple[str, dict]:
    pack = pack_for("en-US")
    return postprocess(SRC, pack, seed="fixed-seed")


def test_protected_spans_survive():
    out, _ = run()
    assert "@@FROZEN-CODE-0@@" in out
    assert "https://example.com/docs?a=1&b=2" in out
    assert "`npm install --save`" in out
    assert "[the guide](https://x.io/g \"Guide\")" in out
    assert "    - deeper item" in out, "nested list indentation was flattened"
    assert "| 1 | 2 |" in out


def test_no_word_phrase_collision():
    out, _ = run()
    assert "a many of" not in out
    assert "in order to" not in out
    # "in order to" must be rewritten to "to", not deleted outright.
    assert "to add the package" in out


def test_em_dash_budget_enforced():
    out, _ = run()
    assert out.count("—") <= 1


def test_sentence_capitalisation_repaired():
    out, _ = run()
    for para in out.split("\n\n"):
        stripped = para.strip()
        if not stripped or stripped.startswith(("#", "-", "|", "@@", ">")):
            continue
        assert stripped[0].isupper() or not stripped[0].isalpha(), (
            f"paragraph starts lowercase: {stripped[:60]!r}"
        )


def test_score_improves():
    out, _ = run()
    _, before, _ = analyse(SRC, "en-US")
    _, after, _ = analyse(out, "en-US")
    assert after.value < before.value


if __name__ == "__main__":
    out, report = run()
    print(out)
    print("---- changes:", report.changes)
    _, b, _ = analyse(SRC, "en-US")
    _, a, _ = analyse(out, "en-US")
    print(f"---- score {b.value} -> {a.value}")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
