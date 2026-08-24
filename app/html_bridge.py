"""HTML in, HTML out.

CMS pipelines hand you `<p>` and `<h2>`, not Markdown. Feeding that straight
into the Markdown parser makes every tag look like prose: headings stop being
headings, so structure preservation means nothing, the guidance detector scores
angle brackets, and the rewriter is invited to "improve" your markup.

So HTML is detected on the way in, converted to Markdown for the pipeline, and
converted back on the way out, matching the format the caller sent. The
round-trip is deliberately narrow — the tags a CMS article actually uses — and
anything unrecognised is frozen verbatim rather than guessed at.

Dependency-free on purpose: the standard library's HTMLParser handles this, and
pulling in bs4/lxml for a dozen tags is not worth the image size.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

# Enough of a signal to switch modes: a block-level tag on its own.
_HTML_HINT_RE = re.compile(
    r"<\s*(p|h[1-6]|div|ul|ol|li|blockquote|article|section|br)\b[^>]*>", re.IGNORECASE
)
# Literal backslash-n, which is what arrives when a JSON string was escaped twice.
_LITERAL_NEWLINE_RE = re.compile(r"\\r\\n|\\n|\\r")

_BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "li", "pre"}
_SKIP_CONTENT = {"script", "style"}


def looks_like_html(text: str) -> bool:
    return bool(_HTML_HINT_RE.search(text))


def normalise_literal_newlines(text: str) -> tuple[str, bool]:
    """Turn literal "\\n" sequences into real newlines.

    A double-encoded JSON payload delivers the two characters backslash and n
    instead of a line break. Left alone they end up inside sentences, and every
    downstream stage — sentence splitting, block parsing, the detector — reads
    them as words.
    """
    if "\n" in text or not _LITERAL_NEWLINE_RE.search(text):
        return text, False
    return _LITERAL_NEWLINE_RE.sub("\n", text), True


@dataclass
class HtmlDocument:
    """The parsed shape of an HTML article, plus what it takes to rebuild it."""

    markdown: str
    # Raw HTML for anything the converter would not round-trip safely.
    frozen: dict[str, str] = field(default_factory=dict)
    had_literal_newlines: bool = False


class _ToMarkdown(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.buffer: list[str] = []
        self.block: str | None = None
        self.list_stack: list[str] = []
        self.ordinal: list[int] = []
        self.skip_depth = 0
        self.frozen: dict[str, str] = {}
        self._freeze_tag: str | None = None
        self._freeze_buf: list[str] = []

    # --- helpers ----------------------------------------------------------

    def _flush(self) -> None:
        text = re.sub(r"[ \t]+", " ", "".join(self.buffer)).strip()
        self.buffer = []
        if not text:
            self.block = None
            return
        tag = self.block
        if tag and tag.startswith("h") and len(tag) == 2 and tag[1].isdigit():
            self.out.append("#" * int(tag[1]) + " " + text)
        elif tag == "blockquote":
            self.out.append("> " + text)
        elif tag == "li":
            if self.list_stack and self.list_stack[-1] == "ol":
                self.ordinal[-1] += 1
                self.out.append(f"{self.ordinal[-1]}. {text}")
            else:
                self.out.append(f"- {text}")
        else:
            self.out.append(text)
        self.block = None

    # --- parser callbacks -------------------------------------------------

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if self._freeze_tag:
            self._freeze_buf.append(self.get_starttag_text() or "")
            return
        if tag in _SKIP_CONTENT:
            self.skip_depth += 1
            return
        if tag == "pre":
            self._freeze_tag = "pre"
            self._freeze_buf = [self.get_starttag_text() or "<pre>"]
            return
        if tag == "table":
            self._freeze_tag = "table"
            self._freeze_buf = [self.get_starttag_text() or "<table>"]
            return

        if tag in ("ul", "ol"):
            self._flush()
            self.list_stack.append(tag)
            self.ordinal.append(0)
            return
        if tag in _BLOCK_TAGS:
            self._flush()
            self.block = tag
            return
        if tag == "br":
            self.buffer.append(" ")
            return
        if tag in ("strong", "b"):
            self.buffer.append("**")
        elif tag in ("em", "i"):
            self.buffer.append("*")
        elif tag == "code":
            self.buffer.append("`")
        elif tag == "a":
            href = dict((k.lower(), v or "") for k, v in attrs).get("href", "")
            self.buffer.append("[")
            self._href = href

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._freeze_tag:
            self._freeze_buf.append(f"</{tag}>")
            if tag == self._freeze_tag:
                token = f"@@FROZEN-HTML-{len(self.frozen)}@@"
                self.frozen[token] = "".join(self._freeze_buf)
                self._flush()
                self.out.append(token)
                self._freeze_tag = None
                self._freeze_buf = []
            return
        if tag in _SKIP_CONTENT:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if tag in ("ul", "ol"):
            self._flush()
            if self.list_stack:
                self.list_stack.pop()
                self.ordinal.pop()
            return
        if tag in _BLOCK_TAGS:
            self._flush()
            return
        if tag in ("strong", "b"):
            self.buffer.append("**")
        elif tag in ("em", "i"):
            self.buffer.append("*")
        elif tag == "code":
            self.buffer.append("`")
        elif tag == "a":
            self.buffer.append(f"]({getattr(self, '_href', '')})")

    def handle_data(self, data: str) -> None:
        if self._freeze_tag:
            self._freeze_buf.append(data)
            return
        if self.skip_depth:
            return
        self.buffer.append(data)

    def close(self):  # type: ignore[override]
        super().close()
        self._flush()
        return self


def html_to_markdown(source: str) -> HtmlDocument:
    text, unescaped = normalise_literal_newlines(source)
    parser = _ToMarkdown()
    parser.feed(text)
    parser.close()
    blocks = [b for b in parser.out if b.strip()]
    return HtmlDocument(
        markdown="\n\n".join(blocks),
        frozen=parser.frozen,
        had_literal_newlines=unescaped,
    )


# --- back to HTML ----------------------------------------------------------

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_UL = re.compile(r"^[-*+]\s+(.*)$")
_MD_OL = re.compile(r"^\d{1,3}[.)]\s+(.*)$")
_MD_QUOTE = re.compile(r"^>\s?(.*)$")
_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_MD_STRONG = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_EM = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", re.DOTALL)
_MD_CODE = re.compile(r"`([^`]+)`")
_FROZEN_HTML_RE = re.compile(r"@@FROZEN-HTML-\d+@@")


def _inline_to_html(text: str) -> str:
    out = html.escape(text, quote=False)
    # Unescape the markers the converter itself introduced, then map them back.
    out = out.replace("&lt;", "<").replace("&gt;", ">") if _FROZEN_HTML_RE.search(out) else out
    out = _MD_CODE.sub(lambda m: f"<code>{m.group(1)}</code>", out)
    out = _MD_STRONG.sub(lambda m: f"<strong>{m.group(1)}</strong>", out)
    out = _MD_EM.sub(lambda m: f"<em>{m.group(1)}</em>", out)
    out = _MD_LINK.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', out)
    return out


def markdown_to_html(markdown: str, frozen: dict[str, str] | None = None) -> str:
    frozen = frozen or {}
    lines_out: list[str] = []
    list_open: str | None = None

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            lines_out.append(f"</{list_open}>")
            list_open = None

    for block in re.split(r"\n\s*\n", markdown.strip()):
        block = block.strip()
        if not block:
            continue

        if block in frozen:
            close_list()
            lines_out.append(frozen[block])
            continue

        block_lines = block.split("\n")
        # A list block: every line is a bullet or a number.
        if all(_MD_UL.match(ln) or _MD_OL.match(ln) for ln in block_lines):
            kind = "ol" if _MD_OL.match(block_lines[0]) else "ul"
            if list_open != kind:
                close_list()
                lines_out.append(f"<{kind}>")
                list_open = kind
            for ln in block_lines:
                m = _MD_UL.match(ln) or _MD_OL.match(ln)
                lines_out.append(f"<li>{_inline_to_html(m.group(1))}</li>")
            continue

        close_list()
        joined = " ".join(ln.strip() for ln in block_lines)

        heading = _MD_HEADING.match(block_lines[0])
        if heading and len(block_lines) == 1:
            level = len(heading.group(1))
            lines_out.append(f"<h{level}>{_inline_to_html(heading.group(2))}</h{level}>")
            continue

        if all(_MD_QUOTE.match(ln) for ln in block_lines):
            inner = " ".join(_MD_QUOTE.match(ln).group(1) for ln in block_lines)
            lines_out.append(f"<blockquote><p>{_inline_to_html(inner)}</p></blockquote>")
            continue

        if _FROZEN_HTML_RE.fullmatch(joined):
            lines_out.append(frozen.get(joined, joined))
            continue

        lines_out.append(f"<p>{_inline_to_html(joined)}</p>")

    close_list()
    return "\n".join(lines_out)
