"""Markdown structure parsing, masking, chunking and structural validation.

The pipeline feeds raw Markdown to the model rather than isolated sentences,
because sentence rhythm is a document-level property and the model needs the
surrounding context to vary it convincingly. The price of that choice is that
the model *could* mangle the structure, so everything here exists to make that
failure detectable: we fingerprint the block skeleton before the rewrite and
compare it after.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FENCE_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)$")
TABLE_ROW_RE = re.compile(r"^[ \t]*\|.*\|[ \t]*$")
TABLE_SEP_RE = re.compile(r"^[ \t]*\|[\s:|-]+\|[ \t]*$")
HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})[ \t]+(?P<text>.+?)[ \t]*#*[ \t]*$")
SETEXT_H1_RE = re.compile(r"^=+[ \t]*$")
SETEXT_H2_RE = re.compile(r"^-{2,}[ \t]*$")
HR_RE = re.compile(r"^[ \t]*(?:\*[ \t]*){3,}$|^[ \t]*(?:-[ \t]*){3,}$|^[ \t]*(?:_[ \t]*){3,}$")
UL_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>[-*+])[ \t]+(?P<text>.*)$")
OL_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>\d{1,3}[.)])[ \t]+(?P<text>.*)$")
QUOTE_RE = re.compile(r"^[ \t]*>[ \t]?(?P<text>.*)$")
URL_RE = re.compile(r"https?://[^\s\)\]\>\"']+")

# ASCII-only sentinel: no encoding surprises in transit, and distinctive enough
# that no model will emit it by accident or "helpfully" reformat it.
MASK_TEMPLATE = "@@FROZEN-{kind}-{idx}@@"
MASK_RE = re.compile(r"@@FROZEN-(?P<kind>[A-Z]+)-(?P<idx>\d+)@@")


# --- Masking ---------------------------------------------------------------
# Fenced code and pipe tables are frozen before the model sees them.
# Reformatting a table or "improving" a code sample is never wanted here, and
# both are trivially destroyed by a well-meaning rewrite.


@dataclass
class MaskResult:
    text: str
    store: dict[str, str] = field(default_factory=dict)

    def restore(self, text: str) -> str:
        """Put the frozen blocks back, tolerating minor mangling of the token."""
        out = text
        for token, original in self.store.items():
            if token in out:
                out = out.replace(token, original)
                continue
            inner = token.strip("@")
            loose = re.compile(r"@@\s*" + re.escape(inner).replace(r"\-", r"\s*-\s*") + r"\s*@@",
                               re.IGNORECASE)
            out, n = loose.subn(lambda _m, o=original: o, out)
            if n == 0:
                # Token vanished entirely. Append rather than silently lose content.
                out = out.rstrip() + "\n\n" + original
        return out

    def missing(self, text: str) -> list[str]:
        return [t for t in self.store if t not in text]


def mask_verbatim(md: str) -> MaskResult:
    """Replace fenced code blocks and pipe tables with opaque placeholders."""
    lines = md.split("\n")
    out: list[str] = []
    store: dict[str, str] = {}
    counters = {"CODE": 0, "TABLE": 0}
    i = 0
    while i < len(lines):
        line = lines[i]
        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group("fence")[0] * 3
            buf = [line]
            i += 1
            while i < len(lines):
                buf.append(lines[i])
                closing = lines[i].strip().startswith(marker)
                i += 1
                if closing:
                    break
            token = MASK_TEMPLATE.format(kind="CODE", idx=counters["CODE"])
            counters["CODE"] += 1
            store[token] = "\n".join(buf)
            out.append(token)
            continue

        if TABLE_ROW_RE.match(line) and i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1]):
            buf = [line]
            i += 1
            while i < len(lines) and TABLE_ROW_RE.match(lines[i]):
                buf.append(lines[i])
                i += 1
            token = MASK_TEMPLATE.format(kind="TABLE", idx=counters["TABLE"])
            counters["TABLE"] += 1
            store[token] = "\n".join(buf)
            out.append(token)
            continue

        out.append(line)
        i += 1
    return MaskResult(text="\n".join(out), store=store)


# --- Block model -----------------------------------------------------------


@dataclass
class Block:
    kind: str  # heading | paragraph | list | quote | masked | hr
    raw: str
    level: int = 0  # heading level, or item count for lists
    text: str = ""  # prose content; headings, paragraphs, lists and quotes only

    @property
    def words(self) -> int:
        return len(self.text.split()) if self.text else 0

    @property
    def signature(self) -> str:
        if self.kind == "heading":
            return f"h{self.level}"
        if self.kind == "list":
            return f"list:{self.level}"
        if self.kind == "masked":
            m = MASK_RE.search(self.raw)
            return f"masked:{m.group('kind').lower()}" if m else "masked"
        return self.kind


def parse_blocks(md: str) -> list[Block]:
    """Split masked Markdown into logical blocks. Blank lines are separators."""
    lines = md.split("\n")
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        if MASK_RE.fullmatch(line.strip()):
            blocks.append(Block(kind="masked", raw=line.strip()))
            i += 1
            continue

        if HR_RE.match(line):
            blocks.append(Block(kind="hr", raw=line))
            i += 1
            continue

        h = HEADING_RE.match(line)
        if h:
            blocks.append(
                Block(
                    kind="heading",
                    raw=line.rstrip(),
                    level=len(h.group("hashes")),
                    text=h.group("text").strip(),
                )
            )
            i += 1
            continue

        # Setext heading: text on one line, ==== or ---- underneath.
        if i + 1 < len(lines):
            if SETEXT_H1_RE.match(lines[i + 1]):
                blocks.append(
                    Block(kind="heading", raw=f"# {line.strip()}", level=1, text=line.strip())
                )
                i += 2
                continue
            if SETEXT_H2_RE.match(lines[i + 1]) and not UL_RE.match(line):
                blocks.append(
                    Block(kind="heading", raw=f"## {line.strip()}", level=2, text=line.strip())
                )
                i += 2
                continue

        if UL_RE.match(line) or OL_RE.match(line):
            buf: list[str] = []
            count = 0
            while i < len(lines) and lines[i].strip():
                if UL_RE.match(lines[i]) or OL_RE.match(lines[i]):
                    count += 1
                buf.append(lines[i])
                i += 1
            texts = []
            for b in buf:
                m = UL_RE.match(b) or OL_RE.match(b)
                texts.append(m.group("text") if m else b.strip())
            blocks.append(Block(kind="list", raw="\n".join(buf), level=count,
                                text="\n".join(texts)))
            continue

        if QUOTE_RE.match(line):
            buf = []
            while i < len(lines) and QUOTE_RE.match(lines[i]):
                buf.append(lines[i])
                i += 1
            text = " ".join(QUOTE_RE.match(b).group("text").strip() for b in buf)
            blocks.append(Block(kind="quote", raw="\n".join(buf), text=text))
            continue

        buf = []
        while i < len(lines) and lines[i].strip():
            if buf and (HEADING_RE.match(lines[i]) or MASK_RE.fullmatch(lines[i].strip())):
                break
            buf.append(lines[i])
            i += 1
        raw = "\n".join(buf)
        blocks.append(Block(kind="paragraph", raw=raw, text=" ".join(b.strip() for b in buf)))
    return blocks


def skeleton(blocks: list[Block]) -> list[str]:
    return [b.signature for b in blocks]


def extract_title(blocks: list[Block]) -> tuple[str | None, int | None]:
    """Return (title, block index). Prefers the first H1, falls back to any heading."""
    for idx, b in enumerate(blocks):
        if b.kind == "heading" and b.level == 1:
            return b.text, idx
    for idx, b in enumerate(blocks):
        if b.kind == "heading":
            return b.text, idx
    return None, None


def prose_words(blocks: list[Block]) -> int:
    return sum(b.words for b in blocks if b.kind in {"paragraph", "list", "quote"})


def prose_text(blocks: list[Block]) -> str:
    """Prose only, for scoring. Headings and frozen blocks are not prose."""
    return "\n\n".join(b.text for b in blocks if b.kind in {"paragraph", "quote"} and b.text)


# --- Chunking --------------------------------------------------------------


@dataclass
class Chunk:
    index: int
    blocks: list[Block]
    heading: str | None
    context_before: str = ""

    @property
    def markdown(self) -> str:
        return blocks_to_markdown(self.blocks)

    @property
    def words(self) -> int:
        return prose_words(self.blocks)

    @property
    def skeleton(self) -> list[str]:
        return skeleton(self.blocks)


def blocks_to_markdown(blocks: list[Block]) -> str:
    return "\n\n".join(b.raw.rstrip() for b in blocks if b.raw.strip())


def chunk_blocks(blocks: list[Block], target_words: int, max_words: int) -> list[Chunk]:
    """Group blocks into rewrite units, preferring heading boundaries.

    A chunk that would exceed `max_words` is cut at the nearest block boundary,
    so one very long section never blows past the budget.
    """
    chunks: list[Chunk] = []
    current: list[Block] = []
    current_heading: str | None = None

    def flush() -> None:
        nonlocal current, current_heading
        if current:
            chunks.append(Chunk(index=len(chunks), blocks=current, heading=current_heading))
            current = []
            current_heading = None

    for b in blocks:
        starts_section = b.kind == "heading" and b.level <= 3
        if current and starts_section and prose_words(current) >= target_words:
            flush()
        elif current and prose_words(current) >= max_words:
            flush()
        if b.kind == "heading" and current_heading is None:
            current_heading = b.text
        current.append(b)
    flush()

    # Give each chunk the tail of the previous one so the model can match the
    # rhythm across the seam instead of restarting its cadence every chunk.
    for i, ch in enumerate(chunks):
        ch.index = i
        if i > 0:
            prev = [b for b in chunks[i - 1].blocks if b.kind == "paragraph"]
            if prev:
                ch.context_before = prev[-1].text[-600:]
    return chunks


# --- Structural validation -------------------------------------------------


@dataclass
class StructureIssue:
    kind: str
    detail: str


def validate_rewrite(
    original: str,
    rewritten: str,
    preserve_terms: list[str] | None = None,
    mask: MaskResult | None = None,
) -> list[StructureIssue]:
    """Compare a rewrite against its original for structural damage."""
    issues: list[StructureIssue] = []

    before = skeleton(parse_blocks(original))
    after = skeleton(parse_blocks(rewritten))
    if before != after:
        issues.append(
            StructureIssue(kind="skeleton", detail=f"block structure changed: {before} -> {after}")
        )

    lost_urls = set(URL_RE.findall(original)) - set(URL_RE.findall(rewritten))
    if lost_urls:
        issues.append(
            StructureIssue(kind="urls", detail="dropped links: " + ", ".join(sorted(lost_urls)[:5]))
        )

    if mask:
        gone = mask.missing(rewritten)
        if gone:
            issues.append(
                StructureIssue(kind="masked", detail=f"{len(gone)} frozen block(s) dropped")
            )

    lowered = rewritten.lower()
    for term in preserve_terms or []:
        if term and term.lower() in original.lower() and term.lower() not in lowered:
            issues.append(StructureIssue(kind="preserve_term", detail=f"lost term: {term}"))

    ow, rw = len(original.split()), len(rewritten.split())
    if ow and (rw < ow * 0.7 or rw > ow * 1.45):
        issues.append(
            StructureIssue(kind="length", detail=f"word count {ow} -> {rw} is out of tolerance")
        )

    # Numbers are facts. A humanizer must not change a stat, a year or a score.
    # Reported as a warning (not a rejecting issue) because a legitimate rewrite
    # may spell a number out ("9" -> "nine"), which would false-positive.
    before_nums = _numbers(original)
    after_nums = _numbers(rewritten)
    dropped = before_nums - after_nums
    added = after_nums - before_nums
    if dropped or added:
        bits = []
        if dropped:
            bits.append("dropped " + ", ".join(sorted(dropped)[:6]))
        if added:
            bits.append("introduced " + ", ".join(sorted(added)[:6]))
        issues.append(StructureIssue(kind="numbers", detail="; ".join(bits)))
    return issues


_NUMBER_RE = re.compile(r"\d[\d.,]*")


def _numbers(text: str) -> set[str]:
    """Numeric tokens, normalised so 9.39 and '9.39,' compare equal."""
    out: set[str] = set()
    for m in _NUMBER_RE.findall(text):
        token = m.strip(".,").replace(",", "")
        if token:
            out.add(token)
    return out
