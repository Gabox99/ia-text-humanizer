"""Deterministic clean-up applied after the model pass.

The model handles everything that needs judgement. This layer handles the
mechanical tells it reliably reintroduces anyway — em dashes, semicolons, a
stray "leverage", an opening "Furthermore" — because a regex is free and does
not need a second API call.

Two design rules keep it from doing damage:

* Nothing runs on protected spans. Inline code, link targets, bare URLs, HTML
  tags and frozen placeholders are lifted out first and put back afterwards.
* Nothing runs at 100%. Purging every single instance of a common word produces
  a distribution no human writer has, which is itself detectable. Rates are
  drawn from a seeded RNG so a given input always yields the same output.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from app.scoring.metrics import LanguagePack

# Spans the transforms must never touch.
_PROTECT_RE = re.compile(
    r"@@FROZEN-[A-Z]+-\d+@@"        # frozen code/table sentinels
    r"|`[^`\n]*`"                   # inline code
    r"|\]\([^)\s]*(?:\s+\"[^\"]*\")?\)"  # markdown link/image target
    r"|https?://[^\s\)\]]+"         # bare URL
    r"|<[^>\n]{1,80}>"              # html tag or autolink
    r"|^[ \t]{0,3}#{1,6}[ \t]"      # heading marker itself
    # `[ \t]*$` rather than `\s*$`: in MULTILINE mode a greedy `\s*` swallows the
    # blank line after the table, which silently merges two paragraphs.
    r"|^[ \t]*\|.*\|[ \t]*$",       # any surviving table row
    re.MULTILINE,
)

_PROTECT_TOKEN = "\x00P{}\x00"

_HEADING_LINE_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t].*$", re.MULTILINE)

# Words that must not be auto-capitalised when they land at a sentence start.
_KEEP_LOWER = {
    "iphone", "ipad", "ios", "macos", "npm", "pytest", "eBay", "ebay", "xml", "html",
}


@dataclass
class PostprocessReport:
    changes: dict[str, int] = field(default_factory=dict)

    def bump(self, key: str, n: int = 1) -> None:
        if n:
            self.changes[key] = self.changes.get(key, 0) + n


def _protect(text: str, protect_headings: bool = False) -> tuple[str, list[str]]:
    store: list[str] = []

    def sub(m: re.Match[str]) -> str:
        store.append(m.group(0))
        return _PROTECT_TOKEN.format(len(store) - 1)

    # Whole heading lines first, when the caller froze them: otherwise the
    # narrower "heading marker" rule leaves the heading text exposed.
    if protect_headings:
        text = _HEADING_LINE_RE.sub(sub, text)
    return _PROTECT_RE.sub(sub, text), store


def _unprotect(text: str, store: list[str]) -> str:
    for i, original in enumerate(store):
        text = text.replace(_PROTECT_TOKEN.format(i), original)
    return text


def _match_case(original: str, replacement: str) -> str:
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _fix_sentence_caps(text: str) -> str:
    """Re-capitalise sentence starts left lowercase by a deletion."""

    def cap(m: re.Match[str]) -> str:
        lead, word = m.group(1), m.group(2)
        if word.lower() in _KEEP_LOWER:
            return m.group(0)
        return lead + word[0].upper() + word[1:]

    # `[a-z']*` and not `{1,}`: a sentence can legitimately start with a
    # one-letter word, and "a end-to-end approach" was staying lowercase.
    text = re.sub(r"((?:^|[.!?]\s+|\n\n))([a-z][a-z']*)", cap, text)
    return text


# Vowel *letter*, consonant *sound*, and the reverse. Swapping a word behind an
# article is the most common way a substitution breaks grammar: "a holistic
# approach" becomes "a end-to-end approach" unless the article is fixed too.
_AN_EXCEPTIONS = re.compile(
    r"^(?:one|once|uni|eu|ubiqu|use|user|using|usual|utili|ukrain)", re.IGNORECASE
)
# Acronyms read as letters ("an SQL query") need no entry here: the swap-to-"a"
# pattern only matches lowercase initials, so uppercase is never touched.
_A_EXCEPTIONS = re.compile(r"^(?:hour|honest|honor|honour|heir|herb\b)", re.IGNORECASE)


def _fix_articles(text: str, report: PostprocessReport) -> str:
    """Repair a/an agreement broken by a word substitution. English only."""
    n = 0

    def swap(m: re.Match[str], target: str, exceptions: re.Pattern[str]) -> str:
        nonlocal n
        article, gap, word = m.group(1), m.group(2), m.group(3)
        if exceptions.match(word):
            return m.group(0)
        n += 1
        fixed = target.capitalize() if article[0].isupper() else target
        return fixed + gap + word

    # "a end-to-end" -> "an end-to-end"
    text = re.sub(
        r"\b([Aa])(\s+)([aeiouAEIOU][\w-]*)",
        lambda m: swap(m, "an", _AN_EXCEPTIONS),
        text,
    )
    # "an solid" -> "a solid"
    text = re.sub(
        r"\b([Aa]n)(\s+)([bcdfgjklmnpqrstvwxyz][\w-]*)",
        lambda m: swap(m, "a", _A_EXCEPTIONS),
        text,
    )
    report.bump("articles_fixed", n)
    return text


def _tidy_spacing(text: str) -> str:
    # Only collapse runs of spaces that follow visible text: leading whitespace
    # is list indentation and flattening it would break nested lists.
    text = re.sub(r"(?<=\S)[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:])(?=[^\s\d])", r"\1 ", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\.\s*\.(?!\.)", ".", text)
    text = re.sub(r"^[ \t]*[,;]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    return text


# --- Individual transforms -------------------------------------------------


def _normalise_punctuation(text: str, report: PostprocessReport) -> str:
    """Typographic characters that models emit and most humans do not type."""
    subs = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "…": "...", " ": " ", "–": "-",
    }
    for bad, good in subs.items():
        n = text.count(bad)
        if n:
            text = text.replace(bad, good)
            report.bump("typographic_chars", n)
    return text


_DASH_RE = re.compile(r"[ \t]*(?:—|--)[ \t]*")


def _enforce_em_dash_budget(text: str, report: PostprocessReport) -> str:
    """At most one em dash per 300 words; the rest become commas or periods.

    Dashes are handled a sentence at a time, because a sentence with two of them
    is a parenthetical pair. Dropping only one leaves "the design - which is
    fine, works", which is worse than either keeping both or dropping both.
    """
    words = len(re.findall(r"\w+", text))
    budget = max(1, words // 300)
    removed = 0
    kept = 0
    out: list[str] = []

    # Keep the sentence delimiters so the text reassembles byte-identically.
    for piece in re.split(r"((?<=[.!?])[ \t]*\n?[ \t]*)", text):
        found = list(_DASH_RE.finditer(piece))
        if not found or kept + len(found) <= budget:
            kept += len(found)
            out.append(piece)
            continue

        if len(found) == 2:
            # Parenthetical pair: comma on both sides reads as intended.
            piece = _DASH_RE.sub(", ", piece)
            removed += 2
        else:
            rebuilt: list[str] = []
            cursor = 0
            for i, m in enumerate(found):
                rebuilt.append(piece[cursor : m.start()])
                after = piece[m.end() : m.end() + 1]
                # A period only works if what follows can open a sentence.
                if i % 3 == 2 and after.isalpha():
                    rebuilt.append(". " + after.upper())
                    cursor = m.end() + 1
                else:
                    rebuilt.append(", ")
                    cursor = m.end()
                removed += 1
            rebuilt.append(piece[cursor:])
            piece = "".join(rebuilt)
        out.append(piece)

    report.bump("em_dashes_removed", removed)
    return "".join(out)


def _reduce_semicolons(text: str, rng: random.Random, report: PostprocessReport) -> str:
    """Convert most semicolons to periods, keeping the occasional one."""
    n = 0

    def sub(m: re.Match[str]) -> str:
        nonlocal n
        if rng.random() > 0.85:  # ~15% survive
            return m.group(0)
        n += 1
        nxt = m.group(1)
        return ". " + nxt.upper()

    text = re.sub(r";\s+([a-zA-Z])", sub, text)
    report.bump("semicolons_converted", n)
    return text


def _delete_phrases(text: str, pack: LanguagePack, report: PostprocessReport) -> str:
    for phrase in pack.deletable:
        pattern = re.compile(re.escape(phrase) + r"\s*", re.IGNORECASE)
        text, n = pattern.subn("", text)
        report.bump("filler_deleted", n)
    return text


def _replace_phrases(text: str, pack: LanguagePack, report: PostprocessReport) -> str:
    # Longest first, so "in order to" is not eaten by a shorter overlapping key.
    for phrase in sorted(pack.phrase_replacements, key=len, reverse=True):
        replacement = pack.phrase_replacements[phrase]
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)

        def sub(m: re.Match[str], r: str = replacement) -> str:
            if not r:
                return ""
            return _match_case(m.group(0), r)

        text, n = pattern.subn(sub, text)
        report.bump("phrases_replaced", n)
    return text


def _replace_tells(
    text: str,
    table: dict[str, tuple[str, ...]],
    rate: float,
    rng: random.Random,
    report: PostprocessReport,
    key: str,
) -> str:
    """Swap flagged vocabulary at `rate`, cycling through the alternatives."""
    total = 0
    for word, options in table.items():
        if not options:
            continue
        pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)

        def sub(m: re.Match[str], opts: tuple[str, ...] = options) -> str:
            nonlocal total
            if rng.random() > rate:
                return m.group(0)
            total += 1
            return _match_case(m.group(0), opts[rng.randrange(len(opts))])

        text = pattern.sub(sub, text)
    report.bump(key, total)
    return text


def _strip_opening_transitions(
    text: str, pack: LanguagePack, rng: random.Random, report: PostprocessReport
) -> str:
    """Drop most sentence-initial connectives, keeping a small residue."""
    if not pack.transitions:
        return text
    alt = "|".join(re.escape(t) for t in sorted(pack.transitions, key=len, reverse=True))
    pattern = re.compile(
        rf"(^|(?<=[.!?])\s+|(?<=\n\n))({alt})\s*,\s*(\w)",
        re.IGNORECASE | re.MULTILINE,
    )
    n = 0

    def sub(m: re.Match[str]) -> str:
        nonlocal n
        if rng.random() > 0.8:  # ~20% survive
            return m.group(0)
        n += 1
        return m.group(1) + m.group(3).upper()

    text = pattern.sub(sub, text)
    report.bump("transitions_stripped", n)
    return text


def _strip_assistant_register(
    text: str, pack: LanguagePack, report: PostprocessReport
) -> str:
    """Remove whole sentences containing chat-assistant phrasing."""
    if not pack.register:
        return text
    n = 0
    out_paragraphs = []
    for para in text.split("\n\n"):
        if re.match(r"^\s*(?:#{1,6}\s|[-*+]\s|\d{1,3}[.)]\s|>|@@FROZEN)", para):
            out_paragraphs.append(para)
            continue
        pieces = re.split(r"(?<=[.!?])\s+", para)
        kept = []
        for piece in pieces:
            low = piece.lower()
            if any(phrase in low for phrase in pack.register) and len(pieces) > 1:
                n += 1
                continue
            kept.append(piece)
        out_paragraphs.append(" ".join(kept) if kept else para)
    report.bump("assistant_sentences_removed", n)
    return "\n\n".join(out_paragraphs)


def _inject_contractions(
    text: str, pack: LanguagePack, rng: random.Random, report: PostprocessReport
) -> str:
    if not pack.contractions or not pack.score_contractions:
        return text
    n = 0
    for full, short in pack.contractions.items():
        pattern = re.compile(r"\b" + re.escape(full) + r"\b", re.IGNORECASE)

        def sub(m: re.Match[str], s: str = short) -> str:
            nonlocal n
            if rng.random() > 0.55:  # leave plenty uncontracted
                return m.group(0)
            n += 1
            return _match_case(m.group(0), s)

        text = pattern.sub(sub, text)
    report.bump("contractions_added", n)
    return text


def _drop_wrapup_openers(text: str, report: PostprocessReport) -> str:
    """Kill leftover 'In conclusion,' style paragraph openers."""
    pattern = re.compile(
        r"(^|\n\n)\s*(?:In conclusion|In summary|To summarize|To sum up|Em conclus[aã]o"
        r"|Em resumo|Concluindo|Em suma)\s*,?\s*",
        re.IGNORECASE,
    )
    text, n = pattern.subn(lambda m: m.group(1), text)
    report.bump("wrapup_openers_removed", n)
    return text


# --- Entry point -----------------------------------------------------------


def postprocess(
    text: str,
    pack: LanguagePack,
    seed: str = "",
    protect_headings: bool = False,
) -> tuple[str, PostprocessReport]:
    """Run the full deterministic pass. Same input, same seed, same output.

    `protect_headings` freezes whole heading lines, for callers that asked for
    headings to be left exactly as written. Without it a heading is prose like
    any other and "A Holistic Approach" becomes "A Whole-picture Approach".
    """
    report = PostprocessReport()
    rng = random.Random(seed or text[:256])

    body, protected = _protect(text, protect_headings=protect_headings)

    body = _normalise_punctuation(body, report)
    body = _strip_assistant_register(body, pack, report)
    body = _drop_wrapup_openers(body, report)
    body = _delete_phrases(body, pack, report)
    body = _replace_phrases(body, pack, report)
    # Connectives are stripped before the vocabulary swap: otherwise "Moreover,"
    # becomes "Also," and the stripper no longer recognises it.
    body = _strip_opening_transitions(body, pack, rng, report)
    body = _replace_tells(body, pack.hard_tells, 0.92, rng, report, "hard_tells_replaced")
    body = _replace_tells(body, pack.soft_tells, 0.55, rng, report, "soft_tells_replaced")
    body = _enforce_em_dash_budget(body, report)
    body = _reduce_semicolons(body, rng, report)
    body = _inject_contractions(body, pack, rng, report)
    body = _tidy_spacing(body)
    body = _fix_sentence_caps(body)
    if pack.code == "en":
        body = _fix_articles(body, report)

    return _unprotect(body, protected), report
