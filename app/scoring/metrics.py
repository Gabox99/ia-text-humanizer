"""Stylometric measurement and the composite AI-likelihood score.

This is a *proxy*, not a detector. It measures the same surface signals that
perplexity/burstiness-class detectors measure, and it measures them cheaply and
offline, which is what makes a retry loop affordable. It cannot see what a
trained classifier sees in the model's RLHF fingerprint. Read the score as
"how many of the known surface tells are still present", and validate the final
output against whichever detector actually gates your work.

Every penalty is a linear ramp between a "clean" threshold and a "bad"
threshold, capped at the signal's weight, so the total is bounded and each
signal's contribution is legible in `penalties`.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field

from app.rules import tells_en, tells_pt

# --- Tokenisation ----------------------------------------------------------

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg", "ie",
    "inc", "ltd", "co", "fig", "no", "vol", "approx", "dept", "est", "min", "max",
    "sra", "srta", "av", "ex", "obs", "pág", "ref",
}

_SENT_END_RE = re.compile(r"([.!?]+)([\"')\]]*)(\s+|$)")
_WORD_RE = re.compile(r"[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'’-]*")
_MD_STRIP_RE = re.compile(
    r"!\[[^\]]*\]\([^)]*\)"          # images
    r"|\[([^\]]*)\]\([^)]*\)"        # links -> keep label
    r"|`[^`]*`"                      # inline code
    r"|@@FROZEN-[A-Z]+-\d+@@"        # our own sentinels
    r"|^\s{0,3}#{1,6}\s+"            # heading markers
    r"|[*_~]{1,3}"                   # emphasis
    r"|^\s*>\s?"                     # quote markers
    r"|^\s*[-*+]\s+"                 # bullets
    r"|^\s*\d{1,3}[.)]\s+",          # numbered
    re.MULTILINE,
)


def strip_markdown(text: str) -> str:
    """Reduce Markdown to plain prose so metrics measure writing, not syntax."""

    def _sub(m: re.Match[str]) -> str:
        return m.group(1) or ""

    return _MD_STRIP_RE.sub(_sub, text)


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter with abbreviation guards.

    Deliberately dependency-free: a spaCy model would add ~500MB to the image
    for a marginal gain on a signal that is already a rough proxy.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for m in _SENT_END_RE.finditer(text):
        end = m.end(2)
        candidate = text[start:end].strip()
        if not candidate:
            continue
        # "Dr." / "etc." / "1." are not sentence ends.
        tail = _WORD_RE.findall(candidate[-12:].lower())
        if tail and tail[-1] in _ABBREVIATIONS and m.group(1) == ".":
            continue
        if re.search(r"\b[A-Z]\.$", candidate):  # initials, e.g. "J. Smith"
            continue
        sentences.append(candidate)
        start = m.end()
    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def words_of(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def paragraphs_of(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


# --- Language packs --------------------------------------------------------


@dataclass
class LanguagePack:
    code: str
    hard_tells: dict[str, tuple[str, ...]]
    soft_tells: dict[str, tuple[str, ...]]
    deletable: tuple[str, ...]
    phrase_replacements: dict[str, str]
    transitions: tuple[str, ...]
    register: tuple[str, ...]
    contractions: dict[str, str]
    triple_re: str
    passive_aux: tuple[str, ...]
    nominal_suffixes: tuple[str, ...]
    # Contractions only count as a human signal in languages that have them.
    score_contractions: bool = True


_EN = LanguagePack(
    code="en",
    hard_tells=tells_en.HARD_TELLS,
    soft_tells=tells_en.SOFT_TELLS,
    deletable=tells_en.DELETABLE_PHRASES,
    phrase_replacements=tells_en.PHRASE_REPLACEMENTS,
    transitions=tells_en.OPENING_TRANSITIONS,
    register=tells_en.ASSISTANT_REGISTER,
    contractions=tells_en.CONTRACTIONS,
    triple_re=tells_en.TRIPLE_RE,
    passive_aux=tells_en.PASSIVE_AUX,
    nominal_suffixes=("tion", "sion", "ment", "ance", "ence", "ity", "ness", "ization",
                      "isation", "ability"),
    score_contractions=True,
)

_PT = LanguagePack(
    code="pt",
    hard_tells=tells_pt.HARD_TELLS,
    soft_tells=tells_pt.SOFT_TELLS,
    deletable=tells_pt.DELETABLE_PHRASES,
    phrase_replacements=tells_pt.PHRASE_REPLACEMENTS,
    transitions=tells_pt.OPENING_TRANSITIONS,
    register=tells_pt.ASSISTANT_REGISTER,
    contractions=tells_pt.CONTRACTIONS,
    triple_re=tells_pt.TRIPLE_RE,
    passive_aux=tells_pt.PASSIVE_AUX,
    nominal_suffixes=("ção", "ções", "mento", "mentos", "idade", "idades", "ância",
                      "ência", "ismo", "agem"),
    score_contractions=False,
)

# Any other language falls back to the structural signals only: burstiness,
# monotone runs, lexical diversity, punctuation and paragraph variance are all
# language-independent. The curated word lists simply do not apply.
_NEUTRAL = LanguagePack(
    code="xx",
    hard_tells={},
    soft_tells={},
    deletable=(),
    phrase_replacements={},
    transitions=(),
    register=(),
    contractions={},
    triple_re=r"(?!x)x",
    passive_aux=(),
    nominal_suffixes=(),
    score_contractions=False,
)


def pack_for(language: str) -> LanguagePack:
    code = (language or "").lower().replace("_", "-")
    if code.startswith("en"):
        return _EN
    if code.startswith("pt"):
        return _PT
    return _NEUTRAL


# --- Measurement -----------------------------------------------------------


@dataclass
class Measurements:
    words: int = 0
    sentences: int = 0
    mean_sentence_length: float = 0.0
    sentence_length_cv: float = 0.0
    short_sentences_per_150w: float = 0.0
    long_sentences_ratio: float = 0.0
    monotone_runs: int = 0
    mattr: float = 0.0
    hapax_ratio: float = 0.0
    em_dashes_per_1k: float = 0.0
    semicolons_per_1k: float = 0.0
    colons_mid_per_1k: float = 0.0
    tell_density_per_1k: float = 0.0
    hard_tell_hits: list[str] = field(default_factory=list)
    soft_tell_hits: list[str] = field(default_factory=list)
    register_hits: list[str] = field(default_factory=list)
    opening_transition_rate: float = 0.0
    transition_hits: list[str] = field(default_factory=list)
    contraction_rate: float = 0.0
    paragraph_length_cv: float = 0.0
    passive_rate: float = 0.0
    nominalization_per_1k: float = 0.0
    triples_per_1k: float = 0.0
    list_ratio: float = 0.0


def _cv(values: list[int | float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    if mean <= 0:
        return 0.0
    return statistics.stdev(values) / mean


def _mattr(tokens: list[str], window: int = 100) -> float:
    """Moving-average type-token ratio. Length-stable, unlike plain TTR."""
    if not tokens:
        return 0.0
    lowered = [t.lower() for t in tokens]
    if len(lowered) <= window:
        return len(set(lowered)) / len(lowered)
    ratios = [
        len(set(lowered[i : i + window])) / window
        for i in range(0, len(lowered) - window + 1, max(1, window // 4))
    ]
    return statistics.fmean(ratios)


def measure(text: str, language: str = "en-US") -> tuple[Measurements, LanguagePack]:
    pack = pack_for(language)
    plain = strip_markdown(text)
    lowered = plain.lower()
    sentences = split_sentences(plain)
    tokens = words_of(plain)
    m = Measurements()
    m.words = len(tokens)
    m.sentences = len(sentences)
    if not tokens:
        return m, pack

    per_1k = 1000.0 / m.words

    lengths = [len(words_of(s)) for s in sentences]
    lengths = [n for n in lengths if n > 0]
    if lengths:
        m.mean_sentence_length = statistics.fmean(lengths)
        m.sentence_length_cv = _cv(lengths)
        m.short_sentences_per_150w = sum(1 for n in lengths if n <= 6) * (150.0 / m.words)
        m.long_sentences_ratio = sum(1 for n in lengths if n >= 25) / len(lengths)
        # Runs of 3+ consecutive sentences all within 5 words of each other.
        run = 1
        for a, b in zip(lengths, lengths[1:]):
            if abs(a - b) <= 5:
                run += 1
            else:
                if run >= 3:
                    m.monotone_runs += 1
                run = 1
        if run >= 3:
            m.monotone_runs += 1

    m.mattr = _mattr(tokens)
    counts: dict[str, int] = {}
    for t in tokens:
        key = t.lower()
        counts[key] = counts.get(key, 0) + 1
    m.hapax_ratio = sum(1 for v in counts.values() if v == 1) / len(counts) if counts else 0.0

    m.em_dashes_per_1k = (plain.count("—") + plain.count("--")) * per_1k
    m.semicolons_per_1k = plain.count(";") * per_1k
    # Colons that sit mid-sentence rather than introducing a list.
    m.colons_mid_per_1k = len(re.findall(r"\w+:\s+[a-zà-ÿ]", plain)) * per_1k

    for word in pack.hard_tells:
        n = len(re.findall(r"\b" + re.escape(word) + r"\b", lowered))
        m.hard_tell_hits.extend([word] * n)
    for word in pack.soft_tells:
        n = len(re.findall(r"\b" + re.escape(word) + r"\b", lowered))
        m.soft_tell_hits.extend([word] * n)
    for phrase in pack.deletable:
        n = lowered.count(phrase)
        m.soft_tell_hits.extend([phrase] * n)
    # Hard tells count double: they are the higher-confidence signal.
    m.tell_density_per_1k = (len(m.hard_tell_hits) * 2 + len(m.soft_tell_hits)) * per_1k

    for phrase in pack.register:
        if phrase in lowered:
            m.register_hits.append(phrase)

    if sentences and pack.transitions:
        hits = 0
        for s in sentences:
            head = s.lstrip("\"'([ ").lower()
            for t in pack.transitions:
                if head.startswith(t) and (
                    len(head) == len(t) or head[len(t)] in ",; "
                ):
                    hits += 1
                    m.transition_hits.append(t)
                    break
        m.opening_transition_rate = hits / len(sentences)

    if pack.contractions:
        apostrophes = len(re.findall(r"\w['’](?:s|t|re|ve|ll|d|m)\b", lowered))
        m.contraction_rate = apostrophes * per_1k

    paras = paragraphs_of(plain)
    m.paragraph_length_cv = _cv([len(words_of(p)) for p in paras]) if len(paras) > 1 else 0.0

    if pack.passive_aux and sentences:
        aux = "|".join(re.escape(a) for a in pack.passive_aux)
        passive = len(re.findall(rf"\b({aux})\s+\w+(?:ed|do|da|to|ta|so|sa)\b", lowered))
        m.passive_rate = passive / len(sentences)

    if pack.nominal_suffixes:
        n = sum(
            1 for t in tokens if any(t.lower().endswith(s) for s in pack.nominal_suffixes)
        )
        m.nominalization_per_1k = n * per_1k

    m.triples_per_1k = len(re.findall(pack.triple_re, lowered)) * per_1k

    total_lines = [ln for ln in text.split("\n") if ln.strip()]
    if total_lines:
        bullets = sum(
            1 for ln in total_lines if re.match(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+", ln)
        )
        m.list_ratio = bullets / len(total_lines)

    return m, pack


# --- Scoring ---------------------------------------------------------------


def _ramp(value: float, clean: float, bad: float, weight: float) -> float:
    """Penalty that grows linearly from 0 at `clean` to `weight` at `bad`."""
    if clean == bad:
        return 0.0
    frac = (value - clean) / (bad - clean)
    return max(0.0, min(1.0, frac)) * weight


@dataclass
class Score:
    value: float
    penalties: dict[str, float]

    @property
    def verdict(self) -> str:
        if self.value <= 15:
            return "human-like"
        if self.value <= 35:
            return "borderline"
        return "ai-like"


# Human reference bands come from the sentence-variance and lexical-diversity
# literature cited in the README. Where sources disagree we take the looser
# bound, so the score errs toward "this still looks generated".
def score(m: Measurements, pack: LanguagePack) -> Score:
    p: dict[str, float] = {}

    # Burstiness. Human articles sit around 0.45-0.75 CV; AI clusters near 0.25.
    p["burstiness"] = _ramp(-m.sentence_length_cv, -0.50, -0.22, 18.0)
    # Runs of near-identical sentence lengths, normalised by document size.
    runs_per_1k = m.monotone_runs * (1000.0 / m.words) if m.words else 0.0
    p["monotone_runs"] = _ramp(runs_per_1k, 1.5, 8.0, 10.0)
    # At least one very short sentence per 150 words.
    p["short_sentences"] = _ramp(-m.short_sentences_per_150w, -1.0, -0.05, 8.0)

    # Lexical diversity: AI has lower MATTR and fewer hapax legomena.
    p["lexical_diversity"] = _ramp(-m.mattr, -0.78, -0.62, 8.0)
    p["hapax"] = _ramp(-m.hapax_ratio, -0.45, -0.28, 6.0)

    # Punctuation. One em dash per 300 words is the documented human ceiling.
    p["em_dashes"] = _ramp(m.em_dashes_per_1k, 3.3, 12.0, 8.0)
    p["semicolons"] = _ramp(m.semicolons_per_1k, 1.0, 6.0, 5.0)
    p["mid_colons"] = _ramp(m.colons_mid_per_1k, 1.5, 8.0, 3.0)

    # Vocabulary tells and chat-assistant register.
    p["ai_vocabulary"] = _ramp(m.tell_density_per_1k, 1.5, 14.0, 14.0)
    p["assistant_register"] = min(8.0, len(m.register_hits) * 4.0)

    # Discourse: sentence-initial connectives.
    p["opening_transitions"] = _ramp(m.opening_transition_rate, 0.06, 0.24, 10.0)

    # Register: missing contractions, only meaningful where the language has them.
    if pack.score_contractions:
        p["contractions"] = _ramp(-m.contraction_rate, -6.0, -0.5, 6.0)

    # Structural uniformity.
    p["paragraph_variance"] = _ramp(-m.paragraph_length_cv, -0.45, -0.15, 5.0)
    p["list_overuse"] = _ramp(m.list_ratio, 0.25, 0.60, 5.0)
    p["rule_of_three"] = _ramp(m.triples_per_1k, 1.0, 6.0, 4.0)
    if pack.nominal_suffixes:
        p["nominalization"] = _ramp(m.nominalization_per_1k, 35.0, 90.0, 5.0)
    if pack.passive_aux:
        p["passive_voice"] = _ramp(m.passive_rate, 0.18, 0.50, 4.0)

    p = {k: round(v, 2) for k, v in p.items() if v > 0.005}
    total = min(100.0, sum(p.values()))
    return Score(value=round(total, 2), penalties=p)


def analyse(text: str, language: str = "en-US") -> tuple[Measurements, Score, LanguagePack]:
    m, pack = measure(text, language)
    return m, score(m, pack), pack


# --- Feedback for the retry loop -------------------------------------------

# Each entry maps a penalty key to an imperative instruction. On a retry we
# inject only the instructions whose penalties actually fired, which is what
# makes the second attempt converge instead of shuffling the text at random.
_FEEDBACK: dict[str, str] = {
    "burstiness": (
        "Sentence lengths are too uniform. Break at least three long sentences into a "
        "long one plus a very short one, and merge two short ones somewhere else. Aim "
        "for a mix that ranges from 3 words to 35 words."
    ),
    "monotone_runs": (
        "There are stretches of consecutive sentences with near-identical length. Find "
        "them and change the length of every second sentence in those stretches."
    ),
    "short_sentences": (
        "Add more very short sentences (six words or fewer). At least one per 150 words, "
        "placed where a point lands, not evenly spaced."
    ),
    "lexical_diversity": (
        "Vocabulary repeats too much. Replace repeated nouns and verbs with different "
        "words rather than pronouns, and vary the words used to name the main subject."
    ),
    "hapax": (
        "Too few words appear only once. Reach for more specific, less generic word "
        "choices, especially for verbs."
    ),
    "em_dashes": "Too many em dashes. Keep at most one per 300 words; use commas, periods or parentheses.",
    "semicolons": "Too many semicolons. Convert nearly all of them into periods.",
    "mid_colons": "Too many mid-sentence colons used to restate. Rewrite those as full sentences.",
    "ai_vocabulary": (
        "AI-marked vocabulary is still present. Replace the flagged words with plainer, "
        "more specific alternatives. Do not swap one abstract word for another."
    ),
    "assistant_register": (
        "Chat-assistant phrasing is present (openers, closers, 'in this article we will'). "
        "Delete it entirely and start on the substance."
    ),
    "opening_transitions": (
        "Too many sentences open with a connective (Furthermore, Moreover, Therefore...). "
        "Delete most of them; let the sentences sit next to each other."
    ),
    "contractions": (
        "The register is too formal for the language. Use contractions where a person "
        "naturally would, unevenly rather than everywhere."
    ),
    "paragraph_variance": (
        "Paragraphs are all about the same length. Make some one sentence long and "
        "others substantially longer."
    ),
    "list_overuse": (
        "Too much of the text is bullet points. Convert at least half of the list items "
        "into flowing prose."
    ),
    "rule_of_three": (
        "Too many three-item parallel lists ('fast, cheap and reliable'). Cut some to "
        "two items, extend others to four, or rewrite as clauses."
    ),
    "nominalization": (
        "Too many abstract -tion/-ment/-ity nouns. Turn them back into verbs with real "
        "human subjects doing the action."
    ),
    "passive_voice": "Too much passive voice. Name who is doing the action.",
}


def metric_feedback(s: Score, m: Measurements, limit: int = 6) -> list[str]:
    """Imperative fixes for the worst-offending signals, biggest penalty first."""
    ordered = sorted(s.penalties.items(), key=lambda kv: kv[1], reverse=True)
    out: list[str] = []
    for key, _weight in ordered:
        text = _FEEDBACK.get(key)
        if not text:
            continue
        if key == "ai_vocabulary":
            flagged = sorted(set(m.hard_tell_hits + m.soft_tell_hits))[:14]
            if flagged:
                text += " Flagged here: " + ", ".join(flagged) + "."
        if key == "opening_transitions" and m.transition_hits:
            text += " Flagged here: " + ", ".join(sorted(set(m.transition_hits))[:10]) + "."
        if key == "assistant_register" and m.register_hits:
            text += " Flagged here: " + ", ".join(m.register_hits[:6]) + "."
        out.append(text)
        if len(out) >= limit:
            break
    return out


def suggestions(s: Score, m: Measurements) -> list[str]:
    """Human-readable diagnostics for the /analyze endpoint."""
    if not s.penalties:
        return ["No strong surface tells detected."]
    lines = []
    for key, weight in sorted(s.penalties.items(), key=lambda kv: kv[1], reverse=True):
        label = key.replace("_", " ")
        lines.append(f"{label}: +{weight:.1f} — {_FEEDBACK.get(key, 'elevated versus human baseline')}")
    return lines


def entropy_hint(text: str) -> float:
    """Shannon entropy over word unigrams. Exposed for debugging, not scored."""
    tokens = [t.lower() for t in words_of(strip_markdown(text))]
    if not tokens:
        return 0.0
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    n = len(tokens)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())
