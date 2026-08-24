"""Request and response contracts for the public API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HumanizeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description=(
            "Article body as Markdown. Headings (#, ##, ...), lists, blockquotes, "
            "tables and fenced code blocks are all recognised and preserved."
        ),
    )
    title: str | None = Field(
        None,
        description=(
            "Article title. If omitted, the first level-1 heading in `text` is used; "
            "if there is none, the first heading of any level is used."
        ),
    )
    language: str = Field(
        "en-US",
        description=(
            "BCP-47-ish language tag. 'en-US' and 'pt-BR' have curated AI-tell "
            "dictionaries; any other tag still works, handled by the model alone."
        ),
    )
    tone: str | None = Field(
        None,
        description="Free-form voice note, e.g. 'skeptical industry analyst, first person'.",
    )
    preserve_terms: list[str] = Field(
        default_factory=list,
        description="Terms that must survive verbatim: brand names, keywords, product names.",
    )
    model: str | None = Field(
        None,
        description=(
            "Override the model for this call, e.g. 'claude-sonnet-5'. Must be a "
            "claude-* id. Defaults to the MODEL environment variable."
        ),
    )
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = Field(
        None,
        description=(
            "Override reasoning effort for this call. Below 'high' the sentence-rhythm "
            "rules get followed loosely. Defaults to the EFFORT environment variable."
        ),
    )
    target_ai_score: float | None = Field(
        None,
        ge=0,
        le=100,
        description="Override the composite AI-likelihood target. Lower is stricter.",
    )
    max_attempts: int | None = Field(
        None, ge=1, le=6, description="Override the rewrite attempts per chunk."
    )
    strength: Literal["standard", "aggressive", "max"] | None = Field(
        None,
        description=(
            "Editing aggressiveness. 'standard' = one structure-preserving pass. "
            "'aggressive' = one hard pass with extra human-texture instructions and a "
            "lower target. 'max' = aggressive plus a second texture pass. Higher "
            "strength helps most against perplexity/burstiness checkers; gains against "
            "trained classifiers are modest. Defaults to the STRENGTH env var."
        ),
    )
    passes: int | None = Field(
        None,
        ge=1,
        le=3,
        description=(
            "Full pipeline passes, overriding the count implied by `strength`. Each "
            "extra pass compounds the rewrite and the cost, and raises fact-drift risk."
        ),
    )
    adversarial: bool | None = Field(
        None,
        description=(
            "Run the detector-guided adversarial pass after the rewrite. This is the "
            "stage that optimises a real neural detector rather than the surface proxy, "
            "and the only one that moves trained classifiers. Requires "
            "GUIDANCE_DETECTOR=local to be meaningful. Defaults to ENABLE_ADVERSARIAL."
        ),
    )
    adversarial_rounds: int | None = Field(
        None, ge=1, le=8, description="Max substitution rounds in the adversarial pass."
    )
    adversarial_target: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Stop the adversarial pass once the guidance detector reports this "
            "probability or lower (0.0-1.0). Default 0.25."
        ),
    )
    format: Literal["auto", "markdown", "html"] | None = Field(
        "auto",
        description=(
            "Input and output format. 'auto' detects HTML by its block tags and returns "
            "the same format it received. HTML is converted to Markdown internally so "
            "headings are treated as headings rather than prose, then rendered back."
        ),
    )
    rewrite_headings: bool = Field(
        True,
        description=(
            "Rewrite heading text too. Set false to freeze headings exactly as given "
            "(useful when they are SEO-locked)."
        ),
    )
    postprocess: bool | None = Field(
        None, description="Override the deterministic regex clean-up pass."
    )

    @field_validator("language")
    @classmethod
    def _normalise_language(cls, v: str) -> str:
        return v.strip() or "en-US"

    @field_validator("model")
    @classmethod
    def _check_model(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        # Deliberately not an allowlist: new Claude models ship faster than this
        # code changes. The prefix check just stops a typo or an OpenAI id from
        # turning into a confusing 502 from the Anthropic API.
        if not v.startswith("claude-"):
            raise ValueError("model must be a Claude model id, e.g. 'claude-sonnet-5'")
        return v


class MetricReport(BaseModel):
    """Raw stylometric measurements, plus the composite score derived from them."""

    ai_score: float = Field(..., description="Composite AI-likelihood, 0-100. Lower is more human.")
    verdict: Literal["human-like", "borderline", "ai-like"]
    words: int
    sentences: int
    mean_sentence_length: float
    sentence_length_cv: float = Field(
        ..., description="Burstiness proxy: stdev/mean of sentence length in words."
    )
    short_sentences_per_150w: float
    monotone_runs: int = Field(
        ..., description="Runs of 3+ consecutive sentences within 5 words of each other."
    )
    mattr: float = Field(..., description="Moving-average type-token ratio, window 100.")
    hapax_ratio: float
    em_dashes_per_1k: float
    semicolons_per_1k: float
    tell_density_per_1k: float
    opening_transition_rate: float
    contraction_rate: float
    paragraph_length_cv: float
    passive_rate: float
    penalties: dict[str, float] = Field(
        default_factory=dict,
        description="Per-signal contribution to ai_score. Use this to see what is still off.",
    )


class ChunkTrace(BaseModel):
    index: int
    heading: str | None
    attempts: int
    ai_score_before: float
    ai_score_after: float
    accepted_attempt: int
    intensity: str
    pass_: int = Field(0, alias="pass", serialization_alias="pass")

    model_config = {"populate_by_name": True}


class AdversarialReport(BaseModel):
    """Outcome of the detector-guided pass.

    `neural` is the field that matters: when it is false the numbers below come
    from the stylometric proxy and say nothing about what GPTZero or Copyleaks
    will report.
    """

    detector: str
    neural: bool = Field(
        ...,
        description=(
            "True when a real neural detector guided the pass. False means the "
            "stylometric proxy was used and these probabilities are not detector scores."
        ),
    )
    ai_probability_before: float
    ai_probability_after: float
    rounds_run: int
    substitutions: int
    trajectory: list[float] = Field(
        default_factory=list, description="Detector probability after each round."
    )


class UsageReport(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    api_calls: int = 0
    estimated_cost_usd: float = 0.0


class HumanizeResponse(BaseModel):
    title: str
    content: str
    language: str
    metrics: MetricReport
    metrics_before: MetricReport
    target_ai_score: float
    target_met: bool
    strength: str = "standard"
    format: str = "markdown"
    passes_run: int = 1
    score_trajectory: list[float] = Field(
        default_factory=list,
        description="Composite ai_score after each pass, oldest first.",
    )
    adversarial: AdversarialReport | None = None
    chunks: list[ChunkTrace]
    usage: UsageReport
    model: str
    warnings: list[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = "en-US"


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1)
    sentences: bool = Field(
        True, description="Include per-sentence attribution, highest score first."
    )
    limit: int = Field(15, ge=1, le=200, description="How many sentences to return.")


class DetectSentence(BaseModel):
    ai_probability: float
    text: str


class DetectResponse(BaseModel):
    detector: str
    neural: bool = Field(
        ...,
        description=(
            "True when a real neural detector answered. False means the stylometric "
            "proxy answered and this is not a detector score."
        ),
    )
    ai_probability: float = Field(..., description="0.0-1.0 for the whole document.")
    ai_percent: float
    verdict: str
    words: int
    sentences: list[DetectSentence] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    language: str
    metrics: MetricReport
    suggestions: list[str] = Field(
        default_factory=list,
        description="Human-readable notes on which signals are driving the score.",
    )
