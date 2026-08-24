"""Runtime configuration, all overridable by environment variables."""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Auth -------------------------------------------------------------
    # Callers must send this in the X-API-Key header. Empty disables auth,
    # which is only sane behind a private network.
    app_api_key: str = ""

    # --- Anthropic --------------------------------------------------------
    anthropic_api_key: str = ""
    # Accepts either MODEL or ANTHROPIC_MODEL; the latter is easier to find in a
    # long EasyPanel env list. A request may override this per call.
    model: str = Field(
        default="claude-opus-5",
        validation_alias=AliasChoices("MODEL", "ANTHROPIC_MODEL", "model"),
    )
    # low | medium | high | xhigh | max. Style rewriting benefits from real
    # reasoning about sentence rhythm, so we do not go below "high".
    effort: str = Field(
        default="high", validation_alias=AliasChoices("EFFORT", "ANTHROPIC_EFFORT", "effort")
    )
    max_tokens: int = 16000
    # Server-side refusal fallback: if the primary model declines, the API
    # re-runs the request on a fallback inside the same call.
    enable_refusal_fallback: bool = True
    request_timeout_seconds: float = 600.0

    # --- Pipeline ---------------------------------------------------------
    # Target composite AI-likelihood score (0-100, lower is more human).
    target_ai_score: float = 10.0
    # Rewrite attempts per chunk before we accept the best result we have.
    max_attempts: int = 3
    # Editing aggressiveness: standard | aggressive | max.
    #   standard   - one structure-preserving rewrite pass.
    #   aggressive - one pass, every section edited hard, extra human-texture
    #                instructions, target lowered.
    #   max        - aggressive plus a second "texture" pass that injects human
    #                irregularity into the already-rewritten text.
    # This helps most against perplexity/burstiness checkers; gains against
    # trained classifiers (Copyleaks, TruthScan) are modest.
    strength: str = "standard"
    # Full pipeline passes. Overrides the count implied by `strength` when set.
    # Capped at 3: more passes mostly add cost and fact-drift risk.
    passes: int = 0  # 0 = derive from strength

    # --- Guidance detector -------------------------------------------------
    # The optimisation target for the adversarial pass. "local" runs a real
    # neural detector on CPU and is the only setting that meaningfully moves
    # trained classifiers (GPTZero, Copyleaks); "none" falls back to the
    # stylometric proxy, which does not.
    #   local | none
    guidance_detector: str = "local"
    # Must be trained on modern LLM output. A GPT-2-era detector teaches the
    # pipeline nothing about current model prose.
    guidance_model: str = "desklib/ai-text-detector-v1.01"
    guidance_batch_size: int = 16
    guidance_threads: int = 0  # 0 = let torch decide
    # Load the weights at startup rather than on the first request.
    guidance_warmup: bool = False

    # --- Adversarial pass --------------------------------------------------
    # Detector-guided sentence substitution, run after the rewrite passes.
    enable_adversarial: bool = True
    adversarial_rounds: int = 3
    adversarial_candidates: int = 4
    adversarial_sentences_per_round: int = 12
    # Stop once the guidance detector is at or below this probability (0-1).
    adversarial_target_probability: float = 0.25
    # Only substitute sentences the detector is at least this suspicious of.
    adversarial_min_sentence_probability: float = 0.35
    # Words per chunk sent to the model. Smaller chunks give the model more
    # attention per sentence; larger chunks give it more rhythmic context.
    chunk_target_words: int = 700
    chunk_max_words: int = 1000
    # Deterministic regex clean-up after the model pass.
    enable_postprocess: bool = True
    # How many chunks to rewrite concurrently.
    concurrency: int = 3

    # --- Server -----------------------------------------------------------
    port: int = 8000
    log_level: str = "info"


@lru_cache
def get_settings() -> Settings:
    return Settings()
