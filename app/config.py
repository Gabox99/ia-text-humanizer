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
