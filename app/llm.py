"""Anthropic client wrapper.

Streaming is used for every call even though nothing streams to the caller:
`max_tokens` is high enough that a non-streaming request can hit the SDK's HTTP
timeout on a long section, and `.get_final_message()` gives back the whole
message anyway.
"""

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field

import anthropic

from app.config import Settings
from app.prompts.builder import REWRITE_CLOSE, REWRITE_OPEN

log = logging.getLogger(__name__)

# USD per million tokens, Anthropic first-party rates.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
}

_REWRITE_RE = re.compile(
    re.escape(REWRITE_OPEN) + r"\s*(.*?)\s*" + re.escape(REWRITE_CLOSE),
    re.DOTALL,
)


class ModelRefusal(RuntimeError):
    """The model, and any fallback, declined the request."""


class MissingCredentials(RuntimeError):
    """No Anthropic credential could be resolved."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    api_calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.api_calls += other.api_calls

    def cost_usd(self, model: str) -> float:
        rate_in, rate_out = PRICING.get(model, (5.00, 25.00))
        return round(
            (self.input_tokens * rate_in
             + self.cache_creation_input_tokens * rate_in * 1.25
             + self.cache_read_input_tokens * rate_in * 0.10
             + self.output_tokens * rate_out) / 1_000_000,
            6,
        )


@dataclass
class Completion:
    text: str
    usage: Usage = field(default_factory=Usage)
    fell_back_to: str | None = None


def extract_rewrite(raw: str) -> str:
    """Pull the payload out of the <rewrite> tags, tolerating a missing closer."""
    m = _REWRITE_RE.search(raw)
    if m:
        return m.group(1).strip()
    if REWRITE_OPEN in raw:
        return raw.split(REWRITE_OPEN, 1)[1].replace(REWRITE_CLOSE, "").strip()
    # No tags at all: strip any leading commentary line before the first heading
    # or paragraph and hope for the best. The structural validator catches the
    # cases where this guess is wrong.
    return raw.strip()


class Rewriter:
    def __init__(self, settings: Settings):
        self.settings = settings
        kwargs: dict = {"timeout": settings.request_timeout_seconds}
        if settings.anthropic_api_key:
            kwargs["api_key"] = settings.anthropic_api_key
        self.client = anthropic.AsyncAnthropic(**kwargs)

        # Decide fallback support from the installed SDK's signature rather than
        # from a caught TypeError: the SDK also raises a bare TypeError when no
        # credential resolves, and treating that as "beta unsupported" hides a
        # misconfigured key behind a misleading log line.
        supported = "fallbacks" in inspect.signature(self.client.beta.messages.stream).parameters
        if settings.enable_refusal_fallback and not supported:
            log.warning(
                "Installed anthropic SDK has no `fallbacks` parameter; "
                "refusal fallback disabled."
            )
        # Downgraded to False for the process if the account lacks the beta.
        self._fallbacks_supported = settings.enable_refusal_fallback and supported

    async def complete(
        self,
        system_blocks: list[dict],
        user_message: str,
        model: str | None = None,
        effort: str | None = None,
    ) -> Completion:
        params: dict = {
            "model": model or self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": system_blocks,
            "messages": [{"role": "user", "content": user_message}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort or self.settings.effort},
        }

        if self._fallbacks_supported:
            try:
                return await self._stream(
                    params
                    | {
                        "betas": ["server-side-fallback-2026-07-01"],
                        "fallbacks": "default",
                    },
                    beta=True,
                )
            except anthropic.BadRequestError as exc:
                log.warning(
                    "Server-side refusal fallback rejected (%s); continuing without it.",
                    exc,
                )
                self._fallbacks_supported = False

        return await self._stream(params, beta=False)

    async def _stream(self, params: dict, *, beta: bool) -> Completion:
        endpoint = self.client.beta.messages if beta else self.client.messages
        try:
            async with endpoint.stream(**params) as stream:
                message = await stream.get_final_message()
        except TypeError as exc:
            # The SDK signals "no credential resolved" with a bare TypeError.
            # Left unhandled it surfaces as an opaque HTTP 500.
            if "authentication" in str(exc).lower():
                raise MissingCredentials(
                    "No Anthropic credential resolved. Set ANTHROPIC_API_KEY."
                ) from exc
            raise

        usage = Usage(
            input_tokens=message.usage.input_tokens or 0,
            output_tokens=message.usage.output_tokens or 0,
            cache_read_input_tokens=getattr(message.usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(
                message.usage, "cache_creation_input_tokens", 0
            ) or 0,
            api_calls=1,
        )

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise ModelRefusal(f"model declined the rewrite (category={category})")

        text = "".join(b.text for b in message.content if b.type == "text")
        fell_back = None
        for block in message.content:
            if getattr(block, "type", None) == "fallback":
                fell_back = getattr(getattr(block, "to", None), "model", None)

        if message.stop_reason == "max_tokens":
            log.warning("Rewrite hit max_tokens; output may be truncated.")

        return Completion(text=extract_rewrite(text), usage=usage, fell_back_to=fell_back)
