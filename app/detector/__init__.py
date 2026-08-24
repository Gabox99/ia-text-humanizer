"""Guidance detector: the optimisation target for the adversarial rewrite loop."""

from __future__ import annotations

import logging
import threading

from app.config import Settings
from app.detector.base import DetectorReading, GuidanceDetector, SentenceScore

log = logging.getLogger(__name__)

_instance: GuidanceDetector | None = None
_lock = threading.Lock()


def get_detector(settings: Settings) -> GuidanceDetector:
    """Process-wide singleton. The model weights must load exactly once."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is not None:
            return _instance
        _instance = _build(settings)
        return _instance


def _build(settings: Settings) -> GuidanceDetector:
    from app.detector.local_model import LocalNeuralDetector, StylometricFallbackDetector

    mode = (settings.guidance_detector or "none").lower()
    if mode in {"none", "off", "disabled", ""}:
        log.info("Guidance detector disabled; using the stylometric proxy.")
        return StylometricFallbackDetector()

    try:
        detector = LocalNeuralDetector(
            model_id=settings.guidance_model,
            batch_size=settings.guidance_batch_size,
            threads=settings.guidance_threads or None,
        )
        if settings.guidance_warmup:
            detector.warmup()
        return detector
    except Exception as exc:
        # A missing torch install or an unreachable model hub must not take the
        # whole service down: degrade to the proxy and say so loudly.
        log.error(
            "Could not initialise the neural guidance detector (%s). "
            "Falling back to the stylometric proxy, which does NOT reflect "
            "trained-classifier scores.",
            exc,
        )
        return StylometricFallbackDetector()


def reset_detector() -> None:
    """Test hook."""
    global _instance
    with _lock:
        _instance = None


__all__ = [
    "DetectorReading",
    "GuidanceDetector",
    "SentenceScore",
    "get_detector",
    "reset_detector",
]
