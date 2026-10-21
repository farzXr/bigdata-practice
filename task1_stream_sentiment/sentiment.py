"""Определение тональности русского текста готовой моделью с HuggingFace.

Модель cointegrated/rubert-tiny-sentiment-balanced (≈ 30 МБ) уже обучена
на русскоязычных отзывах и комментариях — дообучение не требуется.
При первом запуске веса скачиваются автоматически и кэшируются.
"""
from functools import lru_cache

import config

LABELS = ("negative", "neutral", "positive")


@lru_cache(maxsize=1)
def _pipeline():
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=config.MODEL_NAME,
        truncation=True,
        max_length=512,
    )


def classify(texts: list[str]) -> list[tuple[str, float]]:
    """Возвращает [(метка, уверенность), ...] для каждого текста."""
    if not texts:
        return []
    preds = _pipeline()(texts, batch_size=32)
    return [(p["label"].lower(), float(p["score"])) for p in preds]
