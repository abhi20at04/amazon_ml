"""Pairwise similarity features and precision-oriented threshold selection."""

from __future__ import annotations

import math
from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np

from normalization import normalize, token_set, token_set_normalized


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def _containment(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


@lru_cache(maxsize=50000)
def _char_ngrams(text: str, n: int = 3) -> frozenset[str]:
    compact = text.replace(" ", "")
    if len(compact) < n:
        return frozenset((compact,)) if compact else frozenset()
    return frozenset(compact[i:i + n] for i in range(len(compact) - n + 1))


def pair_features(source: dict[str, str], target: dict[str, str], *, normalized: bool = False) -> list[float]:
    """Return a compact numeric vector using only supplied record fields."""
    if normalized:
        name_a, name_b = source.get("business_name", ""), target.get("business_name", "")
        addr_a, addr_b = source.get("business_address", ""), target.get("business_address", "")
        country_a, country_b = source.get("country", ""), target.get("country", "")
        name_ta, name_tb = token_set_normalized(name_a), token_set_normalized(name_b)
        addr_ta, addr_tb = token_set_normalized(addr_a), token_set_normalized(addr_b)
    else:
        name_a, name_b = normalize(source.get("business_name", "")), normalize(target.get("business_name", ""))
        addr_a, addr_b = normalize(source.get("business_address", "")), normalize(target.get("business_address", ""))
        country_a, country_b = normalize(source.get("country", "")), normalize(target.get("country", ""))
        name_ta, name_tb = token_set(name_a), token_set(name_b)
        addr_ta, addr_tb = token_set(addr_a), token_set(addr_b)
    name_ratio = SequenceMatcher(None, name_a, name_b, autojunk=False).ratio() if name_a and name_b else 0.0
    addr_ratio = SequenceMatcher(None, addr_a, addr_b, autojunk=False).ratio() if addr_a and addr_b else 0.0
    return [
        _jaccard(name_ta, name_tb),
        _containment(name_ta, name_tb),
        _jaccard(_char_ngrams(name_a), _char_ngrams(name_b)),
        name_ratio,
        _jaccard(addr_ta, addr_tb),
        _containment(addr_ta, addr_tb),
        _jaccard(_char_ngrams(addr_a), _char_ngrams(addr_b)),
        addr_ratio,
        float(bool(country_a and country_a == country_b)),
        float(bool(name_a and name_a == name_b)),
        float(bool(addr_a and addr_a == addr_b)),
        math.exp(-abs(len(name_a) - len(name_b)) / max(len(name_a), len(name_b), 1)),
    ]


def entity_fbeta(gold: set[str], predicted: set[str], beta: float = 0.5) -> float:
    if not gold:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    precision = len(gold & predicted) / len(predicted)
    recall = len(gold & predicted) / len(gold)
    beta2 = beta * beta
    denom = beta2 * precision + recall
    return (1 + beta2) * precision * recall / denom if denom else 0.0


def macro_fbeta(gold: dict[str, set[str]], predicted: dict[str, set[str]], beta: float = 0.5) -> float:
    if not gold:
        return 0.0
    return sum(entity_fbeta(value, predicted.get(key, set()), beta) for key, value in gold.items()) / len(gold)


def choose_threshold(gold: dict[str, set[str]], scored: dict[str, list[tuple[str, float]]]) -> tuple[float, float]:
    """Choose a probability cutoff maximizing per-entity macro F0.5."""
    probabilities = [probability for pairs in scored.values() for _, probability in pairs]
    quantiles = np.quantile(probabilities, np.linspace(0.5, 1.0, 101)).tolist() if probabilities else []
    thresholds = sorted({0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                         0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, *quantiles})
    best_threshold, best_score = 0.5, -1.0
    for threshold in thresholds:
        predicted = {entity: {candidate for candidate, probability in pairs if probability >= threshold}
                     for entity, pairs in scored.items()}
        score = macro_fbeta(gold, predicted)
        if score > best_score:
            best_threshold, best_score = threshold, score
    return best_threshold, best_score
