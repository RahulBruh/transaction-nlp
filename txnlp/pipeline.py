"""End-to-end categorization with human-review escalation.

1. Try to resolve the descriptor to a known merchant. A confident match takes
   the directory's category (source="directory").
2. Otherwise ask the classifier (source="model").
3. If the classifier's top probability is below the review threshold, the
   result is flagged needs_review instead of being trusted silently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .clean import clean
from .data import CATEGORIES


@dataclass
class Result:
    descriptor: str
    cleaned: str
    category: str
    confidence: float
    source: str  # "directory" | "model"
    merchant: str | None
    needs_review: bool

    def dict(self):
        return asdict(self)


class Categorizer:
    def __init__(self, classifier, resolver=None, match_threshold=0.9, review_threshold=0.6, method="fuzzy"):
        self.classifier = classifier
        self.resolver = resolver
        self.match_threshold = match_threshold
        self.review_threshold = review_threshold
        self.method = method

    def categorize(self, descriptors: list[str]) -> list[Result]:
        matches = self.resolver.resolve(descriptors, self.method) if self.resolver else [None] * len(descriptors)
        proba = self.classifier.predict_proba(descriptors)
        results = []
        for d, match, p in zip(descriptors, matches, proba):
            if match is not None and match.score >= self.match_threshold:
                results.append(Result(d, clean(d), match.merchant.category, round(match.score, 4), "directory", match.merchant.name, False))
                continue
            k = int(np.argmax(p))
            conf = float(p[k])
            results.append(Result(d, clean(d), CATEGORIES[k], round(conf, 4), "model", None, conf < self.review_threshold))
        return results


def coverage_at(proba: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Accuracy on auto-accepted rows and the share sent to review at a threshold."""
    conf = proba.max(axis=1)
    accepted = conf >= threshold
    pred = proba.argmax(axis=1)
    return {
        "threshold": threshold,
        "auto_accept_rate": float(accepted.mean()),
        "accuracy_on_accepted": float((pred[accepted] == y[accepted]).mean()) if accepted.any() else None,
        "review_rate": float(1 - accepted.mean()),
    }
