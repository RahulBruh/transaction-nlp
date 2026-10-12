"""Resolve a noisy descriptor to a canonical merchant in the directory.

Three strategies, compared in scripts/train.py:
  fuzzy     - RapidFuzz string similarity on cleaned text
  embedding - cosine similarity of sentence-transformer embeddings
  hybrid    - weighted blend of the two scores
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz, process

from .clean import clean
from .data import Merchant

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Match:
    merchant: Merchant
    score: float  # 0..1


class MerchantResolver:
    def __init__(self, merchants: list[Merchant], use_embeddings: bool = True, alpha: float = 0.5):
        self.merchants = merchants
        self.alpha = alpha
        # One entry per alias, so "AMZN MKTP US" and "AMAZON.COM" both point to Amazon
        self.alias_text = [clean(a) for m in merchants for a in m.aliases]
        self.alias_owner = [i for i, m in enumerate(merchants) for _ in m.aliases]
        self.encoder = None
        if use_embeddings:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(EMBED_MODEL)
            self.alias_vecs = self.encoder.encode(self.alias_text, normalize_embeddings=True)

    def _per_merchant(self, alias_scores: np.ndarray) -> np.ndarray:
        best = np.zeros(len(self.merchants))
        np.maximum.at(best, self.alias_owner, alias_scores)
        return best

    def fuzzy_scores(self, descriptors: list[str]) -> np.ndarray:
        queries = [clean(d) for d in descriptors]
        alias = process.cdist(queries, self.alias_text, scorer=fuzz.WRatio, workers=-1) / 100.0
        return np.stack([self._per_merchant(row) for row in alias])

    def embedding_scores(self, descriptors: list[str]) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("resolver was built without embeddings")
        q = self.encoder.encode([clean(d) for d in descriptors], normalize_embeddings=True)
        alias = np.clip(q @ self.alias_vecs.T, 0, 1)
        return np.stack([self._per_merchant(row) for row in alias])

    def scores(self, descriptors: list[str], method: str = "hybrid") -> np.ndarray:
        if method == "fuzzy":
            return self.fuzzy_scores(descriptors)
        if method == "embedding":
            return self.embedding_scores(descriptors)
        if method == "hybrid":
            return self.alpha * self.fuzzy_scores(descriptors) + (1 - self.alpha) * self.embedding_scores(descriptors)
        raise ValueError(f"unknown method {method!r}")

    def resolve(self, descriptors: list[str], method: str = "hybrid") -> list[Match]:
        s = self.scores(descriptors, method)
        idx = s.argmax(axis=1)
        return [Match(self.merchants[i], float(s[r, i])) for r, i in enumerate(idx)]
