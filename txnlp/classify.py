"""Spend-category classifiers.

BaselineClassifier   - word + char n-gram TF-IDF and engineered features -> logistic regression
DistilBertClassifier - distilbert-base-uncased fine-tuned for sequence classification
Both expose predict_proba(descriptors) -> (n, n_classes) in CATEGORIES order.
"""

from __future__ import annotations

import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from .clean import clean, features
from .data import CATEGORIES

LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}


def _cleaned(xs):
    return [clean(x) for x in xs]


def _engineered(xs):
    return [features(x) for x in xs]


class BaselineClassifier:
    def __init__(self):
        self.pipe = Pipeline([
            ("features", FeatureUnion([
                ("words", Pipeline([("clean", FunctionTransformer(_cleaned)), ("tfidf", TfidfVectorizer(ngram_range=(1, 2)))])),
                ("chars", Pipeline([("clean", FunctionTransformer(_cleaned)), ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True))])),
                ("engineered", Pipeline([("extract", FunctionTransformer(_engineered)), ("vec", DictVectorizer())])),
            ])),
            ("clf", LogisticRegression(C=10, max_iter=2000)),
        ])

    def fit(self, descriptors: list[str], labels: list[str]) -> "BaselineClassifier":
        self.pipe.fit(descriptors, [LABEL2ID[y] for y in labels])
        return self

    def predict_proba(self, descriptors: list[str]) -> np.ndarray:
        proba = np.zeros((len(descriptors), len(CATEGORIES)))
        proba[:, self.pipe.classes_] = self.pipe.predict_proba(descriptors)
        return proba

    def save(self, path: Path):
        joblib.dump(self.pipe, path)

    @classmethod
    def load(cls, path: Path) -> "BaselineClassifier":
        obj = cls.__new__(cls)
        obj.pipe = joblib.load(path)
        return obj


class DistilBertClassifier:
    BASE = "distilbert-base-uncased"
    MAX_LEN = 32

    def __init__(self, model_dir: str | Path | None = None, seed: int = 0):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(seed)  # the classification head is randomly initialized here
        src = str(model_dir) if model_dir else self.BASE
        self.tokenizer = AutoTokenizer.from_pretrained(src)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            src,
            num_labels=len(CATEGORIES),
            id2label=dict(enumerate(CATEGORIES)),
            label2id=LABEL2ID,
        )

    def _batch(self, descriptors: list[str]):
        return self.tokenizer(_cleaned(descriptors), padding=True, truncation=True, max_length=self.MAX_LEN, return_tensors="pt")

    def fit(self, descriptors, labels, epochs=3, batch_size=32, lr=5e-5, seed=0, log=print):
        import torch

        torch.manual_seed(seed)
        rng = random.Random(seed)
        y = [LABEL2ID[l] for l in labels]
        order = list(range(len(descriptors)))
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        steps = epochs * ((len(order) + batch_size - 1) // batch_size)
        sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=steps)
        self.model.train()
        for epoch in range(epochs):
            rng.shuffle(order)
            total = 0.0
            for i in range(0, len(order), batch_size):
                idx = order[i:i + batch_size]
                enc = self._batch([descriptors[j] for j in idx])
                out = self.model(**enc, labels=torch.tensor([y[j] for j in idx]))
                out.loss.backward()
                opt.step()
                sched.step()
                opt.zero_grad()
                total += out.loss.item() * len(idx)
            log(f"  epoch {epoch + 1}/{epochs}  loss {total / len(order):.4f}")
        self.model.eval()
        return self

    def predict_proba(self, descriptors: list[str], batch_size=128) -> np.ndarray:
        import torch

        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(descriptors), batch_size):
                logits = self.model(**self._batch(descriptors[i:i + batch_size])).logits
                out.append(torch.softmax(logits, dim=-1).numpy())
        return np.concatenate(out)

    def save(self, path: Path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
