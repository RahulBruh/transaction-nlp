import numpy as np
import pytest

from txnlp.classify import BaselineClassifier
from txnlp.data import CATEGORIES, MERCHANTS, generate, split_merchants
from txnlp.pipeline import Categorizer, coverage_at
from txnlp.resolve import MerchantResolver


@pytest.fixture(scope="module")
def resolver():
    return MerchantResolver(MERCHANTS, use_embeddings=False)


@pytest.fixture(scope="module")
def baseline():
    seen, _ = split_merchants()
    rows = generate(seen, 20, seed=1)
    return BaselineClassifier().fit([r["descriptor"] for r in rows], [r["category"] for r in rows])


def test_fuzzy_resolves_noisy_descriptors(resolver):
    rows = generate(MERCHANTS, 3, seed=9)
    matches = resolver.resolve([r["descriptor"] for r in rows], "fuzzy")
    acc = np.mean([m.merchant.name == r["merchant"] for m, r in zip(matches, rows)])
    assert acc > 0.85


def test_resolver_scores_are_bounded(resolver):
    s = resolver.scores(["SQ *BLUE BOTTLE COFFEE 4471"], "fuzzy")
    assert s.shape == (1, len(MERCHANTS)) and 0 <= s.min() and s.max() <= 1


def test_baseline_learns_seen_merchants(baseline):
    seen, _ = split_merchants()
    rows = generate(seen, 5, seed=2)
    proba = baseline.predict_proba([r["descriptor"] for r in rows])
    assert proba.shape == (len(rows), len(CATEGORIES))
    pred = [CATEGORIES[i] for i in proba.argmax(axis=1)]
    assert np.mean([p == r["category"] for p, r in zip(pred, rows)]) > 0.9


def test_known_merchant_uses_directory(baseline, resolver):
    r = Categorizer(baseline, resolver).categorize(["STARBUCKS STORE #1234 PHOENIX AZ"])[0]
    assert r.source == "directory" and r.merchant == "Starbucks" and r.category == "Dining"


def test_low_confidence_is_flagged_for_review(baseline):
    r = Categorizer(baseline, resolver=None, review_threshold=1.01).categorize(["XQZ 99"])[0]
    assert r.source == "model" and r.needs_review


def test_coverage_at():
    proba = np.array([[0.9, 0.1], [0.55, 0.45], [0.2, 0.8]])
    y = np.array([0, 1, 1])
    c = coverage_at(proba, y, 0.6)
    assert c["auto_accept_rate"] == pytest.approx(2 / 3)
    assert c["accuracy_on_accepted"] == 1.0
