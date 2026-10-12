import pytest
from fastapi.testclient import TestClient

from txnlp import api
from txnlp.classify import BaselineClassifier
from txnlp.data import MERCHANTS, generate, split_merchants
from txnlp.pipeline import Categorizer
from txnlp.resolve import MerchantResolver


@pytest.fixture(scope="module")
def client():
    seen, _ = split_merchants()
    rows = generate(seen, 10, seed=1)
    clf = BaselineClassifier().fit([r["descriptor"] for r in rows], [r["category"] for r in rows])
    original = api.get_categorizer
    api.get_categorizer = lambda: Categorizer(clf, MerchantResolver(MERCHANTS, use_embeddings=False))
    yield TestClient(api.app)
    api.get_categorizer = original


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_categorize(client):
    res = client.post("/categorize", json={"descriptors": ["SQ *BLUE BOTTLE COFFEE 4471 SAN FRANCISCO CA", "NETFLIX.COM*8HD72K"]})
    assert res.status_code == 200
    body = res.json()
    assert [r["category"] for r in body["results"]] == ["Dining", "Subscriptions"]
    assert {"confidence", "source", "needs_review", "cleaned"} <= body["results"][0].keys()


def test_rejects_empty_batch(client):
    assert client.post("/categorize", json={"descriptors": []}).status_code == 422
