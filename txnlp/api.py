"""FastAPI service: POST /categorize with raw descriptors.

Run: uvicorn txnlp.api:app --reload
Set TXNLP_MODEL=distilbert to serve the fine-tuned model (default: baseline).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .classify import BaselineClassifier, DistilBertClassifier
from .data import MERCHANTS
from .pipeline import Categorizer
from .resolve import MerchantResolver

MODELS = Path(__file__).resolve().parents[1] / "models"

app = FastAPI(title="Transaction Categorizer")


class CategorizeRequest(BaseModel):
    descriptors: list[str] = Field(min_length=1, max_length=500, examples=[["SQ *BLUE BOTTLE COFFEE 4471 SAN FRANCISCO CA"]])


@lru_cache
def get_categorizer() -> Categorizer:
    kind = os.environ.get("TXNLP_MODEL", "baseline")
    clf = DistilBertClassifier(MODELS / "distilbert") if kind == "distilbert" else BaselineClassifier.load(MODELS / "baseline.joblib")
    # Fuzzy-only resolver keeps the service light; embeddings are evaluated offline
    return Categorizer(clf, MerchantResolver(MERCHANTS, use_embeddings=False), method="fuzzy")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/categorize")
def categorize(req: CategorizeRequest):
    results = get_categorizer().categorize(req.descriptors)
    return {"results": [r.dict() for r in results], "needs_review": sum(r.needs_review for r in results)}
