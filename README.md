# Transaction NLP: Merchant Matching & Spend Categorization

Turns messy card-statement lines into a canonical merchant and a spend category, and sends the predictions it isn't sure about to a human.

```
SQ *BLUE BOTTLE COFFEE 4471 SAN FRANCISCO CA  ->  Blue Bottle Coffee · Dining     (directory match)
SHAKE SHACK 0419 NEW YORK NY                  ->  Shake Shack · Dining            (directory match)
XQZ HOLDINGS 99                               ->  Utilities, confidence 0.30      needs_review
```

## How it works

1. **Data pipeline.** `txnlp/data.py` generates seeded synthetic descriptors for 80 merchants in 10 categories, with the noise real statements carry: processor prefixes (`SQ *`, `TST*`, `PAYPAL *`), order references, store numbers, city/state suffixes, typos and 18-25 character truncation. `txnlp/clean.py` normalizes them and derives 8 engineered features (processor type, store number, online signals, digit ratio, ...).
2. **Merchant resolution.** `txnlp/resolve.py` matches a descriptor against the merchant directory three ways: RapidFuzz string similarity, `all-MiniLM-L6-v2` sentence embeddings, and a hybrid blend.
3. **Category classification.** `txnlp/classify.py` compares a TF-IDF (word + char n-gram) + engineered-feature logistic regression baseline with `distilbert-base-uncased` fine-tuned for sequence classification.
4. **Escalation.** `txnlp/pipeline.py` takes the directory's category on a confident merchant match, otherwise uses the classifier, and flags anything below a confidence threshold as `needs_review`.
5. **Service.** `txnlp/api.py` exposes `POST /categorize` with FastAPI.

## Results

From `python scripts/train.py`; full tables and confusion matrices are in [reports/RESULTS.md](reports/RESULTS.md).

**Merchant resolution, top-1 accuracy over 80 merchants**

| method | standard noise | hard noise (2 typos, dropped word, heavy truncation) |
|---|---|---|
| RapidFuzz | 1.000 | **0.959** |
| Embeddings (MiniLM) | 0.974 | 0.578 |
| Hybrid | 1.000 | 0.924 |

General-purpose sentence embeddings break down on misspelled, truncated merchant codes, while character-level fuzzy matching holds up and is about 10x faster.

**Category classification.** The split is by merchant: 20 of the 80 merchants never appear in training.

| model | seen merchants | unseen merchants | train time (CPU) |
|---|---|---|---|
| TF-IDF + features -> logistic regression | 1.000 | 0.176 | < 1 s |
| DistilBERT fine-tuned (3 seeds) | 0.98-0.99 | **0.453 ± 0.031** | ~5 min / seed |

Both models memorize merchants they have seen. On new merchants, the baseline is barely above chance (0.10), while DistilBERT's pretrained language knowledge carries over: it gets every unseen Health merchant right (Rite Aid Pharmacy, Dental Care Clinic).

**Review routing (DistilBERT, unseen merchants).** At a 0.6 threshold, 60% of lines are auto-accepted at 66.6% accuracy. At 0.8, 39% are auto-accepted at 80.0% accuracy and the rest go to review.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements.txt
pytest -q                          # 21 tests
python scripts/train.py            # data, both models, reports (--skip-bert for a 10 s run)
uvicorn txnlp.api:app --reload     # TXNLP_MODEL=distilbert to serve the fine-tuned model
```

```bash
curl -X POST localhost:8000/categorize -H "content-type: application/json" \
  -d '{"descriptors": ["SQ *BLUE BOTTLE COFFEE 4471 SAN FRANCISCO CA"]}'
```

CI (GitHub Actions) runs the tests and the baseline training and evaluation on every push.

## Responsible AI

See [MODEL_CARD.md](MODEL_CARD.md) for intended use, data, limitations and escalation. In short: all data is synthetic, every result says where it came from and how confident it is, uncertain predictions go to a human, and the full report is reproducible from one seeded command.
