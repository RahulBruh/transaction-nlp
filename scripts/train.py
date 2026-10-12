"""Generate data, train both classifiers, evaluate resolvers and classifiers.

Writes data/*.csv, models/, reports/metrics.json and reports/RESULTS.md.
Usage: python scripts/train.py [--skip-bert] [--epochs 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from txnlp.classify import LABEL2ID, BaselineClassifier, DistilBertClassifier  # noqa: E402
from txnlp.data import CATEGORIES, MERCHANTS, generate, split_merchants  # noqa: E402
from txnlp.pipeline import coverage_at  # noqa: E402
from txnlp.resolve import MerchantResolver  # noqa: E402

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def evaluate(name, proba, df):
    y = df["category"].map(LABEL2ID).to_numpy()
    pred = proba.argmax(axis=1)
    return {
        "model": name,
        "n": len(df),
        "accuracy": round(accuracy_score(y, pred), 4),
        "macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "coverage": [coverage_at(proba, y, t) for t in THRESHOLDS],
        "confusion": confusion_matrix(y, pred, labels=range(len(CATEGORIES))).tolist(),
        "report": classification_report(y, pred, labels=range(len(CATEGORIES)), target_names=CATEGORIES, zero_division=0, output_dict=True),
    }


def md_confusion(cm):
    head = "| true \\ pred | " + " | ".join(c[:5] for c in CATEGORIES) + " |"
    sep = "|---" * (len(CATEGORIES) + 1) + "|"
    rows = [f"| {c} | " + " | ".join(str(v) for v in row) + " |" for c, row in zip(CATEGORIES, cm)]
    return "\n".join([head, sep, *rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bert", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    for d in ("data", "models", "reports"):
        (ROOT / d).mkdir(exist_ok=True)

    seen, held = split_merchants()
    train = pd.DataFrame(generate(seen, per_merchant=40, seed=1))
    test_seen = pd.DataFrame(generate(seen, per_merchant=10, seed=2))
    test_unseen = pd.DataFrame(generate(held, per_merchant=25, seed=3))
    resolve_test = pd.DataFrame(generate(MERCHANTS, per_merchant=10, seed=4))
    resolve_hard = pd.DataFrame(generate(MERCHANTS, per_merchant=10, seed=5, hard=True))
    for name, df in [("train", train), ("test_seen", test_seen), ("test_unseen", test_unseen), ("resolve_test", resolve_test), ("resolve_hard", resolve_hard)]:
        df.to_csv(ROOT / "data" / f"{name}.csv", index=False)
    print(f"data: train {len(train)} | test seen-merchant {len(test_seen)} | test unseen-merchant {len(test_unseen)} | resolver {len(resolve_test)}")
    print(f"merchants: {len(seen)} seen, {len(held)} held out, {len(CATEGORIES)} categories")

    metrics = {"data": {"train": len(train), "test_seen": len(test_seen), "test_unseen": len(test_unseen), "resolve_test": len(resolve_test), "resolve_hard": len(resolve_hard),
                        "merchants_seen": len(seen), "merchants_held_out": len(held)}}

    # --- Merchant resolution -------------------------------------------------
    print("\n[resolver] merchant resolution, top-1 accuracy")
    resolver = MerchantResolver(MERCHANTS)
    metrics["resolver"] = {}
    for level, df in [("standard", resolve_test), ("hard", resolve_hard)]:
        truth = df["merchant"].to_numpy()
        descs = df["descriptor"].tolist()
        metrics["resolver"][level] = {}
        for method in ("fuzzy", "embedding", "hybrid"):
            t0 = time.perf_counter()
            matches = resolver.resolve(descs, method)
            ms = (time.perf_counter() - t0) * 1000 / len(descs)
            acc = float(np.mean([m.merchant.name == t for m, t in zip(matches, truth)]))
            metrics["resolver"][level][method] = {"top1_accuracy": round(acc, 4), "ms_per_descriptor": round(ms, 3)}
            print(f"  {level:8s} {method:9s} {acc:.3f}  ({ms:.2f} ms/descriptor)")

    # --- Category classification ---------------------------------------------
    metrics["classifier"] = {}
    print("\n[baseline] TF-IDF (word + char n-grams) + engineered features -> logistic regression")
    t0 = time.perf_counter()
    base = BaselineClassifier().fit(train["descriptor"].tolist(), train["category"].tolist())
    base.save(ROOT / "models" / "baseline.joblib")
    metrics["classifier"]["baseline"] = {
        "train_seconds": round(time.perf_counter() - t0, 2),
        "seen": evaluate("baseline", base.predict_proba(test_seen["descriptor"].tolist()), test_seen),
        "unseen": evaluate("baseline", base.predict_proba(test_unseen["descriptor"].tolist()), test_unseen),
    }

    if not args.skip_bert:
        # Fine-tuning varies run to run, so train several seeds and report the spread.
        # The seed-0 model is saved and gets the detailed breakdown.
        runs = []
        for seed in range(args.seeds):
            print(f"\n[distilbert] fine-tuning distilbert-base-uncased, seed {seed}, {args.epochs} epochs (CPU)")
            t0 = time.perf_counter()
            bert = DistilBertClassifier(seed=seed).fit(train["descriptor"].tolist(), train["category"].tolist(), epochs=args.epochs, seed=seed)
            run = {
                "seed": seed,
                "train_seconds": round(time.perf_counter() - t0, 2),
                "seen": evaluate("distilbert", bert.predict_proba(test_seen["descriptor"].tolist()), test_seen),
                "unseen": evaluate("distilbert", bert.predict_proba(test_unseen["descriptor"].tolist()), test_unseen),
            }
            print(f"  unseen-merchant accuracy {run['unseen']['accuracy']:.3f}")
            if seed == 0:
                bert.save(ROOT / "models" / "distilbert")
                metrics["classifier"]["distilbert"] = run
            runs.append(run)
        acc = np.array([r["unseen"]["accuracy"] for r in runs])
        f1 = np.array([r["unseen"]["macro_f1"] for r in runs])
        metrics["distilbert_seeds"] = {
            "seeds": len(runs),
            "unseen_accuracy": acc.tolist(),
            "unseen_accuracy_mean": round(float(acc.mean()), 4),
            "unseen_accuracy_std": round(float(acc.std(ddof=1)), 4) if len(runs) > 1 else 0.0,
            "unseen_macro_f1_mean": round(float(f1.mean()), 4),
            "seen_accuracy_mean": round(float(np.mean([r["seen"]["accuracy"] for r in runs])), 4),
        }
        s = metrics["distilbert_seeds"]
        print(f"\n[distilbert] unseen-merchant accuracy over {len(runs)} seeds: {s['unseen_accuracy_mean']:.3f} +/- {s['unseen_accuracy_std']:.3f}")

    print("\n[classifier] accuracy / macro-F1")
    for name, m in metrics["classifier"].items():
        print(f"  {name:10s} seen-merchant {m['seen']['accuracy']:.3f} / {m['seen']['macro_f1']:.3f}   "
              f"unseen-merchant {m['unseen']['accuracy']:.3f} / {m['unseen']['macro_f1']:.3f}   ({m['train_seconds']}s train)")
        for c in m["unseen"]["coverage"]:
            if c["threshold"] in (0.6, 0.8):
                acc = c["accuracy_on_accepted"]
                print(f"    unseen @ conf>={c['threshold']}: auto-accept {c['auto_accept_rate']:.2f}, "
                      f"accuracy on accepted {acc if acc is None else round(acc, 3)}, review {c['review_rate']:.2f}")

    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_results(metrics)
    print("\nwrote reports/metrics.json and reports/RESULTS.md")


def write_results(m):
    lines = ["# Results", "", "Generated by `python scripts/train.py`. All data is synthetic (see MODEL_CARD.md).", "",
             "## Merchant resolution (top-1 accuracy, all 80 merchants)", "", "| noise | method | accuracy | ms / descriptor |", "|---|---|---|---|"]
    for level, methods in m["resolver"].items():
        for k, v in methods.items():
            lines.append(f"| {level} | {k} | {v['top1_accuracy']:.3f} | {v['ms_per_descriptor']} |")
    lines += ["", "## Category classification", "",
              "| model | seen-merchant acc | seen macro-F1 | unseen-merchant acc | unseen macro-F1 | train time |", "|---|---|---|---|---|---|"]
    for k, v in m["classifier"].items():
        lines.append(f"| {k} | {v['seen']['accuracy']:.3f} | {v['seen']['macro_f1']:.3f} | {v['unseen']['accuracy']:.3f} | {v['unseen']['macro_f1']:.3f} | {v['train_seconds']}s |")
    if "distilbert_seeds" in m:
        s = m["distilbert_seeds"]
        lines += ["", f"DistilBERT over {s['seeds']} seeds: unseen-merchant accuracy **{s['unseen_accuracy_mean']:.3f} +/- {s['unseen_accuracy_std']:.3f}** "
                  f"(runs: {', '.join(f'{a:.3f}' for a in s['unseen_accuracy'])}), macro-F1 {s['unseen_macro_f1_mean']:.3f}. "
                  "The table row and everything below use the saved seed-0 model."]
    lines += ["", "## Human-review routing on unseen merchants", "", "| model | confidence threshold | auto-accepted | accuracy on accepted | sent to review |", "|---|---|---|---|---|"]
    for k, v in m["classifier"].items():
        for c in v["unseen"]["coverage"]:
            acc = "-" if c["accuracy_on_accepted"] is None else f"{c['accuracy_on_accepted']:.3f}"
            lines.append(f"| {k} | {c['threshold']} | {c['auto_accept_rate']:.2f} | {acc} | {c['review_rate']:.2f} |")
    for k, v in m["classifier"].items():
        lines += ["", f"## Confusion matrix: {k}, unseen merchants", "", md_confusion(v["unseen"]["confusion"])]
    (ROOT / "reports" / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
