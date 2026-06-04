"""
step9_evaluate_saved_model.py

Evaluate an already-saved Step7 model on labeled Step6 sequences.
This script does not train. It reloads model_artifacts, predicts on a chosen
split/date range, and writes report JSON plus row-level prediction CSV.

Example:
  python step9_evaluate_saved_model.py 06_model_input/lstm_sequences.npz --artifacts 07_eval/model_artifacts --split test
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from step8_predict_latest import (
    HORIZON_WEEKS,
    check_features,
    load_meta,
    parse_week_end,
    predict_lstm,
    predict_xgb,
    remap_country_ids,
)


def load_eval_sequences(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    cid = d["country_id"].astype(np.int64)
    split = d["split"].astype(str) if "split" in d else np.array(["all"] * len(y))
    week_end = d["week_end"].astype(str) if "week_end" in d else np.array([""] * len(y))
    feature_names = [str(c) for c in d["feature_names"]]
    country_index = [str(c) for c in d["country_index"]]
    if (y < 0).any():
        raise ValueError("Evaluation requires labels, but this npz contains unlabeled rows.")
    return X, y, cid, split, week_end, feature_names, country_index


def build_mask(split, week_end, split_name, target_start=None, target_end=None):
    if split_name == "all":
        mask = np.ones(len(split), dtype=bool)
    else:
        mask = split == split_name

    target_week_end = parse_week_end(week_end) + pd.to_timedelta(HORIZON_WEEKS, unit="W")
    if target_start:
        mask &= target_week_end >= pd.Timestamp(target_start)
    if target_end:
        mask &= target_week_end <= pd.Timestamp(target_end)
    return mask, target_week_end.dt.strftime("%Y-%m-%d").to_numpy()


def metrics(y_true, p, threshold):
    pred = (p >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, p)),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, p))
        out["ap"] = float(average_precision_score(y_true, p))
    else:
        out["auc"] = None
        out["ap"] = None

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return out


def choose_threshold_for_min_precision(y_true, p, min_precision):
    if len(np.unique(y_true)) < 2:
        return None

    precision, recall, thresholds = precision_recall_curve(y_true, p)
    candidates = []
    for i, threshold in enumerate(thresholds):
        if precision[i] >= min_precision:
            f1 = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-8)
            candidates.append(
                {
                    "threshold": float(threshold),
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1),
                }
            )
    if not candidates:
        return None

    candidates.sort(key=lambda x: (x["recall"], x["f1"], -x["threshold"]), reverse=True)
    best = candidates[0]
    best["min_precision"] = float(min_precision)
    best["n_candidates"] = len(candidates)
    return best


def predict_with_saved_model(artifact_dir, meta, X, cid):
    final_model = meta["final_model"]
    alpha = float(meta.get("alpha", 0.0))
    p_xgb = None
    p_lstm = None

    if final_model in {"xgb", "ensemble"}:
        p_xgb = predict_xgb(artifact_dir, meta, X, cid)
    if final_model in {"lstm", "ensemble"}:
        p_lstm = predict_lstm(artifact_dir, meta, X, cid)

    if final_model == "xgb":
        return p_xgb, p_xgb, p_lstm
    if final_model == "lstm":
        return p_lstm, p_xgb, p_lstm
    if final_model == "ensemble":
        return alpha * p_xgb + (1 - alpha) * p_lstm, p_xgb, p_lstm
    raise ValueError(f"Unknown final_model in metadata: {final_model}")


def group_metrics(pred_df):
    out = {}
    for country, g in pred_df.groupby("country"):
        if len(g) < 2:
            continue
        out[country] = metrics(g["y_true"].to_numpy(), g["p_conflict"].to_numpy(), float(g["threshold"].iloc[0]))
    return out


def run(
    npz_path,
    artifact_dir,
    split_name,
    output_json,
    output_csv,
    target_start=None,
    target_end=None,
    threshold_mode="saved",
    min_precision=0.8,
):
    meta = load_meta(artifact_dir)
    X, y, cid_latest, split, week_end, feature_names, country_index_latest = load_eval_sequences(npz_path)
    check_features(feature_names, meta["feature_names"])
    countries, cid = remap_country_ids(cid_latest, country_index_latest, meta["country_index"])

    mask, target_week_end = build_mask(split, week_end, split_name, target_start, target_end)
    if not mask.any():
        _, all_targets = build_mask(split, week_end, split_name, None, None)
        valid_targets = pd.Series(all_targets).dropna()
        available = "unknown"
        if not valid_targets.empty:
            available = f"{valid_targets.min()} ~ {valid_targets.max()}"
        raise ValueError(
            "No rows matched the requested split/date filter. "
            f"Available target_week_end range for split={split_name}: {available}"
        )

    p_final, p_xgb, p_lstm = predict_with_saved_model(artifact_dir, meta, X[mask], cid[mask])
    saved_threshold = float(meta["threshold"])
    threshold = saved_threshold
    threshold_selection = {
        "mode": threshold_mode,
        "saved_threshold": saved_threshold,
        "selected_threshold": saved_threshold,
    }

    if threshold_mode == "min-precision":
        tune_mask = split == "val"
        if tune_mask.sum() == 0:
            raise ValueError("Cannot tune threshold: val split is empty.")
        p_val, _, _ = predict_with_saved_model(artifact_dir, meta, X[tune_mask], cid[tune_mask])
        selected = choose_threshold_for_min_precision(y[tune_mask], p_val, min_precision)
        if selected is None:
            raise ValueError(
                f"No validation threshold satisfied precision >= {min_precision:.3f}. "
                "Try lowering --min-precision."
            )
        threshold = float(selected["threshold"])
        threshold_selection.update(
            {
                "selected_threshold": threshold,
                "tuned_on_split": "val",
                "validation": selected,
            }
        )

    pred = (p_final >= threshold).astype(int)

    pred_df = pd.DataFrame(
        {
            "country": countries[mask],
            "week_end": week_end[mask],
            "target_week_end": target_week_end[mask],
            "split": split[mask],
            "y_true": y[mask],
            "p_conflict": p_final,
            "pred": pred,
            "final_model": meta["final_model"],
            "threshold": threshold,
        }
    )
    if p_xgb is not None:
        pred_df["p_xgb"] = p_xgb
    if p_lstm is not None:
        pred_df["p_lstm"] = p_lstm

    overall = metrics(pred_df["y_true"].to_numpy(), pred_df["p_conflict"].to_numpy(), threshold)
    by_country = group_metrics(pred_df)
    top_risk = (
        pred_df.sort_values("p_conflict", ascending=False)
        .head(20)[["country", "week_end", "target_week_end", "y_true", "p_conflict", "pred"]]
        .to_dict("records")
    )

    report = {
        "artifact_dir": artifact_dir,
        "final_model": meta["final_model"],
        "final_model_reason": meta.get("final_model_reason"),
        "split": split_name,
        "target_start": target_start,
        "target_end": target_end,
        "threshold_selection": threshold_selection,
        "overall": overall,
        "by_country": by_country,
        "top_risk_rows": top_risk,
    }

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    pred_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("step9: saved-model evaluation")
    print("=" * 70)
    print(f"Rows: {overall['n']:,} | positive rate: {overall['positive_rate']:.3f}")
    print(f"Model: {meta['final_model']} | threshold: {threshold:.3f}")
    if threshold_mode == "min-precision":
        v = threshold_selection["validation"]
        print(
            f"Threshold tuned on val: min_precision={min_precision:.3f} | "
            f"val P={v['precision']:.4f} | val R={v['recall']:.4f} | "
            f"val F1={v['f1']:.4f}"
        )
    auc_text = "NA" if overall["auc"] is None else f"{overall['auc']:.4f}"
    ap_text = "NA" if overall["ap"] is None else f"{overall['ap']:.4f}"
    print(
        f"AUC {auc_text} | AP {ap_text} | F1 {overall['f1']:.4f} | "
        f"P {overall['precision']:.4f} | R {overall['recall']:.4f} | "
        f"Brier {overall['brier']:.4f}"
    )
    print(f"Saved report: {output_json}")
    print(f"Saved predictions: {output_csv}")


def main():
    p = argparse.ArgumentParser(description="Evaluate saved Step7 model without retraining.")
    p.add_argument("npz", nargs="?", default="06_model_input/lstm_sequences.npz")
    p.add_argument("--artifacts", default="07_eval/model_artifacts")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--target-start", default=None, help="optional target_week_end lower bound, e.g. 2024-10-01")
    p.add_argument("--target-end", default=None, help="optional target_week_end upper bound, e.g. 2024-12-31")
    p.add_argument("--threshold-mode", default="saved", choices=["saved", "min-precision"])
    p.add_argument("--min-precision", type=float, default=0.8)
    p.add_argument("--out", default="07_eval/final_eval_report.json")
    p.add_argument("--pred-out", default="07_eval/final_eval_predictions.csv")
    args = p.parse_args()
    run(
        args.npz,
        args.artifacts,
        args.split,
        args.out,
        args.pred_out,
        args.target_start,
        args.target_end,
        args.threshold_mode,
        args.min_precision,
    )


if __name__ == "__main__":
    main()
