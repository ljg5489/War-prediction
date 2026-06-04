"""
step7_train_ensemble.py

Train XGBoost and LSTM on Step6 sequences, evaluate on the temporal test split,
and save reusable model artifacts for inference-only prediction in Step8.

Outputs:
  07_eval/ensemble_report.json
  07_eval/ensemble_predictions.csv
  07_eval/model_artifacts/model_meta.json
  07_eval/model_artifacts/lstm_model.pt
  07_eval/model_artifacts/xgb_model.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


SEED = 42
np.random.seed(SEED)

LSTM_CONFIG = {"hidden": 128, "embed_dim": 16, "dropout": 0.3}


def load_sequences(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    cid = d["country_id"].astype(np.int64)
    split = d["split"].astype(str)
    week_end = d["week_end"].astype(str) if "week_end" in d else np.array([""] * len(y))
    feature_names = [str(c) for c in d["feature_names"]]
    country_index = [str(c) for c in d["country_index"]]

    print(f"Loaded sequences: X={X.shape}, countries={len(country_index)}, features={len(feature_names)}")
    if (y < 0).any():
        raise ValueError("This Step7 script requires labels. Use step8_predict_latest.py for inference-only data.")
    return X, y, cid, split, week_end, feature_names, country_index


def resolve_splits(split: np.ndarray):
    tr = split == "train"
    va = split == "val"
    te = split == "test"
    if te.sum() == 0:
        print("Warning: test split is empty; evaluating on val.")
        te = va.copy()
    if va.sum() == 0:
        print("Warning: val split is empty; using the last 20% of train as val.")
        idx = np.where(tr)[0]
        cut = int(len(idx) * 0.8)
        va = np.zeros_like(tr)
        va[idx[cut:]] = True
        tr = np.zeros_like(tr)
        tr[idx[:cut]] = True
    print(f"split: train={tr.sum():,} val={va.sum():,} test={te.sum():,}")
    return tr, va, te


def flatten_for_xgb(X: np.ndarray, cid: np.ndarray) -> np.ndarray:
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    trend = X[:, -1, :] - X[:, 0, :]
    return np.hstack([last, mean, trend, cid.reshape(-1, 1).astype(np.float32)])


def train_xgb(X_tr, y_tr, X_va, y_va):
    pos = max(1, int((y_tr == 1).sum()))
    neg = max(1, int((y_tr == 0).sum()))
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        eval_metric="auc",
        random_state=SEED,
        early_stopping_rounds=30,
    )
    if len(np.unique(y_va)) > 1:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        model.set_params(early_stopping_rounds=None)
        model.fit(X_tr, y_tr, verbose=False)
    return model


def fit_standardizer(X: np.ndarray, tr_mask: np.ndarray, eps: float = 1e-8):
    flat = X[tr_mask].reshape(-1, X.shape[2])
    mu = flat.mean(axis=0).astype(np.float32)
    sig = (flat.std(axis=0) + eps).astype(np.float32)
    return mu, sig


def apply_standardizer(X: np.ndarray, mu: np.ndarray, sig: np.ndarray):
    return ((X - mu) / sig).astype(np.float32)


def make_lstm_model(n_countries: int, n_features: int, hidden: int, embed_dim: int, dropout: float):
    import torch
    import torch.nn as nn

    class GlobalConflictLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_countries, embed_dim)
            self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden + embed_dim, 64)
            self.fc2 = nn.Linear(64, 1)

        def forward(self, x_seq, x_cid):
            out, _ = self.lstm(x_seq)
            last = out[:, -1, :]
            h = torch.cat([last, self.emb(x_cid)], dim=1)
            h = self.drop(torch.relu(self.fc1(h)))
            return self.fc2(h).squeeze(-1)

    return GlobalConflictLSTM()


def train_lstm(X, y, cid, tr, va, n_countries, n_features, epochs=20, batch_size=32, lr=1e-3):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(SEED)
    model = make_lstm_model(n_countries, n_features, **LSTM_CONFIG)

    def to_tensor(mask):
        return (
            torch.FloatTensor(X[mask]),
            torch.LongTensor(cid[mask]),
            torch.FloatTensor(y[mask].astype(np.float32)),
        )

    Xtr, ctr, ytr = to_tensor(tr)
    Xva, cva, _ = to_tensor(va)
    pos = max(1, int((y[tr] == 1).sum()))
    neg = max(1, int((y[tr] == 0).sum()))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(Xtr, ctr, ytr), batch_size=batch_size, shuffle=True)

    best_auc = -1.0
    best_state = None
    for _ in range(epochs):
        model.train()
        for xb, cb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb, cb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva, cva)).numpy()
        if len(np.unique(y[va])) > 1:
            auc = roc_auc_score(y[va], pv)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"LSTM best val AUC = {best_auc:.4f}")
    return model


def predict_lstm(model, X, cid, mask):
    import torch

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.FloatTensor(X[mask]), torch.LongTensor(cid[mask]))).numpy()


def optimize_alpha(y_va, p_xgb, p_lstm):
    if len(np.unique(y_va)) < 2:
        return 0.0, float("nan")
    best_alpha, best_auc = 0.0, -1.0
    for alpha in np.linspace(0, 1, 21):
        auc = roc_auc_score(y_va, alpha * p_xgb + (1 - alpha) * p_lstm)
        if auc > best_auc:
            best_alpha, best_auc = float(alpha), float(auc)
    return best_alpha, best_auc


def tune_threshold(y_va, p_va):
    if len(np.unique(y_va)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_va, p_va)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best = int(np.argmax(f1))
    return float(thr[best]) if best < len(thr) else 0.5


def evaluate(y_true, p, threshold):
    out = {}
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, p))
        out["ap"] = float(average_precision_score(y_true, p))
    else:
        out["auc"] = float("nan")
        out["ap"] = float("nan")
    pred = (p >= threshold).astype(int)
    out["f1"] = float(f1_score(y_true, pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, pred, zero_division=0))
    out["brier"] = float(brier_score_loss(y_true, p))
    out["threshold"] = float(threshold)
    return out


def choose_final_model(alpha, m_xgb, m_lstm, m_ens):
    if alpha <= 0.05:
        return "lstm", "soft_voting_alpha_is_zero"
    if m_ens["auc"] < m_lstm["auc"] or m_ens["ap"] < m_lstm["ap"]:
        return "lstm", "ensemble_not_better_than_lstm"
    if m_ens["auc"] < m_xgb["auc"] or m_ens["ap"] < m_xgb["ap"]:
        return "xgb", "ensemble_not_better_than_xgb"
    return "ensemble", "ensemble_best_or_tied"


def fmt(name, m):
    return (
        f"{name:<9} AUC {m['auc']:.4f} | AP {m['ap']:.4f} | "
        f"F1 {m['f1']:.4f} | P {m['precision']:.4f} | "
        f"R {m['recall']:.4f} | Brier {m['brier']:.4f}"
    )


def load_context(features_path):
    if not features_path or not os.path.exists(features_path):
        return None, None
    f = pd.read_parquet(features_path)
    if "Y" not in f.columns or "week_id" not in f.columns:
        return None, None
    y_now = {
        (str(c), str(w)): yv
        for c, w, yv in zip(f["country_std"].astype(str), f["week_id"].astype(str), f["Y"])
    }
    group_map = {}
    if "group" in f.columns:
        group_map = (
            f.dropna(subset=["group"])
            .drop_duplicates("country_std")
            .set_index("country_std")["group"]
            .astype(str)
            .to_dict()
        )
    return y_now, group_map


def diagnostics(country_arr, week_arr, y_future, p, threshold, y_now_lookup, group_map):
    pred = (p >= threshold).astype(int)
    y_now = np.array(
        [y_now_lookup.get((str(c), str(w)), np.nan) for c, w in zip(country_arr, week_arr)],
        dtype=float,
    )
    valid = ~np.isnan(y_now)
    out = {}

    if valid.sum() and len(np.unique(y_future[valid])) > 1:
        yn = y_now[valid].astype(int)
        yf = y_future[valid].astype(int)
        pers_auc = float(roc_auc_score(yf, y_now[valid]))
        pers_f1 = float(f1_score(yf, yn, zero_division=0))
        pers_acc = float((yn == yf).mean())
        model_auc = float(roc_auc_score(yf, p[valid]))
        model_f1 = float(f1_score(yf, pred[valid], zero_division=0))
        out.update(
            persistence_auc=pers_auc,
            persistence_f1=pers_f1,
            persistence_acc=pers_acc,
            model_auc_valid=model_auc,
            model_f1_valid=model_f1,
            lift_auc=model_auc - pers_auc,
        )
        print(f"Persistence AUC {pers_auc:.4f} | F1 {pers_f1:.4f} | acc {pers_acc:.4f}")
        print(f"Final model AUC {model_auc:.4f} | F1 {model_f1:.4f}")

    onset = valid & (y_now == 0) & (y_future == 1)
    decay = valid & (y_now == 1) & (y_future == 0)
    out["onset_n"] = int(onset.sum())
    out["decay_n"] = int(decay.sum())
    if onset.sum():
        out["onset_recall"] = float((pred[onset] == 1).mean())
        print(f"Onset cases: {onset.sum()} | recall {out['onset_recall']:.3f}")
    else:
        print("Onset cases: 0 | early-warning onset evaluation is not available in this test split.")

    if group_map:
        groups = np.array([group_map.get(str(c), "?") for c in country_arr])
        out["by_group"] = {}
        for g in sorted(set(groups)):
            m = groups == g
            base = float(y_future[m].mean())
            if len(np.unique(y_future[m])) > 1:
                auc = float(roc_auc_score(y_future[m], p[m]))
                f1 = float(f1_score(y_future[m], pred[m], zero_division=0))
                print(f"Group {g}: n={m.sum()} Y=1 {base:.2f} AUC {auc:.4f} F1 {f1:.4f}")
                out["by_group"][g] = {"n": int(m.sum()), "base_rate": base, "auc": auc, "f1": f1}
            else:
                out["by_group"][g] = {"n": int(m.sum()), "base_rate": base, "auc": None, "f1": None}
    return out


def save_artifacts(
    output_dir,
    xgb,
    lstm,
    mu,
    sig,
    final_model,
    final_reason,
    alpha,
    threshold,
    feature_names,
    country_index,
):
    import torch

    artifact_dir = os.path.join(output_dir, "model_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    xgb_path = os.path.join(artifact_dir, "xgb_model.json")
    lstm_path = os.path.join(artifact_dir, "lstm_model.pt")
    meta_path = os.path.join(artifact_dir, "model_meta.json")

    xgb.save_model(xgb_path)
    torch.save(lstm.state_dict(), lstm_path)

    meta = {
        "version": 1,
        "final_model": final_model,
        "final_model_reason": final_reason,
        "alpha": float(alpha),
        "threshold": float(threshold),
        "feature_names": feature_names,
        "country_index": country_index,
        "n_features": len(feature_names),
        "n_countries": len(country_index),
        "standardizer_mean": [float(v) for v in mu],
        "standardizer_std": [float(v) for v in sig],
        "lstm_config": LSTM_CONFIG,
        "artifacts": {
            "xgb_model": "xgb_model.json",
            "lstm_model": "lstm_model.pt",
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return artifact_dir


def run(npz_path, output_dir, epochs, features_path=None):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("step7: train, evaluate, and save reusable artifacts")
    print("=" * 70)

    X, y, cid, split, week_end, feature_names, country_index = load_sequences(npz_path)
    tr, va, te = resolve_splits(split)
    n_countries = len(country_index)
    n_features = X.shape[2]

    if y[tr].sum() == 0 or (y[tr] == 0).sum() == 0:
        raise ValueError("Train split has only one class. Check labels/split/horizon.")

    print("\nTraining XGBoost...")
    Xf = flatten_for_xgb(X, cid)
    xgb = train_xgb(Xf[tr], y[tr], Xf[va], y[va])
    pxgb_va = xgb.predict_proba(Xf[va])[:, 1]
    pxgb_te = xgb.predict_proba(Xf[te])[:, 1]

    print("\nTraining LSTM...")
    mu, sig = fit_standardizer(X, tr)
    Xs = apply_standardizer(X, mu, sig)
    lstm = train_lstm(Xs, y, cid, tr, va, n_countries, n_features, epochs=epochs)
    plstm_va = predict_lstm(lstm, Xs, cid, va)
    plstm_te = predict_lstm(lstm, Xs, cid, te)

    alpha, val_auc = optimize_alpha(y[va], pxgb_va, plstm_va)
    pens_va = alpha * pxgb_va + (1 - alpha) * plstm_va
    pens_te = alpha * pxgb_te + (1 - alpha) * plstm_te
    print(f"\nSoft voting alpha={alpha:.2f} (XGB weight), val AUC={val_auc:.4f}")

    ens_threshold = tune_threshold(y[va], pens_va)
    m_xgb = evaluate(y[te], pxgb_te, ens_threshold)
    m_lstm = evaluate(y[te], plstm_te, ens_threshold)
    m_ens = evaluate(y[te], pens_te, ens_threshold)

    print(f"\nTEST evaluation at ensemble threshold={ens_threshold:.3f}")
    print(fmt("XGBoost", m_xgb))
    print(fmt("LSTM", m_lstm))
    print(fmt("Ensemble", m_ens))

    final_model, final_reason = choose_final_model(alpha, m_xgb, m_lstm, m_ens)
    prob_va = {"xgb": pxgb_va, "lstm": plstm_va, "ensemble": pens_va}[final_model]
    prob_te = {"xgb": pxgb_te, "lstm": plstm_te, "ensemble": pens_te}[final_model]
    final_threshold = tune_threshold(y[va], prob_va)
    final_test = evaluate(y[te], prob_te, final_threshold)
    print(f"\nFinal model: {final_model.upper()} ({final_reason})")
    print(fmt("Final", final_test))

    y_now_lookup, group_map = load_context(features_path)
    diag = {}
    if y_now_lookup is not None:
        country_te = np.array([country_index[i] for i in cid[te]])
        diag = diagnostics(country_te, week_end[te], y[te], prob_te, final_threshold, y_now_lookup, group_map)

    artifact_dir = save_artifacts(
        output_dir,
        xgb,
        lstm,
        mu,
        sig,
        final_model,
        final_reason,
        alpha,
        final_threshold,
        feature_names,
        country_index,
    )

    report = {
        "alpha": float(alpha),
        "val_auc": float(val_auc),
        "ensemble_threshold": float(ens_threshold),
        "final_model": final_model,
        "final_model_reason": final_reason,
        "final_threshold": float(final_threshold),
        "n_countries": n_countries,
        "n_features": n_features,
        "features": feature_names,
        "test": {"xgb": m_xgb, "lstm": m_lstm, "ensemble": m_ens},
        "final_test": final_test,
        "diagnostics": diag,
        "split_counts": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "model_artifacts": artifact_dir,
    }
    with open(os.path.join(output_dir, "ensemble_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    pd.DataFrame(
        {
            "country_id": cid[te],
            "country": [country_index[i] for i in cid[te]],
            "week_end": week_end[te],
            "y_true": y[te],
            "p_xgb": pxgb_te,
            "p_lstm": plstm_te,
            "p_ensemble": pens_te,
            "p_final": prob_te,
            "final_model": final_model,
            "pred": (prob_te >= final_threshold).astype(int),
        }
    ).to_csv(os.path.join(output_dir, "ensemble_predictions.csv"), index=False, encoding="utf-8-sig")

    print(f"\nSaved report and predictions to: {output_dir}")
    print(f"Saved reusable model artifacts to: {artifact_dir}")


def main():
    p = argparse.ArgumentParser(description="Train/evaluate conflict models and save artifacts.")
    p.add_argument("npz", nargs="?", help="06_model_input/lstm_sequences.npz")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--features", default=None, help="05_features/features_weekly.parquet for diagnostics")
    p.add_argument("--out", default="07_eval", help="output directory")
    args = p.parse_args()

    npz = args.npz
    if not npz:
        print("lstm_sequences.npz path:")
        npz = input("  >> ").strip().strip('"').strip("'")
    if not os.path.exists(npz):
        raise FileNotFoundError(npz)
    run(npz, args.out, args.epochs, args.features)


if __name__ == "__main__":
    main()
