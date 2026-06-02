"""
step7_train_ensemble.py
──────────────────────────────────────────────────────────────────────────────
글로벌 XGBoost + LSTM Soft Voting 앙상블 (step6 산출물 기반)

이전 run_global_ensemble.py 의 문제를 모두 수정한 버전:
  ✗ 연 단위 데이터에 seq_len=90 → 93% 제로패딩   ✓ step6 의 13주 시퀀스 그대로 사용
  ✗ WRI 성분(CII/ESC/SFI/Econ/Spill/Polit) 피처화 ✓ step6 에서 WRI·theta 제외된 피처 사용
  ✗ groupby 후 인덱스 분할 = 국가 holdout         ✓ step6 의 연도 기반 split 사용
  ✗ threshold 를 val 에서 튜닝하고 val 에서 평가   ✓ threshold 는 val, 최종 평가는 test
  ✗ 학습 안 된 랜덤 임베딩을 XGBoost 에 투입       ✓ XGB 는 country_id, LSTM 은 학습된 임베딩
  ✗ shuffle=False / 스케일 정규화 없음             ✓ shuffle=True / train 기준 표준화
  ✗ α 하드코딩                                     ✓ val AUC 로 α 최적화
x
입력:  lstm_sequences.npz  (step6 출력)
       └ X(N,T,F) / y(N) / country_id(N) / split(N) / feature_names / country_index

출력:  콘솔 평가 리포트 (XGB / LSTM / Ensemble × test)
       ensemble_report.json  +  ensemble_predictions.csv

사용법:
    python step7_train_ensemble.py <lstm_sequences.npz> [--epochs 20] [--out 07_eval]
    python step7_train_ensemble.py   (인자 없으면 경로 입력 프롬프트)
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import json
import argparse
import os
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, precision_recall_curve,
                             precision_score, recall_score, f1_score)

SEED = 42
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 데이터 로딩
# ─────────────────────────────────────────────────────────────────────────────

def load_sequences(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    X   = d["X"].astype(np.float32)               # (N, T, F)
    y   = d["y"].astype(np.int64)                 # (N,)
    cid = d["country_id"].astype(np.int64)        # (N,)
    split = d["split"].astype(str)                # (N,)
    feat_names = [str(c) for c in d["feature_names"]]
    country_index = [str(c) for c in d["country_index"]]
    week_end = d["week_end"].astype(str) if "week_end" in d else np.array([""] * len(y))

    print(f"📂 시퀀스 로드: X={X.shape}  ({len(country_index)}개국, 피처 {len(feat_names)}개)")
    if (y < 0).any():
        raise ValueError("라벨(y)이 없는 시퀀스가 포함돼 있습니다. 추론 전용 데이터로 학습 불가.")
    return X, y, cid, split, week_end, feat_names, country_index


def resolve_splits(split: np.ndarray):
    """train/val/test 마스크. test 가 비면 val 로 평가, val 도 비면 train 에서 분리."""
    tr = split == "train"
    va = split == "val"
    te = split == "test"
    if te.sum() == 0:
        print("   ⚠️  test 비어 있음 → val 을 test 로 대체 평가")
        te = va.copy()
    if va.sum() == 0:
        print("   ⚠️  val 비어 있음 → train 끝 20%를 val 로 분리")
        idx = np.where(tr)[0]
        cut = int(len(idx) * 0.8)
        va = np.zeros_like(tr); va[idx[cut:]] = True
        tr = np.zeros_like(tr); tr[idx[:cut]] = True
    print(f"   split: train={tr.sum():,}  val={va.sum():,}  test={te.sum():,}")
    return tr, va, te


# ─────────────────────────────────────────────────────────────────────────────
# 2. XGBoost (시퀀스 평면화: 끝값 + 평균 + 추세 + country_id)
# ─────────────────────────────────────────────────────────────────────────────

def flatten_for_xgb(X: np.ndarray, cid: np.ndarray) -> np.ndarray:
    last  = X[:, -1, :]
    mean_ = X.mean(axis=1)
    trend = X[:, -1, :] - X[:, 0, :]
    return np.hstack([last, mean_, trend, cid.reshape(-1, 1).astype(np.float32)])


def train_xgb(Xf_tr, y_tr, Xf_va, y_va):
    pos = max(1, int((y_tr == 1).sum()))
    neg = max(1, int((y_tr == 0).sum()))
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / pos,          # 클래스 불균형 보정
        eval_metric="auc", random_state=SEED,
        early_stopping_rounds=30,
    )
    if len(np.unique(y_va)) > 1:
        model.fit(Xf_tr, y_tr, eval_set=[(Xf_va, y_va)], verbose=False)
    else:
        model.set_params(early_stopping_rounds=None)
        model.fit(Xf_tr, y_tr, verbose=False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. LSTM (torch 는 함수 안에서 지연 임포트 — 미설치 환경에서도 모듈 임포트 가능)
# ─────────────────────────────────────────────────────────────────────────────

def standardize(X, tr_mask, eps=1e-8):
    """train 기준 (피처별) 표준화. (N,T,F) → 동일 shape."""
    mu  = X[tr_mask].reshape(-1, X.shape[2]).mean(axis=0)
    sig = X[tr_mask].reshape(-1, X.shape[2]).std(axis=0) + eps
    return (X - mu) / sig


def train_lstm(X, y, cid, tr, va, n_countries, n_features,
               epochs=20, batch_size=32, hidden=128, embed_dim=16, lr=1e-3):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(SEED)

    class GlobalConflictLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb  = nn.Embedding(n_countries, embed_dim)
            self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
            self.drop = nn.Dropout(0.3)
            self.fc1  = nn.Linear(hidden + embed_dim, 64)
            self.fc2  = nn.Linear(64, 1)

        def forward(self, x_seq, x_cid):           # logits 반환 (sigmoid 미적용)
            out, _ = self.lstm(x_seq)
            last = out[:, -1, :]
            h = torch.cat([last, self.emb(x_cid)], dim=1)
            h = self.drop(torch.relu(self.fc1(h)))
            return self.fc2(h).squeeze(-1)

    def to_t(mask):
        return (torch.FloatTensor(X[mask]),
                torch.LongTensor(cid[mask]),
                torch.FloatTensor(y[mask].astype(np.float32)))

    Xtr, ctr, ytr = to_t(tr)
    Xva, cva, yva = to_t(va)

    pos = max(1, int((y[tr] == 1).sum()))
    neg = max(1, int((y[tr] == 0).sum()))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)

    model = GlobalConflictLSTM()
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(Xtr, ctr, ytr),
                        batch_size=batch_size, shuffle=True)

    best_auc, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for xb, cb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb, cb), yb)
            loss.backward()
            opt.step()
        # val AUC 로 best 체크포인트
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva, cva)).numpy()
        if len(np.unique(y[va])) > 1:
            auc = roc_auc_score(y[va], pv)
            if auc > best_auc:
                best_auc, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"   LSTM best val AUC = {best_auc:.4f}")

    def predict(mask):
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(torch.FloatTensor(X[mask]),
                                       torch.LongTensor(cid[mask]))).numpy()
    return predict


# ─────────────────────────────────────────────────────────────────────────────
# 4. 앙상블 α 최적화 / threshold 튜닝 / 평가
# ─────────────────────────────────────────────────────────────────────────────

def optimize_alpha(y_va, p_xgb, p_lstm):
    """val AUC 를 최대화하는 α (p = α·xgb + (1−α)·lstm)."""
    if len(np.unique(y_va)) < 2:
        return 0.5, float("nan")
    best_a, best_auc = 0.5, -1.0
    for a in np.linspace(0, 1, 21):
        auc = roc_auc_score(y_va, a * p_xgb + (1 - a) * p_lstm)
        if auc > best_auc:
            best_auc, best_a = auc, a
    return float(best_a), float(best_auc)


def tune_threshold(y_va, p_va):
    """val 에서 F1 최대 threshold."""
    if len(np.unique(y_va)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_va, p_va)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best = int(np.argmax(f1))
    return float(thr[best]) if best < len(thr) else 0.5


def evaluate(y_true, p, threshold):
    out = {}
    if len(np.unique(y_true)) > 1:
        out["auc"] = roc_auc_score(y_true, p)
        out["ap"]  = average_precision_score(y_true, p)
    else:
        out["auc"] = float("nan"); out["ap"] = float("nan")
    yb = (p >= threshold).astype(int)
    out["f1"]        = f1_score(y_true, yb, zero_division=0)
    out["precision"] = precision_score(y_true, yb, zero_division=0)
    out["recall"]    = recall_score(y_true, yb, zero_division=0)
    out["brier"]     = brier_score_loss(y_true, p)
    out["threshold"] = threshold
    return out


def fmt(name, m):
    return (f"  {name:<9} AUC {m['auc']:.4f} | AP {m['ap']:.4f} | "
            f"F1 {m['f1']:.4f} | P {m['precision']:.4f} | "
            f"R {m['recall']:.4f} | Brier {m['brier']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. 메인
# ─────────────────────────────────────────────────────────────────────────────

def load_context(features_path):
    """features_weekly.parquet → (현재Y lookup, group map). 없으면 (None, None)."""
    if not features_path or not os.path.exists(features_path):
        return None, None
    f = pd.read_parquet(features_path)
    if "Y" not in f.columns or "week_id" not in f.columns:
        print("   ⚠️  features 파일에 Y/week_id 없음 → persistence/onset 진단 생략")
        return None, None
    ynow = {(str(c), str(w)): yv
            for c, w, yv in zip(f["country_std"].astype(str),
                                f["week_id"].astype(str), f["Y"])}
    grp = {}
    if "group" in f.columns:
        grp = (f.dropna(subset=["group"]).drop_duplicates("country_std")
                 .set_index("country_std")["group"].astype(str).to_dict())
    return ynow, grp


def diagnostics(country_arr, week_arr, y_future, p_ens, th, ynow_lookup, group_map):
    """Persistence 베이스라인 / 전환(onset) 포착 / 그룹별 성능."""
    pred = (p_ens >= th).astype(int)
    y_now = np.array([ynow_lookup.get((str(c), str(w)), np.nan)
                      for c, w in zip(country_arr, week_arr)], dtype=float)
    valid = ~np.isnan(y_now)
    out = {}

    print("\n" + "─" * 70)
    print('  🔬 진단 1: Persistence 베이스라인 ("13주 뒤 Y = 지금 Y")')
    print("─" * 70)
    if valid.sum() and len(np.unique(y_future[valid])) > 1:
        yn = y_now[valid].astype(int); yf = y_future[valid].astype(int)
        pers_auc = roc_auc_score(yf, y_now[valid])
        pers_acc = float((yn == yf).mean())
        pers_f1  = f1_score(yf, yn, zero_division=0)
        ens_auc  = roc_auc_score(yf, p_ens[valid])
        ens_f1   = f1_score(yf, pred[valid], zero_division=0)
        lift = ens_auc - pers_auc
        print(f"   Persistence : AUC {pers_auc:.4f} | F1 {pers_f1:.4f} | 정확도 {pers_acc:.4f}")
        print(f"   Ensemble    : AUC {ens_auc:.4f} | F1 {ens_f1:.4f}")
        print(f"   → 모델이 persistence 대비 더하는 AUC: {lift:+.4f}  "
              f"({'거의 없음 → 과업이 쉬움/누수 의심' if lift < 0.03 else '유의미한 향상'})")
        out.update(persistence_auc=pers_auc, persistence_f1=pers_f1,
                   persistence_acc=pers_acc, ensemble_auc_valid=ens_auc, lift_auc=lift)
    else:
        print("   (현재 Y 조회 불가 또는 단일 클래스 → 생략)")

    print("\n" + "─" * 70)
    print("  🔬 진단 2: 전환(onset) 포착 — 지금 평화(0) → 13주 뒤 위험(1)")
    print("─" * 70)
    onset = valid & (y_now == 0) & (y_future == 1)
    decay = valid & (y_now == 1) & (y_future == 0)
    n_on, n_de = int(onset.sum()), int(decay.sum())
    if n_on:
        rec_on = float((pred[onset] == 1).mean())
        print(f"   onset(0→1) {n_on}개 | Ensemble recall {rec_on:.3f}  "
              f"(persistence는 정의상 0.000)")
        out.update(onset_n=n_on, onset_recall=rec_on)
    else:
        print("   onset(0→1) 사례 없음 — 테스트 구간에 전환이 없음(조기경보 평가 불가)")
    if n_de:
        print(f"   참고 decay(1→0) {n_de}개 | Ensemble 0예측 비율 "
              f"{float((pred[decay] == 0).mean()):.3f}")

    if group_map:
        print("\n" + "─" * 70)
        print("  🔬 진단 3: 그룹별 Ensemble 성능 (A=상시위험·C=상시평화는 쉬움)")
        print("─" * 70)
        groups = np.array([group_map.get(str(c), "?") for c in country_arr])
        out["by_group"] = {}
        for g in sorted(set(groups)):
            m = groups == g
            if m.sum() == 0:
                continue
            base = float(y_future[m].mean())
            if len(np.unique(y_future[m])) > 1:
                gauc = roc_auc_score(y_future[m], p_ens[m])
                gf1  = f1_score(y_future[m], pred[m], zero_division=0)
                print(f"   [{g:<6}] n={m.sum():>5}  Y=1비율 {base:.2f}  "
                      f"AUC {gauc:.4f}  F1 {gf1:.4f}")
                out["by_group"][g] = dict(n=int(m.sum()), base_rate=base, auc=gauc, f1=gf1)
            else:
                print(f"   [{g:<6}] n={m.sum():>5}  Y=1비율 {base:.2f}  (단일 클래스 → AUC 미정)")
                out["by_group"][g] = dict(n=int(m.sum()), base_rate=base, auc=None, f1=None)
    return out


def run(npz_path, output_dir, epochs, features_path=None):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("  step7: XGBoost + LSTM Soft Voting 앙상블")
    print("=" * 70)

    X, y, cid, split, week_end, feat_names, country_index = load_sequences(npz_path)
    tr, va, te = resolve_splits(split)
    n_countries = len(country_index)
    n_features  = X.shape[2]

    if y[tr].sum() == 0 or (y[tr] == 0).sum() == 0:
        print("❌ train 라벨이 한 클래스뿐입니다. split/horizon 을 확인하세요.")
        return

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\n🌳 XGBoost 학습...")
    Xf = flatten_for_xgb(X, cid)
    xgb = train_xgb(Xf[tr], y[tr], Xf[va], y[va])
    pxgb_va = xgb.predict_proba(Xf[va])[:, 1]
    pxgb_te = xgb.predict_proba(Xf[te])[:, 1]

    # ── LSTM ────────────────────────────────────────────────────────────────────
    print("\n🔁 LSTM 학습 (train 기준 표준화)...")
    Xs = standardize(X, tr)
    try:
        predict_lstm = train_lstm(Xs, y, cid, tr, va, n_countries, n_features, epochs=epochs)
        plstm_va = predict_lstm(va)
        plstm_te = predict_lstm(te)
    except ImportError:
        print("   ⚠️  torch 미설치 → LSTM 생략, XGBoost 단독 평가로 진행")
        plstm_va = pxgb_va.copy()
        plstm_te = pxgb_te.copy()

    # ── 앙상블 ───────────────────────────────────────────────────────────────────
    alpha, val_auc = optimize_alpha(y[va], pxgb_va, plstm_va)
    pens_va = alpha * pxgb_va + (1 - alpha) * plstm_va
    pens_te = alpha * pxgb_te + (1 - alpha) * plstm_te
    print(f"\n⚖️  앙상블 α = {alpha:.2f} (xgb 비중), val AUC = {val_auc:.4f}")

    # ── 평가: threshold 는 val, 점수는 test ────────────────────────────────────
    th = tune_threshold(y[va], pens_va)
    m_xgb = evaluate(y[te], pxgb_te, th)
    m_lst = evaluate(y[te], plstm_te, th)
    m_ens = evaluate(y[te], pens_te, th)

    print(f"\n📊 TEST 평가 (threshold={th:.3f}, val 에서 선택)")
    print(fmt("XGBoost",  m_xgb))
    print(fmt("LSTM",     m_lst))
    print(fmt("Ensemble", m_ens))

    # ── 진단: persistence / onset / 그룹별 ─────────────────────────────────────
    ynow_lookup, group_map = load_context(features_path)
    diag = {}
    if ynow_lookup is not None:
        country_arr_te = np.array([country_index[i] for i in cid[te]])
        diag = diagnostics(country_arr_te, week_end[te], y[te], pens_te, th,
                           ynow_lookup, group_map)
    else:
        print("\n   ℹ️  --features 미지정 → persistence/onset/그룹 진단 생략 "
              "(features_weekly.parquet 경로를 주면 활성화)")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    report = {"alpha": alpha, "val_auc": val_auc, "threshold": th,
              "n_countries": n_countries, "n_features": n_features,
              "features": feat_names,
              "test": {"xgb": m_xgb, "lstm": m_lst, "ensemble": m_ens},
              "diagnostics": diag,
              "split_counts": {"train": int(tr.sum()), "val": int(va.sum()),
                               "test": int(te.sum())}}
    with open(os.path.join(output_dir, "ensemble_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "country_id": cid[te],
        "country": [country_index[i] for i in cid[te]],
        "week_end": week_end[te],
        "y_true": y[te],
        "p_xgb": pxgb_te, "p_lstm": plstm_te, "p_ensemble": pens_te,
        "pred": (pens_te >= th).astype(int),
    }).to_csv(os.path.join(output_dir, "ensemble_predictions.csv"),
              index=False, encoding="utf-8-sig")

    print(f"\n✅ 저장: {output_dir}/ensemble_report.json, ensemble_predictions.csv")


def main():
    p = argparse.ArgumentParser(description="XGBoost+LSTM 앙상블 학습/평가 (step6 입력)")
    p.add_argument("npz", nargs="?", help="lstm_sequences.npz (step6 출력)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--features", default=None,
                   help="features_weekly.parquet (persistence/onset/그룹 진단용)")
    p.add_argument("--out", default=None, help="출력 폴더 (기본: ./07_eval)")
    args = p.parse_args()

    npz = args.npz
    if not npz:
        print("lstm_sequences.npz 경로:")
        npz = input("  >> ").strip().strip('"').strip("'")
    if not os.path.exists(npz):
        print(f"❌ 파일 없음: {npz}")
        return

    out = args.out or os.path.join(os.path.dirname(npz) or ".", "07_eval")
    run(npz, out, args.epochs, args.features)


if __name__ == "__main__":
    main()