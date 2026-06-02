"""
step6_make_sequences.py
──────────────────────────────────────────────────────────────────────────────
주간 피처 행렬 → LSTM 시퀀스 + XGBoost 평면 테이블

설정: window = 13주 (=90일),  stride = 4주 (=30일),  horizon = 13주 (≈3개월 후)

처리:
  ① features_weekly.parquet 로드 (step5 출력)
  ② 국가별로 시간순 정렬 (이미 연속 그리드 — step5 가 보장)
  ③ 슬라이딩 윈도우(13주, 4주 이동)로 (N, 13, F) 시퀀스 생성
  ④ 라벨: 윈도우 끝 + horizon(13주) 시점의 Y  → "약 3개월 후" 예측
       horizon 위치가 데이터 범위를 벗어나면 해당 윈도우는 제외
  ⑤ country_id 별도 배열로 생성 (LSTM Embedding 입력용)
  ⑥ XGBoost 평면 테이블 = 윈도우 끝 시점 피처 + 윈도우 통계(평균/추세)
  ⑦ 시간 기반 train/val/test 분할 인덱스 생성 (연도 경계, 미래 누수 방지)

출력:
  lstm_sequences.npz   X(N,13,F) / y(N) / country_id(N) / week_end(N) / feature_names
  xgb_features.parquet 평면 피처 + Y + split
  split_summary.json   분할/라벨 분포 요약

사용법:
    python step6_make_sequences.py <features_weekly.parquet> [--out 06_model_input]
    python step6_make_sequences.py   (인자 없으면 경로 입력 프롬프트)
──────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import json
import argparse
import os

# ── 시퀀스 설정 (주 단위) ─────────────────────────────────────────────────────
WINDOW  = 13   # 윈도우 길이 (주) = 90일
STRIDE  = 4    # 이동 간격 (주)   = 30일
HORIZON = 13   # 예측 시차 (주)   ≈ 3개월 후 라벨

# 시간 기반 분할 경계 (윈도우 '끝 연도' 기준)
VAL_START_YEAR  = 2022   # 2022 이전 = train, 2022 = val, 2023+ = test
TEST_START_YEAR = 2023

# 시퀀스에서 제외할 컬럼 (식별자 / 라벨 / 누수성)
EXCLUDE = {
    "country_std", "year", "week_num", "week_id", "group",
    "Y", "is_war_year",
    "WRI", "theta_c",          # WRI·임계값은 라벨 직접 유래 → 피처에서 제외(누수 방지)
}


def get_feature_cols(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns
            if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def make_sequences(df: pd.DataFrame, feat_cols: list):
    """국가별 슬라이딩 윈도우 → (N,WINDOW,F) + 라벨/메타."""
    X_list, y_list, cid_list, wend_list, country_list = [], [], [], [], []
    skipped_no_label = 0

    countries = sorted(df["country_std"].unique())
    cid_map = {c: i for i, c in enumerate(countries)}

    for country, g in df.groupby("country_std", sort=False):
        g = g.sort_values(["year", "week_num"]).reset_index(drop=True)
        feats = np.nan_to_num(g[feat_cols].to_numpy(dtype=np.float32), nan=0.0)
        years = g["year"].to_numpy()
        wids  = g["week_id"].to_numpy()
        has_y = "Y" in g.columns
        yv    = g["Y"].to_numpy() if has_y else None
        n     = len(g)

        start = 0
        while start + WINDOW <= n:
            end = start + WINDOW - 1          # 윈도우 마지막 인덱스
            label_idx = end + HORIZON         # 예측 대상 시점
            if has_y:
                # horizon 이 범위 밖이거나 라벨이 NaN(=라벨 없는 국가/기간)이면 제외
                if label_idx >= n or pd.isna(yv[label_idx]):
                    skipped_no_label += 1
                    start += STRIDE
                    continue
                label = int(yv[label_idx])
            else:
                label = -1                    # 라벨 없음(추론 전용)

            X_list.append(feats[start:start + WINDOW])
            y_list.append(label)
            cid_list.append(cid_map[country])
            wend_list.append(wids[end])
            country_list.append(country)
            start += STRIDE

    X = np.stack(X_list).astype(np.float32) if X_list else np.empty((0, WINDOW, len(feat_cols)), np.float32)
    y = np.array(y_list, dtype=np.int64)
    cid = np.array(cid_list, dtype=np.int64)
    wend = np.array(wend_list)
    country_arr = np.array(country_list)

    print(f"   시퀀스 생성: {X.shape[0]:,}개  shape={X.shape}")
    if skipped_no_label:
        print(f"   ℹ️  horizon({HORIZON}주) 범위 밖으로 제외된 윈도우: {skipped_no_label:,}")
    return X, y, cid, wend, country_arr, cid_map


def make_xgb_flat(X, y, cid, wend, country_arr, feat_cols):
    """윈도우 끝 시점 값 + 윈도우 평균 + 추세(마지막-처음) → 평면 피처."""
    last  = X[:, -1, :]                       # 윈도우 끝 시점
    mean_ = X.mean(axis=1)                    # 윈도우 평균
    trend = X[:, -1, :] - X[:, 0, :]          # 추세

    data = {"country_std": country_arr, "country_id": cid, "week_end": wend}
    for j, c in enumerate(feat_cols):
        data[f"{c}__last"]  = last[:, j]
        data[f"{c}__mean"]  = mean_[:, j]
        data[f"{c}__trend"] = trend[:, j]
    data["Y"] = y
    return pd.DataFrame(data)


def assign_split(wend: np.ndarray) -> np.ndarray:
    """윈도우 끝 연도 기준 시간 분할 (train/val/test)."""
    end_year = np.array([int(str(w)[:4]) for w in wend])
    split = np.where(end_year >= TEST_START_YEAR, "test",
             np.where(end_year >= VAL_START_YEAR, "val", "train"))
    return split


def run(features_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 64)
    print(f"  step6: LSTM 시퀀스 (window={WINDOW}주 / stride={STRIDE}주 / "
          f"horizon={HORIZON}주)")
    print("=" * 64)

    df = pd.read_parquet(features_path)
    print(f"📂 입력: {len(df):,}행 / {df['country_std'].nunique()}개국 "
          f"/ {df['week_id'].min()}~{df['week_id'].max()}")

    feat_cols = get_feature_cols(df)
    print(f"\n🧬 피처 {len(feat_cols)}개:")
    print("   " + ", ".join(feat_cols))

    print("\n🔢 시퀀스 생성...")
    X, y, cid, wend, country_arr, cid_map = make_sequences(df, feat_cols)
    if X.shape[0] == 0:
        print("❌ 생성된 시퀀스가 없습니다. WINDOW/HORIZON 또는 데이터 길이를 확인하세요.")
        return

    split = assign_split(wend)

    if (y >= 0).any():
        print("\n📊 라벨 분포")
        for s in ["train", "val", "test"]:
            m = split == s
            if m.sum():
                pos = (y[m] == 1).mean()
                print(f"   {s:<5}: {m.sum():>6,}개  Y=1 {pos:.1%}")

    # ── LSTM 저장 ─────────────────────────────────────────────────────────────
    lstm_path = os.path.join(output_dir, "lstm_sequences.npz")
    np.savez_compressed(
        lstm_path,
        X=X, y=y, country_id=cid, week_end=wend, split=split,
        feature_names=np.array(feat_cols),
        country_index=np.array(list(cid_map.keys())),
    )
    print(f"\n✅ LSTM 시퀀스: {lstm_path}")
    print(f"   X={X.shape}  y={y.shape}  country_id={cid.shape}  "
          f"n_countries={len(cid_map)}")

    # ── XGBoost 평면 테이블 ────────────────────────────────────────────────────
    flat = make_xgb_flat(X, y, cid, wend, country_arr, feat_cols)
    flat["split"] = split
    xgb_path = os.path.join(output_dir, "xgb_features.parquet")
    flat.to_parquet(xgb_path, index=False)
    print(f"\n✅ XGBoost 평면 테이블: {xgb_path}")
    print(f"   shape={flat.shape}  (피처 {len(feat_cols)}개 × 3통계 + 메타)")

    # ── 요약 저장 ──────────────────────────────────────────────────────────────
    summary = {
        "window_weeks": WINDOW, "stride_weeks": STRIDE, "horizon_weeks": HORIZON,
        "n_sequences": int(X.shape[0]),
        "n_features": len(feat_cols),
        "n_countries": len(cid_map),
        "feature_names": feat_cols,
        "split_counts": {s: int((split == s).sum()) for s in ["train", "val", "test"]},
        "val_start_year": VAL_START_YEAR, "test_start_year": TEST_START_YEAR,
    }
    with open(os.path.join(output_dir, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n📌 다음 단계")
    print(f"   - XGBoost: xgb_features.parquet (split 컬럼으로 train/val/test 분리)")
    print(f"   - LSTM   : lstm_sequences.npz  (X + country_id 임베딩 입력)")
    print(f"   - 앙상블 : 두 모델 확률 → Soft Voting")


def main():
    p = argparse.ArgumentParser(description="주간 피처 → LSTM 시퀀스 + XGBoost 평면")
    p.add_argument("features", nargs="?", help="features_weekly.parquet (step5 출력)")
    p.add_argument("--out", default=None, help="출력 폴더 (기본: ./06_model_input)")
    args = p.parse_args()

    fp = args.features
    if not fp:
        print("features_weekly.parquet 경로:")
        fp = input("  >> ").strip().strip('"').strip("'")
    if not os.path.exists(fp):
        print(f"❌ 파일 없음: {fp}")
        return

    out = args.out or os.path.join(os.path.dirname(fp) or ".", "06_model_input")
    run(fp, out)


if __name__ == "__main__":
    main()