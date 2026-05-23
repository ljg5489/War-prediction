"""
step5_preprocess_monthly.py
──────────────────────────────────────────────────────────────────────────────
ML 학습용 월 단위 데이터셋 생성 (데이터 전처리 1단계)

처리 흐름:
  ① 월 단위 시간 인덱스 생성 (2013-01 ~ 2024-12 × 전 국가)
  ② 연간 데이터 → 월간 보간
       일반 피처    : 선형 보간 (limit_direction='both')
       이산 피처    : Forward Fill   (Y, is_war_year, theta_c, group)
       GDP 계열     : 3차 스플라인   (관측치 < 4이면 선형 fallback)
  ③ 주간 ACLED → 월간 집계 (있으면 LEFT JOIN)
  ④ 이상치 winsorize (국가별 Z > ±3.5)
  ⑤ 범주형 인코딩
       country_std → country_id (LabelEncode)
       group        → group_A / B / C / OTHER (One-Hot)
  ⑥ 국가별 μ_c, σ_c 저장 → 예측 시 재사용

입력:
  wri_labeled.csv         (필수, step4 출력)
  acled_weekly.parquet    (선택, step1b 출력)

출력:
  training_monthly.parquet : 학습용 월 단위 데이터
  country_stats.parquet    : 국가별 μ_c, σ_c
  feature_meta.json        : 피처 카테고리·인코딩 매핑

사용법:
    python step5_preprocess_monthly.py <wri_labeled.csv> [<acled_weekly.parquet>]
    python step5_preprocess_monthly.py   (인자 없으면 경로 입력 프롬프트)
──────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import json
import argparse
import os
from sklearn.preprocessing import LabelEncoder

# ── 설정 ─────────────────────────────────────────────────────────────────────
YEAR_MIN = 2013
YEAR_MAX = 2024
ZSCORE_THRESHOLD = 3.5    # 이상치 winsorize 기준

# ── 피처 카테고리 (보간 방법 결정) ───────────────────────────────────────────
# 이산형 — ffill (값이 그대로 유지되는 게 자연스러움)
FEATURES_FFILL = [
    "Y", "is_war_year", "theta_c", "group", "annual_deaths",
]

# GDP 계열 — 부드러운 추세, 스플라인 보간
FEATURES_SPLINE = [
    "GDP_Growth", "GDP_Growth_z",
    "Inflation", "Inflation_z",
    "Unemployment",
]

# 그 외 연속형 — 선형 보간
# (코드에서 자동 판별)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 월 단위 시간 인덱스 생성
# ─────────────────────────────────────────────────────────────────────────────

def create_monthly_grid(countries: list, year_min: int, year_max: int) -> pd.DataFrame:
    """모든 국가 × 모든 (연,월) 조합의 grid 생성."""
    grid = pd.MultiIndex.from_product(
        [countries, range(year_min, year_max + 1), range(1, 13)],
        names=["country_std", "year", "month"],
    ).to_frame(index=False)

    grid["yearmonth"] = grid["year"] * 100 + grid["month"]
    grid["date"] = pd.to_datetime(
        grid["year"].astype(str) + "-"
        + grid["month"].astype(str).str.zfill(2) + "-01"
    )
    return grid.sort_values(["country_std", "year", "month"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 연간 → 월간 보간
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_to_monthly(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    국가별로 월 단위 보간.

    각 피처를 카테고리별로 분류해 다른 방식 적용:
      - FEATURES_FFILL  : forward fill
      - FEATURES_SPLINE : 3차 스플라인 (관측치 < 4면 선형)
      - 그 외           : 선형 보간
    """
    df = df.sort_values(["country_std", "year", "month"]).copy()

    ffill_cols  = [c for c in feature_cols if c in FEATURES_FFILL]
    spline_cols = [c for c in feature_cols if c in FEATURES_SPLINE]
    linear_cols = [c for c in feature_cols
                   if c not in FEATURES_FFILL and c not in FEATURES_SPLINE]

    print(f"   ffill  ({len(ffill_cols)}개)  : {ffill_cols}")
    print(f"   spline ({len(spline_cols)}개) : {spline_cols}")
    print(f"   linear ({len(linear_cols)}개) : {linear_cols[:6]}{'...' if len(linear_cols)>6 else ''}")

    # ── Forward Fill ─────────────────────────────────────────────────────
    for col in ffill_cols:
        df[col] = df.groupby("country_std")[col].transform("ffill")
        df[col] = df.groupby("country_std")[col].transform("bfill")

    # ── 선형 보간 ─────────────────────────────────────────────────────────
    for col in linear_cols:
        df[col] = df.groupby("country_std")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )

    # ── 3차 스플라인 (관측치 4개 이상일 때만) ────────────────────────────
    for col in spline_cols:
        def spline_or_linear(s: pd.Series) -> pd.Series:
            n_valid = s.notna().sum()
            if n_valid >= 4:
                try:
                    return s.interpolate(method="spline", order=3,
                                         limit_direction="both")
                except Exception:
                    return s.interpolate(method="linear", limit_direction="both")
            return s.interpolate(method="linear", limit_direction="both")

        df[col] = df.groupby("country_std")[col].transform(spline_or_linear)

    # 잔여 결측 → 0
    df[feature_cols] = df[feature_cols].fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. ACLED 주간 → 월간 집계
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_acled_to_monthly(acled_path: str) -> pd.DataFrame | None:
    """ACLED 주간 데이터를 월 단위로 집계."""
    if not os.path.exists(acled_path):
        print(f"   ⚠️  ACLED 파일 없음 → 스킵: {acled_path}")
        return None

    print(f"   ACLED 로딩: {acled_path}")
    df = pd.read_parquet(acled_path)
    print(f"   원본: {len(df):,}행 (주 단위)")

    # week_num → month 근사 (ISO 주차 ÷ 4.33)
    if "week_num" in df.columns:
        df["month"] = ((df["week_num"] - 1) // 4 + 1).clip(1, 12).astype(int)
    else:
        print("   ⚠️  week_num 없음 → ACLED 스킵")
        return None

    # 합산 vs 평균 피처 분류
    sum_cols  = ["weekly_fatalities", "total_events",
                 "battle_count", "vac_count", "nonstate_count",
                 "geographic_spread"]
    mean_cols = ["CII", "CII_roll7", "CII_roll90",
                 "battle_ratio", "civilian_targeting_ratio",
                 "spillover"]

    agg_dict = {}
    for c in sum_cols:
        if c in df.columns:
            agg_dict[c] = "sum"
    for c in mean_cols:
        if c in df.columns:
            agg_dict[c] = "mean"

    monthly = (
        df.groupby(["country_std", "year", "month"])
        .agg(agg_dict)
        .reset_index()
    )

    # 컬럼명 접두사로 출처 명시 (다른 데이터와 충돌 방지)
    rename_map = {c: f"acled_{c}" for c in monthly.columns
                  if c not in ["country_std", "year", "month"]}
    monthly = monthly.rename(columns=rename_map)

    print(f"   집계: {len(monthly):,}행 (월 단위)")
    return monthly


# ─────────────────────────────────────────────────────────────────────────────
# 4. 이상치 winsorize (국가별 Z-score 기준)
# ─────────────────────────────────────────────────────────────────────────────

def winsorize(df: pd.DataFrame, cols: list, z_threshold: float = 3.5) -> pd.DataFrame:
    """국가별 평균·표준편차로 Z-score 계산, 극단값 → 임계값으로 clip."""
    df = df.copy()
    n_clipped_total = 0

    for col in cols:
        if col not in df.columns:
            continue

        mu = df.groupby("country_std")[col].transform("mean")
        sd = df.groupby("country_std")[col].transform("std").fillna(0)
        sd = sd.replace(0, 1e-8)
        z  = (df[col] - mu) / sd

        upper = mu + z_threshold * sd
        lower = mu - z_threshold * sd

        n_high = (z > z_threshold).sum()
        n_low  = (z < -z_threshold).sum()

        df[col] = np.where(z >  z_threshold, upper, df[col])
        df[col] = np.where(z < -z_threshold, lower, df[col])

        if n_high + n_low > 0:
            n_clipped_total += (n_high + n_low)

    print(f"   winsorize: {n_clipped_total:,}개 값 clip (|Z| > {z_threshold})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. 범주형 인코딩
# ─────────────────────────────────────────────────────────────────────────────

def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """country → LabelEncode, group → One-Hot."""
    df = df.copy()

    # country LabelEncode
    le = LabelEncoder()
    df["country_id"] = le.fit_transform(df["country_std"].astype(str))
    country_map = {c: int(i) for c, i in zip(le.classes_, le.transform(le.classes_))}

    # group One-Hot (A / B / C / OTHER)
    if "group" in df.columns:
        group_dummies = pd.get_dummies(
            df["group"].fillna("OTHER"),
            prefix="group",
            dummy_na=False,
        ).astype(int)
        df = pd.concat([df, group_dummies], axis=1)
        print(f"   group One-Hot: {list(group_dummies.columns)}")

    print(f"   country LabelEncode: {len(country_map)}개국")
    return df, country_map


# ─────────────────────────────────────────────────────────────────────────────
# 6. 국가별 μ_c, σ_c 계산
# ─────────────────────────────────────────────────────────────────────────────

def compute_country_stats(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """국가별 평균·표준편차 저장 (예측 시 새 데이터 정규화에 재사용)."""
    rows = []
    for country, sub in df.groupby("country_std"):
        row = {"country_std": country}
        for col in feature_cols:
            if col in sub.columns:
                row[f"{col}_mu"]    = float(sub[col].mean())
                row[f"{col}_sigma"] = float(sub[col].std())
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7. 메인 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(wri_path: str, acled_path: str | None, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. 연 단위 데이터 로딩 ─────────────────────────────────────────────
    print(f"\n📂 로딩: {wri_path}")
    df_annual = pd.read_csv(wri_path, low_memory=False)
    print(f"   행 수: {len(df_annual):,}  /  국가 수: {df_annual['country_std'].nunique()}")
    print(f"   연도: {int(df_annual['year'].min())}~{int(df_annual['year'].max())}")

    # 연도 필터
    df_annual = df_annual[df_annual["year"].between(YEAR_MIN, YEAR_MAX)].copy()
    countries = sorted(df_annual["country_std"].dropna().unique().tolist())

    # 보간 대상 피처 (메타 컬럼 제외)
    meta_cols = {"country_std", "year", "month", "yearmonth", "date", "group"}
    feature_cols = [c for c in df_annual.columns if c not in meta_cols]
    print(f"   대상 피처 {len(feature_cols)}개")

    # ── 2. 월 단위 grid 생성 ──────────────────────────────────────────────
    print(f"\n── STEP 1: 월 단위 grid 생성 ─────────────────────────────────")
    grid = create_monthly_grid(countries, YEAR_MIN, YEAR_MAX)
    print(f"   grid: {len(grid):,}행 = {len(countries)}국 × {(YEAR_MAX-YEAR_MIN+1)}년 × 12월")

    # ── 3. 연 단위 → 월 단위 LEFT JOIN ─────────────────────────────────────
    print(f"\n── STEP 2: 연간 → 월간 JOIN ───────────────────────────────────")
    monthly = grid.merge(df_annual, on=["country_std", "year"], how="left",
                          suffixes=("", "_annual"))
    if "group_annual" in monthly.columns:
        monthly["group"] = monthly["group"].fillna(monthly["group_annual"])
        monthly = monthly.drop(columns=["group_annual"])
    print(f"   JOIN 후: {len(monthly):,}행")

    # ── 4. 보간 ────────────────────────────────────────────────────────────
    print(f"\n── STEP 3: 보간 ──────────────────────────────────────────────")
    monthly = interpolate_to_monthly(monthly, feature_cols)

    # ── 5. ACLED 주간 → 월간 (선택) ────────────────────────────────────────
    if acled_path:
        print(f"\n── STEP 4: ACLED 주간 → 월간 집계 ────────────────────────────")
        acled_monthly = aggregate_acled_to_monthly(acled_path)
        if acled_monthly is not None:
            monthly = monthly.merge(
                acled_monthly,
                on=["country_std", "year", "month"],
                how="left",
            )
            acled_cols = [c for c in monthly.columns if c.startswith("acled_")]
            monthly[acled_cols] = monthly[acled_cols].fillna(0)
            print(f"   ACLED 피처 {len(acled_cols)}개 병합")

    # ── 6. 이상치 winsorize ────────────────────────────────────────────────
    print(f"\n── STEP 5: 이상치 winsorize ──────────────────────────────────")
    numeric_cols = monthly.select_dtypes(include=[np.number]).columns.tolist()
    skip = {"year", "month", "yearmonth", "Y", "is_war_year"}
    winsor_cols = [c for c in numeric_cols if c not in skip]
    monthly = winsorize(monthly, winsor_cols, ZSCORE_THRESHOLD)

    # ── 7. 범주형 인코딩 ───────────────────────────────────────────────────
    print(f"\n── STEP 6: 범주형 인코딩 ────────────────────────────────────")
    monthly, country_map = encode_categorical(monthly)

    # ── 8. 국가별 μ_c, σ_c ─────────────────────────────────────────────────
    print(f"\n── STEP 7: 국가별 μ_c, σ_c 저장 ──────────────────────────────")
    stats = compute_country_stats(monthly, feature_cols)
    print(f"   stats: {len(stats):,}행 × {len(stats.columns)}컬럼")

    # ── 9. 저장 ─────────────────────────────────────────────────────────────
    print(f"\n── STEP 8: 저장 ──────────────────────────────────────────────")
    train_path = os.path.join(output_dir, "training_monthly.parquet")
    stats_path = os.path.join(output_dir, "country_stats.parquet")
    meta_path  = os.path.join(output_dir, "feature_meta.json")

    monthly.to_parquet(train_path, compression="snappy", index=False)
    stats.to_parquet(stats_path, compression="snappy", index=False)

    feature_meta = {
        "year_range":      [YEAR_MIN, YEAR_MAX],
        "n_countries":     len(countries),
        "country_mapping": country_map,
        "interpolation": {
            "ffill_features":  [c for c in FEATURES_FFILL  if c in feature_cols],
            "spline_features": [c for c in FEATURES_SPLINE if c in feature_cols],
            "linear_features": [c for c in feature_cols
                                if c not in FEATURES_FFILL and c not in FEATURES_SPLINE],
        },
        "zscore_threshold": ZSCORE_THRESHOLD,
        "feature_columns":  feature_cols,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, ensure_ascii=False, indent=2)

    # ── 결과 요약 ──────────────────────────────────────────────────────────
    print(f"\n📊 최종 결과")
    print(f"   training_monthly : {len(monthly):,}행 × {len(monthly.columns)}컬럼")
    print(f"   country_stats    : {len(stats):,}행 × {len(stats.columns)}컬럼")

    if "Y" in monthly.columns:
        y_dist = monthly["Y"].value_counts(normalize=True).round(3)
        print(f"\n📊 Y 레이블 분포 (월 단위)")
        print(f"   Y=0: {y_dist.get(0, 0):.1%}")
        print(f"   Y=1: {y_dist.get(1, 0):.1%}")

    print(f"\n✅ 저장 완료")
    print(f"   {train_path}  ({os.path.getsize(train_path)/1024/1024:.2f} MB)")
    print(f"   {stats_path}  ({os.path.getsize(stats_path)/1024:.1f} KB)")
    print(f"   {meta_path}")

    print(f"\n📌 다음 단계: ML 모델 학습 (XGBoost + LSTM)")
    print(f"   training_monthly.parquet 를 학습 입력으로 사용")
    print(f"   country_stats.parquet 는 예측 시 새 데이터 정규화에 재사용")


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="월 단위 ML 학습 데이터 생성 (전처리 1단계)"
    )
    parser.add_argument("wri_path",   nargs="?", help="wri_labeled.csv 경로")
    parser.add_argument("acled_path", nargs="?", default=None,
                        help="acled_weekly.parquet 경로 (선택)")
    parser.add_argument("--out",      default=None,
                        help="출력 폴더 (기본: ./04_ml_data)")
    args = parser.parse_args()

    wri_path   = args.wri_path
    acled_path = args.acled_path

    if not wri_path:
        print("wri_labeled.csv 경로:")
        wri_path = input("  >> ").strip().strip('"').strip("'")
    if not acled_path:
        print("acled_weekly.parquet 경로 (엔터=스킵):")
        ans = input("  >> ").strip().strip('"').strip("'")
        acled_path = ans if ans else None

    if not os.path.exists(wri_path):
        print(f"❌ 파일 없음: {wri_path}")
        return

    output_dir = args.out
    if not output_dir:
        default_out = os.path.join(os.path.dirname(wri_path) or ".", "04_ml_data")
        print(f"출력 폴더 (엔터={default_out}):")
        output_dir = input("  >> ").strip().strip('"').strip("'") or default_out

    preprocess(wri_path, acled_path, output_dir)


if __name__ == "__main__":
    main()