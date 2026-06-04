"""
step5_build_features_weekly.py
──────────────────────────────────────────────────────────────────────────────
주(week) 단위 F1~F15 피처 행렬 조립 (모델 입력 X 생성)

설계 문서 v3.1 의 피처를 "국가-주 단위" 하나의 패널로 합칩니다.
  - 주간 동적 피처   : acled_weekly.parquet (step1b 출력) 그대로 사용
  - 연간 맥락 피처   : wri_labeled_fixed.csv (step4/4b 출력) → 주 단위로 broadcast
  - 추가 연간 피처   : vdem_z / wb_z parquet (선택) → Anocracy, 인구
  - 시계열 파생      : 주간 사망자 기반 롤링/lag/가속도
  - 라벨 Y           : 연간 Y → 주 단위로 broadcast

핵심 처리:
  ① acled_weekly 를 주간 베이스 그리드로 로드
  ② (country, year) 키로 연간 피처 LEFT JOIN → 주 단위 broadcast
  ③ 전체 주간 달력으로 그리드 완성(빈 주 = 이벤트 0, 맥락은 ffill)
     → 시퀀스가 시간상 연속이 되도록 보장 (step6 에서 그대로 윈도우)
  ④ F1~F15 파생 계산 (없는 소스는 fallback, 경고 출력)
  ⑤ 시계열 파생 피처 계산 (완성된 그리드 위에서)
  ⑥ features_weekly.parquet / .csv 저장

F1~F15 매핑:
  F1  CII_pc      = CII / (Pop/1e6)         (인구 없으면 CII 그대로, 경고)
  F2  Z_CII       = Z_CII_90d               (acled_weekly)
  F3  ESC         = ESC                      (acled_weekly)
  F4  Anocracy    = 4·libdem·(1−libdem)      (vdem 없으면 0, 경고)
  F6  SFI         = SFI                       (연간 broadcast)
  F7  EconStress  = Econ                      (연간 broadcast)
  F9  Spillover   = spillover                 (acled_weekly)
  F10 RegCluster  = regcluster                (acled_weekly)
  F11 PolitShock  = Polit                     (연간 broadcast; REIGN 미연동 시 ElecViol 기반)
  F15 ForcedMig   = idp_count_z (proxy)       (UNHCR 난민유출·인구 연동 시 정식 수식)

사용법:
    python step5_build_features_weekly.py --acled acled_weekly.parquet \
           --wri wri_labeled_fixed.csv [--annual 02_normalized] [--out 05_features]
    python step5_build_features_weekly.py   (인자 없으면 경로 입력 프롬프트)

입력:
    acled_weekly.parquet   (필수, step1b 출력)
    wri_labeled_fixed.csv  (권장, step4/4b 출력 — Y + 연간 피처)
    02_normalized/         (선택, vdem_z.parquet / wb_z.parquet — Anocracy·인구)

출력:
    features_weekly.parquet  +  features_weekly.csv
──────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import argparse
import os

# ── 시계열 파생 윈도우 (주 단위) ──────────────────────────────────────────────
MA_SHORT = 4    # fatality_4w_ma
MA_LONG  = 12   # fatality_12w_ma
LAG_1    = 4    # event_count_lag4
LAG_2    = 8    # event_count_lag8
EPS      = 1e-8
Z_CLIP   = 8.0
REGCLUSTER_THRESHOLD = 1.5

# ── 연간 → 주 broadcast 대상 (wri_labeled_fixed.csv 안에서 가져옴) ───────────
# 주간 acled 에 이미 있는 컬럼(CII/ESC/Z_CII/regcluster 등)은 제외하고
# 연간에만 있는 맥락 피처만 가져온다.
ANNUAL_FEATURES = [
    "SFI", "Econ", "Spill", "Polit",          # WRI 서브인덱스 (연간)
    "idp_count_z", "disaster_flag_z", "disaster_deaths_z",
    "WRI", "theta_c",
]
# ?쇰꺼 怨꾩뿴
LABEL_COLS = ["Y", "is_war_year", "annual_deaths"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. 입력 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_acled(acled_path: str) -> pd.DataFrame:
    df = pd.read_parquet(acled_path)
    print(f"📂 ACLED 주간 베이스: {len(df):,}행 / {df['country_std'].nunique()}개국")
    print(f"   기간: {df['week_id'].min()} ~ {df['week_id'].max()}")
    return df

def infer_acled_z_path(acled_path: str, annual_dir: str | None = None) -> str | None:
    """Find acled_weekly_z.parquet when --acled-z is omitted."""
    candidates = []
    if annual_dir:
        candidates.append(os.path.join(annual_dir, "acled_weekly_z.parquet"))

    candidates.extend([
        os.path.join(os.getcwd(), "02_zscore", "acled_weekly_z.parquet"),
        os.path.join(os.path.dirname(acled_path), "acled_weekly_z.parquet"),
        acled_path.replace("acled_weekly.parquet", "acled_weekly_z.parquet"),
    ])

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def load_acled_z(acled_z_path: str | None) -> pd.DataFrame | None:
    """Load weekly ACLED z-score columns and normalize their names."""
    if not acled_z_path or not os.path.exists(acled_z_path):
        print("   acled_weekly_z.parquet 없음 -> raw ACLED rolling 값으로 z-score fallback 계산")
        return None

    z = pd.read_parquet(acled_z_path)
    rename = {
        "CII_roll7_z": "Z_CII_7d",
        "CII_roll90_z": "Z_CII_90d",
    }
    z = z.rename(columns={k: v for k, v in rename.items() if k in z.columns})

    keep = ["country_std", "week_id"]
    for col in ["Z_CII_7d", "Z_CII_90d", "spillover_z"]:
        if col in z.columns:
            keep.append(col)

    missing = [c for c in ["Z_CII_7d", "Z_CII_90d"] if c not in z.columns]
    if missing:
        print(f"   acled z 파일에 없는 필수 컬럼: {missing} -> 가능한 값만 병합")
    else:
        print(f"   ACLED z-score 로드: {os.path.basename(acled_z_path)}")

    return z[keep].drop_duplicates(subset=["country_std", "week_id"])


def merge_acled_z(df_week: pd.DataFrame, df_z: pd.DataFrame | None) -> pd.DataFrame:
    if df_z is None:
        return df_week

    drop_existing = [c for c in ["Z_CII_7d", "Z_CII_90d", "spillover_z"] if c in df_week.columns]
    df = df_week.drop(columns=drop_existing, errors="ignore")
    df = df.merge(df_z, on=["country_std", "week_id"], how="left")

    for col in ["Z_CII_7d", "Z_CII_90d"]:
        if col in df.columns:
            nz = int((df[col].fillna(0) != 0).sum())
            print(f"   {col}: nonzero {nz:,}/{len(df):,}")
    return df


def load_wri_annual(wri_path: str | None) -> pd.DataFrame | None:
    if not wri_path or not os.path.exists(wri_path):
        print("⚠️  wri_labeled_fixed.csv 없음 → 연간 맥락 피처·Y 없이 진행")
        return None
    df = pd.read_csv(wri_path, low_memory=False)
    keep = ["country_std", "year"]
    keep += [c for c in ANNUAL_FEATURES if c in df.columns]
    keep += [c for c in LABEL_COLS    if c in df.columns]
    missing = [c for c in (ANNUAL_FEATURES + LABEL_COLS) if c not in df.columns]
    if missing:
        print(f"   ℹ️  wri 파일에 없는 컬럼(스킵): {missing}")
    df = df[keep].drop_duplicates(subset=["country_std", "year"])
    print(f"📂 연간 맥락 피처: {len(df):,}행, 컬럼 {list(df.columns)}")
    return df


def load_annual_folder(folder: str | None, con_keys=("country_std", "year")) -> dict:
    """02_normalized 폴더에서 vdem_z / wb_z parquet 자동 탐색."""
    found = {"vdem": None, "wb": None}
    if not folder or not os.path.isdir(folder):
        return found
    for fpath in glob.glob(os.path.join(folder, "*.parquet")):
        fl = os.path.basename(fpath).lower()
        if found["vdem"] is None and ("vdem" in fl or "v-dem" in fl):
            found["vdem"] = fpath
        elif found["wb"] is None and ("wb" in fl or "worldbank" in fl or "world_bank" in fl):
            found["wb"] = fpath
    for k, v in found.items():
        if v:
            print(f"📂 추가 연간({k}): {os.path.basename(v)}")
    return found


# ─────────────────────────────────────────────────────────────────────────────
# 2. 연간 피처 병합 (주 단위 broadcast)
# ─────────────────────────────────────────────────────────────────────────────

def merge_annual(df_week: pd.DataFrame,
                 df_wri: pd.DataFrame | None,
                 annual: dict) -> pd.DataFrame:
    df = df_week.copy()

    # (1) wri_labeled_fixed 연간 피처 + Y
    if df_wri is not None:
        df = df.merge(df_wri, on=["country_std", "year"], how="left")

    # (2) V-Dem: Anocracy / ElecViol / corruption 원시값
    if annual.get("vdem"):
        v = pd.read_parquet(annual["vdem"])
        vcols = ["country_std", "year"]
        for c in ["v2x_libdem", "v2elintim", "v2x_corr"]:
            if c in v.columns:
                vcols.append(c)
        v = v[vcols].drop_duplicates(subset=["country_std", "year"])
        df = df.merge(v, on=["country_std", "year"], how="left")

    # (3) World Bank: 인구(있으면) + 경제 원시값
    if annual.get("wb"):
        w = pd.read_parquet(annual["wb"])
        wcols = ["country_std", "year"]
        # 인구 컬럼 후보 (소문자 표준화 가정)
        pop_candidates = [c for c in w.columns
                          if c.lower() in ("sp_pop_totl", "population", "pop")]
        for c in pop_candidates + ["ny_gdp_pcap_kd_zg", "fp_cpi_totl_zg",
                                   "ms_mil_xpnd_gd_zs"]:
            if c in w.columns and c not in wcols:
                wcols.append(c)
        w = w[wcols].drop_duplicates(subset=["country_std", "year"])
        df = df.merge(w, on=["country_std", "year"], how="left")
        df.attrs["pop_col"] = pop_candidates[0] if pop_candidates else None
    else:
        df.attrs["pop_col"] = None

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. 전체 주간 달력으로 그리드 완성
# ─────────────────────────────────────────────────────────────────────────────

def complete_weekly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    (국가 × 관측된 모든 주) 그리드를 완성한다.
    - 빈 주의 이벤트/카운트 피처 → 0
    - Z_CII / spillover / regcluster → ffill 후 0
    - 연간 맥락 피처·Y → 국가 내 ffill/bfill
    시퀀스가 시간상 연속이 되도록 보장 (step6 윈도우의 전제).
    """
    # 관측된 전체 주 달력
    weeks = (df[["week_id", "year", "week_num"]]
             .drop_duplicates()
             .sort_values(["year", "week_num"])
             .reset_index(drop=True))
    countries = df["country_std"].drop_duplicates().tolist()
    grp_map   = (df.dropna(subset=["group"])
                   .drop_duplicates("country_std")
                   .set_index("country_std")["group"].to_dict())

    # 국가 × 주 전체 조합
    full = (pd.MultiIndex
            .from_product([countries, weeks["week_id"]],
                          names=["country_std", "week_id"])
            .to_frame(index=False))
    full = full.merge(weeks, on="week_id", how="left")

    merged = full.merge(
        df.drop(columns=["year", "week_num", "group"], errors="ignore"),
        on=["country_std", "week_id"], how="left",
    )
    merged["group"] = merged["country_std"].map(grp_map).fillna("OTHER")

    before, after = len(df), len(merged)
    print(f"\n🗓️  주간 그리드 완성: {before:,} → {after:,}행 "
          f"({len(countries)}개국 × {len(weeks)}주)")

    # 채우기 규칙
    zero_fill = ["total_events", "weekly_fatalities", "battle_count",
                 "vac_count", "nonstate_count", "battle_ratio",
                 "civilian_targeting_ratio", "geographic_spread",
                 "CII", "CII_roll7", "CII_roll90"]
    ffill_then_zero = ["Z_CII_7d", "Z_CII_90d", "ESC", "spillover", "regcluster"]
    ffill_context   = [c for c in ANNUAL_FEATURES if c in merged.columns]
    ffill_context  += [c for c in ["v2x_libdem", "v2elintim", "v2x_corr",
                                    "ny_gdp_pcap_kd_zg", "fp_cpi_totl_zg",
                                    "ms_mil_xpnd_gd_zs"] if c in merged.columns]
    pop_col = df.attrs.get("pop_col")
    if pop_col and pop_col in merged.columns:
        ffill_context.append(pop_col)
    ffill_labels = [c for c in LABEL_COLS if c in merged.columns]

    merged = merged.sort_values(["country_std", "year", "week_num"])

    for c in zero_fill:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0.0)

    g = merged.groupby("country_std", sort=False)
    for c in ffill_then_zero:
        if c in merged.columns:
            merged[c] = g[c].ffill().fillna(0.0)
    for c in (ffill_context + ffill_labels):
        merged[c] = g[c].ffill().bfill()

    return merged.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. F1~F15 파생 계산
# ─────────────────────────────────────────────────────────────────────────────

def _country_zscore(df: pd.DataFrame, source_col: str, out_col: str) -> pd.DataFrame:
    g = df.groupby("country_std", sort=False)[source_col]
    mu = g.transform("mean")
    sd = g.transform("std").fillna(0.0)
    z = np.where(sd < 1e-6, 0.0, (df[source_col].astype(float) - mu) / sd)
    df[out_col] = np.clip(z, -Z_CLIP, Z_CLIP)
    return df


def ensure_weekly_risk_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure weekly Z_CII, ESC, and regcluster are live before F-feature mapping."""
    df = df.sort_values(["country_std", "year", "week_num"]).copy()

    if "Z_CII_7d" not in df.columns or df["Z_CII_7d"].fillna(0).abs().sum() == 0:
        if "CII_roll7" in df.columns:
            print("   Z_CII_7d 없음/상수 -> CII_roll7 기준 fallback z-score 계산")
            df = _country_zscore(df, "CII_roll7", "Z_CII_7d")
        else:
            raise ValueError("Z_CII_7d 또는 CII_roll7 컬럼이 필요합니다.")

    if "Z_CII_90d" not in df.columns or df["Z_CII_90d"].fillna(0).abs().sum() == 0:
        if "CII_roll90" in df.columns:
            print("   Z_CII_90d 없음/상수 -> CII_roll90 기준 fallback z-score 계산")
            df = _country_zscore(df, "CII_roll90", "Z_CII_90d")
        else:
            raise ValueError("Z_CII_90d 또는 CII_roll90 컬럼이 필요합니다.")

    df["ESC"] = (
        (df["Z_CII_7d"].astype(float) - df["Z_CII_90d"].astype(float))
        / (df["Z_CII_90d"].astype(float).abs() + EPS)
    ).clip(-Z_CLIP, Z_CLIP)

    if "regcluster" not in df.columns or df["regcluster"].fillna(0).abs().sum() == 0:
        high_share = (
            df.assign(_high=(df["Z_CII_90d"].astype(float) > REGCLUSTER_THRESHOLD).astype(float))
              .groupby(["group", "week_id"], sort=False)["_high"]
              .mean()
              .rename("regcluster")
              .reset_index()
        )
        df = df.drop(columns=["regcluster"], errors="ignore").merge(
            high_share, on=["group", "week_id"], how="left"
        )
        df["regcluster"] = df["regcluster"].fillna(0.0)
        print("   regcluster: group-week high-risk share 기준 계산")

    for col in ["Z_CII_90d", "ESC", "regcluster"]:
        nz = int((df[col].fillna(0) != 0).sum())
        print(f"   {col}: nonzero {nz:,}/{len(df):,}")

    return df.reset_index(drop=True)


def build_f_features(df: pd.DataFrame, pop_col: str | None) -> pd.DataFrame:
    df = df.copy()

    # F2 / F3 / F9 / F10  : 이미 acled_weekly 에 존재 → 별칭만 부여
    required_weekly = ["Z_CII_90d", "ESC", "regcluster"]
    missing_weekly = [c for c in required_weekly if c not in df.columns]
    if missing_weekly:
        raise ValueError(f"필수 주간 위험 피처가 없습니다: {missing_weekly}")

    df["F2_Z_CII"]      = df["Z_CII_90d"].fillna(0.0)
    df["F3_ESC"]        = df["ESC"].fillna(0.0)
    df["F9_Spillover"]  = df.get("spillover", 0.0)
    df["F10_RegCluster"] = df["regcluster"].fillna(0.0)

    # F1 CII_pc = CII / (Pop / 1e6)
    if pop_col and pop_col in df.columns and df[pop_col].notna().any():
        pop_millions = df[pop_col].replace(0, np.nan) / 1e6
        df["F1_CII_pc"] = (df["CII"] / pop_millions).fillna(0.0)
        print("   F1 CII_pc: 인구 정규화 적용")
    else:
        df["F1_CII_pc"] = df["CII"]
        print("   ⚠️  F1 CII_pc: 인구 컬럼 없음 → CII 원값 사용 (인구당 미적용)")

    # F4 Anocracy = 4 · libdem · (1 − libdem)
    if "v2x_libdem" in df.columns and df["v2x_libdem"].notna().any():
        lib = df["v2x_libdem"].clip(0, 1)
        df["F4_Anocracy"] = (4.0 * lib * (1.0 - lib)).fillna(0.0)
        print("   F4 Anocracy: V-Dem v2x_libdem 기반 계산")
    else:
        df["F4_Anocracy"] = 0.0
        print("   ⚠️  F4 Anocracy: V-Dem 없음 → 0.0 (SFI 내에는 이미 반영됨)")

    # F6 SFI / F7 EconStress / F11 PolitShock : 연간 broadcast
    df["F6_SFI"]        = df.get("SFI", 0.0)
    df["F7_EconStress"] = df.get("Econ", 0.0)
    if "Polit" in df.columns and df["Polit"].abs().sum() > 0:
        df["F11_PolitShock"] = df["Polit"]
        print("   F11 PolitShock: 연간 Polit(=ElecViol 기반) 사용 (REIGN 미연동)")
    elif "v2elintim" in df.columns:
        df["F11_PolitShock"] = df["v2elintim"].fillna(0.0)
        print("   ⚠️  F11 PolitShock: Polit 없음 → ElecViol(v2elintim) 대체")
    else:
        df["F11_PolitShock"] = 0.0
        print("   ⚠️  F11 PolitShock: 소스 없음 → 0.0 (REIGN 연동 필요)")

    # F15 ForcedMig : (IDP + Refugee)/Pop 의 proxy (현재 idp_count_z)
    if "idp_count_z" in df.columns and df["idp_count_z"].abs().sum() > 0:
        df["F15_ForcedMig"] = df["idp_count_z"]
        print("   F15 ForcedMig: idp_count_z proxy 사용 (난민유출·인구 연동 시 정식 수식)")
    else:
        df["F15_ForcedMig"] = 0.0
        print("   ⚠️  F15 ForcedMig: UNHCR 없음 → 0.0")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. 시계열 파생 피처 (완성 그리드 위)
# ─────────────────────────────────────────────────────────────────────────────

def build_timeseries_features(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in df.groupby("country_std", sort=False):
        g = g.sort_values(["year", "week_num"]).copy()
        fat = g["weekly_fatalities"].astype(float)
        ev  = g["total_events"].astype(float)

        g["fatality_4w_ma"]  = fat.rolling(MA_SHORT, min_periods=1).mean()
        g["fatality_12w_ma"] = fat.rolling(MA_LONG,  min_periods=1).mean()
        g["fatality_trend"]  = g["fatality_4w_ma"] - g["fatality_12w_ma"]
        g["event_count_lag4"] = ev.shift(LAG_1).fillna(0.0)
        g["event_count_lag8"] = ev.shift(LAG_2).fillna(0.0)
        g["conflict_accel"]   = g["fatality_trend"].diff().fillna(0.0)
        out.append(g)
    print("   시계열 파생: fatality_4w_ma / 12w_ma / trend / accel / event_lag4·8")
    return pd.concat(out, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 메인
# ─────────────────────────────────────────────────────────────────────────────

def run(acled_path, wri_path, annual_dir, output_dir, acled_z_path=None):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 64)
    print("  step5: 주 단위 F1~F15 피처 행렬 조립")
    print("=" * 64)

    df_week = load_acled(acled_path)
    acled_z_path = acled_z_path or infer_acled_z_path(acled_path, annual_dir)
    df_week = merge_acled_z(df_week, load_acled_z(acled_z_path))
    df_wri  = load_wri_annual(wri_path)
    annual  = load_annual_folder(annual_dir)

    print("\n🔗 연간 피처 병합 (주 단위 broadcast)...")
    df = merge_annual(df_week, df_wri, annual)
    pop_col = df.attrs.get("pop_col")

    df = complete_weekly_grid(df)
    print("\n?㎜ 二쇨컙 Z_CII / ESC / regcluster ?앹꽦...")
    df = ensure_weekly_risk_signals(df)
    df.attrs["pop_col"] = pop_col   # groupby 후 attrs 유실 방지

    print("\n🧮 F1~F15 파생 계산...")
    df = build_f_features(df, pop_col)

    print("\n🧮 시계열 파생 계산...")
    df = build_timeseries_features(df)


    # ── 컬럼 정리 ─────────────────────────────────────────────────────────────
    id_cols = ["country_std", "year", "week_num", "week_id", "group"]
    f_cols  = ["F1_CII_pc", "F2_Z_CII", "F3_ESC", "F4_Anocracy", "F6_SFI",
               "F7_EconStress", "F9_Spillover", "F10_RegCluster",
               "F11_PolitShock", "F15_ForcedMig"]
    raw_week = ["weekly_fatalities", "total_events", "battle_ratio",
                "civilian_targeting_ratio", "geographic_spread",
                "CII", "Z_CII_7d"]
    ts_cols = ["fatality_4w_ma", "fatality_12w_ma", "fatality_trend",
               "event_count_lag4", "event_count_lag8", "conflict_accel"]
    extra   = [c for c in ["disaster_flag_z", "disaster_deaths_z", "WRI",
                           "theta_c"] if c in df.columns]
    label   = [c for c in ["Y", "is_war_year"] if c in df.columns]

    ordered = id_cols + f_cols + raw_week + ts_cols + extra + label
    ordered = [c for c in ordered if c in df.columns]
    df = df[ordered]

    # ── 요약 ─────────────────────────────────────────────────────────────────
    print("\n📊 최종 피처 행렬")
    print(f"   행 수   : {len(df):,}")
    print(f"   국가 수 : {df['country_std'].nunique()}")
    print(f"   기간    : {df['week_id'].min()} ~ {df['week_id'].max()}")
    print(f"   피처 수 : {len([c for c in df.columns if c not in id_cols + label])}")
    if "Y" in df.columns:
        print(f"   Y=1 비율: {df['Y'].mean():.1%}")

    print("\n📊 F1~F15 기술통계")
    print(df[[c for c in f_cols if c in df.columns]].describe().round(3).T.to_string())

    # ── 저장 ─────────────────────────────────────────────────────────────────
    out_pq  = os.path.join(output_dir, "features_weekly.parquet")
    out_csv = os.path.join(output_dir, "features_weekly.csv")
    pq.write_table(pa.Table.from_pandas(df), out_pq, compression="snappy")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n✅ 저장 완료")
    print(f"   {out_pq}  ({os.path.getsize(out_pq)/1024/1024:.2f} MB)")
    print(f"   {out_csv}  ({os.path.getsize(out_csv)/1024/1024:.2f} MB)")
    print(f"\n📌 다음 단계: python step6_make_sequences.py {out_pq}")
    return out_pq


def main():
    p = argparse.ArgumentParser(description="주 단위 F1~F15 피처 행렬 조립")
    p.add_argument("--acled",  default=None, help="acled_weekly.parquet (필수)")
    p.add_argument("--acled-z", default=None, help="acled_weekly_z.parquet (권장, 없으면 자동 탐색/fallback)")
    p.add_argument("--wri",    default=None, help="wri_labeled_fixed.csv (권장)")
    p.add_argument("--annual", default=None, help="vdem_z/wb_z parquet 폴더 (선택)")
    p.add_argument("--out",    default=None, help="출력 폴더 (기본: ./05_features)")
    args = p.parse_args()

    acled = args.acled
    if not acled:
        print("acled_weekly.parquet 경로:")
        acled = input("  >> ").strip().strip('"').strip("'")
    if not os.path.exists(acled):
        print(f"❌ 파일 없음: {acled}")
        return

    wri = args.wri
    if wri is None:
        print("wri_labeled_fixed.csv 경로 (엔터=스킵):")
        ans = input("  >> ").strip().strip('"').strip("'")
        wri = ans if ans else None

    annual = args.annual
    if annual is None:
        print("vdem_z/wb_z parquet 폴더 (엔터=스킵):")
        ans = input("  >> ").strip().strip('"').strip("'")
        annual = ans if ans else None

    out = args.out or os.path.join(os.path.dirname(acled) or ".", "05_features")
    run(acled, wri, annual, out, args.acled_z)


if __name__ == "__main__":
    main()

