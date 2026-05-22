"""
step1b_aggregate_acled_weekly.py
──────────────────────────────────────────────────────────────────────────────
ACLED 이벤트 단위 → 국가+주(week) 단위 RAW 집계

이 스크립트는 원본값(raw)만 출력합니다.
  Z_CII, ESC  → step2 정규화 후 step3에서 계산
  regcluster  → step3에서 Z_CII 기반으로 계산

출력 피처:
  [5.1 분쟁강도 — raw]
    weekly_fatalities         주간 총 사망자
    total_events              주간 총 이벤트 수
    battle_count              전투(Battles) 이벤트 수
    vac_count                 민간인 대상 폭력 이벤트 수
    nonstate_count            비국가 폭력 이벤트 수
    battle_ratio              전투 비율 (battle_count / total_events)
    civilian_targeting_ratio  민간인 폭력 비율 (vac_count / total_events)
    geographic_spread         주간 고유 좌표 수 (지리적 확산)
    CII                       주간 분쟁 강도 (= weekly_fatalities)
    CII_roll7                 7일 롤링 평균 CII  (1주, raw)
    CII_roll90                90일 롤링 평균 CII (13주, raw)

  [5.4 지역전파 — raw]
    spillover                 인접국 가중 CII 합산
                              = Σ CII_n / (d_cn^0.5 + 1)

  ※ Z_CII_7d / Z_CII_90d / ESC / regcluster 는 step2 정규화 후 step3에서 생성

시간 컬럼 자동 감지:
    'week'       우선 사용 (YYYY-WW 형식)
    'event_date' 에서 연도·주차 추출 (YYYY-MM-DD 형식)

위경도 컬럼 자동 감지:
    'latitude'/'longitude' 또는 'lat'/'lon'

파이프라인 순서:
    normalize_and_group.py → step1b → step2 → step3 → step4

    step2 FEATURE_COLS 에 아래 항목 추가 필요:
        "acled_weekly": [
            "weekly_fatalities", "CII", "CII_roll7", "CII_roll90",
            "battle_ratio", "civilian_targeting_ratio",
            "geographic_spread", "spillover",
        ],

사용법:
    python step1b_aggregate_acled_weekly.py <grouped_acled.csv>
    python step1b_aggregate_acled_weekly.py <csv> --out <출력폴더>
    python step1b_aggregate_acled_weekly.py   (인자 없으면 경로 입력 프롬프트)

입력:  normalize_and_group.py 출력 CSV (country_std, group 컬럼 포함)
출력:  acled_weekly.parquet  +  acled_weekly.csv
──────────────────────────────────────────────────────────────────────────────
"""

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import argparse
import os

# ── 상수 ─────────────────────────────────────────────────────────────────────
EPS            = 1e-8
ROLL_SHORT_WKS = 1    # 단기 롤링 윈도우 (≈7일  = 1주)
ROLL_LONG_WKS  = 13   # 장기 롤링 윈도우 (≈90일 = 13주)

# ── 인접국 & 국경 거리(km) ────────────────────────────────────────────────────
# Spillover_c,t = Σ_{n∈N(c)} CII_n,t / (d_cn^0.5 + 1)
# 프로젝트 30개국 + 예측 대상(아프가니스탄) 포함
# 인접국이 데이터에 없으면 자동 스킵 (0 기여)
NEIGHBORS = {
    "Afghanistan":              {"Pakistan": 2430, "Iran": 921},
    "Syria":                    {"Iraq": 605, "Lebanon": 375},
    "Yemen":                    {},
    "Somalia":                  {"Ethiopia": 1600},
    "Myanmar":                  {},
    "Ethiopia":                 {"Somalia": 1600, "Sudan": 744, "South Sudan": 883},
    "South Sudan":              {"Sudan": 2158, "Ethiopia": 883,
                                 "DR Congo": 628, "Central African Republic": 1165},
    "Mali":                     {},
    "DR Congo":                 {"South Sudan": 628, "Central African Republic": 1577},
    "Ukraine":                  {},
    "Iraq":                     {"Syria": 605},
    "Pakistan":                 {"Afghanistan": 2430},
    "Nigeria":                  {},
    "Venezuela":                {"Colombia": 2341},
    "Sudan":                    {"Ethiopia": 744, "South Sudan": 2158},
    "Central African Republic": {"South Sudan": 1165, "DR Congo": 1577},
    "Taiwan":                   {},
    "Haiti":                    {},
    "Lebanon":                  {"Syria": 375},
    "Colombia":                 {"Venezuela": 2341, "Ecuador": 708},
    "Ecuador":                  {"Colombia": 708},
    "Norway":                   {},
    "Switzerland":              {},
    "Japan":                    {},
    "South Korea":              {},
    "Portugal":                 {},
    "Uruguay":                  {},
    "Botswana":                 {},
    "Mongolia":                 {},
    "Canada":                   {},
    "Germany":                  {},
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. 컬럼 자동 감지
# ─────────────────────────────────────────────────────────────────────────────

def detect_time_col(cols: list) -> str | None:
    """'week' 또는 'event_date' 컬럼 자동 감지."""
    col_map = {c.lower(): c for c in cols}
    return col_map.get("week") or col_map.get("event_date")


def detect_latlon_cols(cols: list) -> tuple:
    """위경도 컬럼 자동 감지. (lat_col, lon_col) 반환. 없으면 (None, None)."""
    col_map = {c.lower(): c for c in cols}
    lat = col_map.get("CENTROID_LATITUDE") or col_map.get("lat")
    lon = col_map.get("CENTROID_LONGITUDE") or col_map.get("lon")
    return lat, lon


# ─────────────────────────────────────────────────────────────────────────────
# 2. 시간 컬럼 전처리
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_time(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    시간 컬럼 → year(int), week_num(int), week_id(str 'YYYY-WW') 생성.
    'week'       : 'YYYY-WW' 형식 직접 파싱
    'event_date' : 'YYYY-MM-DD' → ISO 주차 추출
    """
    df = df.copy()
    sample = str(df[time_col].dropna().iloc[0]).strip()

    if len(sample) <= 8 and "-" in sample:
        # YYYY-WW 형식
        parts          = df[time_col].astype(str).str.strip().str.split("-", expand=True)
        df["year"]     = parts[0].astype(int)
        df["week_num"] = parts[1].astype(int)
        print(f"  ℹ️  '{time_col}' → YYYY-WW 형식 파싱")
    else:
        # YYYY-MM-DD 형식
        dt             = pd.to_datetime(df[time_col], errors="coerce")
        iso            = dt.dt.isocalendar()
        df["year"]     = iso.year.astype(int)
        df["week_num"] = iso.week.astype(int)
        print(f"  ℹ️  '{time_col}' → ISO 주차 파싱")

    df["week_id"] = (
        df["year"].astype(str) + "-"
        + df["week_num"].astype(str).str.zfill(2)
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. DuckDB 주간 집계 쿼리
# ─────────────────────────────────────────────────────────────────────────────

def build_weekly_agg_query(lat_col: str | None, lon_col: str | None) -> str:
    """
    국가+week_id 단위 집계.
    lat/lon 없으면 geographic_spread = 0.
    """
    if lat_col and lon_col:
        geo_expr = (
            f"COUNT(DISTINCT CONCAT("
            f"CAST(ROUND({lat_col}, 2) AS VARCHAR), ',',"
            f"CAST(ROUND({lon_col}, 2) AS VARCHAR)"
            f")) AS geographic_spread"
        )
    else:
        geo_expr = "0 AS geographic_spread"

    return f"""
SELECT
    country_std,
    year,
    week_num,
    week_id,
    COALESCE("group", 'OTHER')                                         AS "group",

    COUNT(*)                                                           AS total_events,
    COALESCE(SUM(fatalities), 0)                                       AS weekly_fatalities,

    COUNT(CASE WHEN event_type = 'Battles'
               THEN 1 END)                                             AS battle_count,
    COUNT(CASE WHEN event_type = 'Violence against civilians'
               THEN 1 END)                                             AS vac_count,
    COUNT(CASE WHEN event_type IN (
                   'Riots', 'Explosions/Remote violence',
                   'Strategic developments', 'Protests')
               THEN 1 END)                                             AS nonstate_count,

    {geo_expr}

FROM df_raw
GROUP BY country_std, year, week_num, week_id, "group"
ORDER BY country_std, year, week_num
"""


# ─────────────────────────────────────────────────────────────────────────────
# 4. 롤링 파생 피처 계산 (raw값만, Z-score 없음)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    국가별 롤링 피처 계산.
    ※ Z-score(Z_CII), ESC는 step2 정규화 후 step3에서 계산.
    """
    results = []

    for country, grp in df.groupby("country_std", sort=False):
        g = grp.sort_values(["year", "week_num"]).copy()

        # ── 비율 피처 ──────────────────────────────────────────────────────
        g["battle_ratio"]             = (
            g["battle_count"] / (g["total_events"] + EPS)
        )
        g["civilian_targeting_ratio"] = (
            g["vac_count"] / (g["total_events"] + EPS)
        )

        # ── CII (raw 분쟁 강도) ────────────────────────────────────────────
        g["CII"] = g["weekly_fatalities"].astype(float)

        # ── 롤링 평균 (raw — 정규화 전) ────────────────────────────────────
        # step2가 이 값들을 국가별 Z-score로 변환
        # step3가 Z_CII_7d / Z_CII_90d / ESC를 계산
        g["CII_roll7"]  = (
            g["CII"]
            .rolling(window=ROLL_SHORT_WKS, min_periods=1)
            .mean()
        )
        g["CII_roll90"] = (
            g["CII"]
            .rolling(window=ROLL_LONG_WKS, min_periods=1)
            .mean()
        )

        results.append(g)

    return pd.concat(results, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Spillover 계산 (raw CII 기반)
# ─────────────────────────────────────────────────────────────────────────────

def compute_spillover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spillover_c,t = Σ_{n∈N(c)} CII_n,t / (d_cn^0.5 + 1)

    raw CII 기준으로 계산.
    regcluster(Z_CII_90d 기반)는 step3에서 계산.
    """
    # week_id × country_std → CII 조회 테이블
    cii_lookup = (
        df.groupby(["week_id", "country_std"])["CII"]
        .mean()
    )

    spillover_vals = []

    for _, row in df.iterrows():
        country   = row["country_std"]
        week_id   = row["week_id"]
        neighbors = NEIGHBORS.get(country, {})

        spill = 0.0
        for nbr, dist in neighbors.items():
            try:
                nbr_cii = cii_lookup.loc[(week_id, nbr)]
                spill  += float(nbr_cii) / (dist ** 0.5 + 1)
            except KeyError:
                pass  # 인접국 데이터 없으면 0 기여

        spillover_vals.append(spill)

    df = df.copy()
    df["spillover"] = spillover_vals
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. 요약 출력
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print(f"\n📊 집계 결과")
    print(f"   행 수   : {len(df):,}")
    print(f"   국가 수 : {df['country_std'].nunique()}")
    print(f"   기간    : {df['week_id'].min()} ~ {df['week_id'].max()}")

    print(f"\n📊 그룹별 집계")
    print(
        df.groupby("group")[
            ["total_events", "weekly_fatalities", "battle_count", "vac_count"]
        ].sum().reset_index().to_string(index=False)
    )

    print(f"\n📊 피처 기술통계")
    feat_cols = [
        "CII", "CII_roll7", "CII_roll90",
        "battle_ratio", "civilian_targeting_ratio",
        "geographic_spread", "spillover",
    ]
    existing = [c for c in feat_cols if c in df.columns]
    print(df[existing].describe().round(4).to_string())

    print(f"\n🔥 CII 상위 10개 (국가+주)")
    print(
        df.nlargest(10, "CII")[
            ["country_std", "year", "week_num", "group",
             "weekly_fatalities", "CII", "CII_roll90", "spillover"]
        ].reset_index(drop=True).to_string(index=False)
    )

    print(f"\n📌 다음 단계 안내")
    print(f"   1. step2_zscore_duckdb.py 의 FEATURE_COLS 에 acled_weekly 항목 추가:")
    print(f'      "acled_weekly": [')
    print(f'          "weekly_fatalities", "CII", "CII_roll7", "CII_roll90",')
    print(f'          "battle_ratio", "civilian_targeting_ratio",')
    print(f'          "geographic_spread", "spillover",')
    print(f'      ],')
    print(f"   2. step2 실행 → acled_weekly_z.parquet 생성")
    print(f"   3. step3 실행 → Z_CII / ESC / regcluster 자동 계산")


# ─────────────────────────────────────────────────────────────────────────────
# 7. 메인 처리
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(input_path: str, output_dir: str):

    # ── 로딩 ─────────────────────────────────────────────────────────────────
    print(f"\n📂 로딩: {input_path}")
    df_raw = pd.read_csv(input_path, low_memory=False)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    print(f"   행 수: {len(df_raw):,}  /  컬럼: {list(df_raw.columns)}")

    # ── 필수 컬럼 확인 ────────────────────────────────────────────────────────
    if "country_std" not in df_raw.columns:
        print("❌ country_std 없음 → normalize_and_group.py 를 먼저 실행하세요.")
        return
    if "fatalities" not in df_raw.columns or "event_type" not in df_raw.columns:
        print("❌ ACLED 필수 컬럼(fatalities, event_type) 없음.")
        return

    # ── 시간 컬럼 감지 & 전처리 ──────────────────────────────────────────────
    time_col = detect_time_col(df_raw.columns.tolist())
    if not time_col:
        print("❌ 시간 컬럼(week 또는 event_date)을 찾을 수 없습니다.")
        return
    df_raw = preprocess_time(df_raw, time_col)

    # ── 위경도 컬럼 감지 ─────────────────────────────────────────────────────
    lat_col, lon_col = detect_latlon_cols(df_raw.columns.tolist())
    if lat_col and lon_col:
        print(f"  ℹ️  위경도: '{lat_col}', '{lon_col}'")
    else:
        print(f"  ⚠️  위경도 컬럼 없음 → geographic_spread = 0")

    # ── DuckDB 주간 집계 ─────────────────────────────────────────────────────
    print(f"\n🔢 주간 집계 중 (DuckDB)...")
    con = duckdb.connect()
    con.register("df_raw", df_raw)

    print(f"\n📊 원본 현황")
    print(con.execute("""
        SELECT COUNT(*) AS total_events,
               COUNT(DISTINCT country_std) AS countries,
               MIN(week_id) AS period_start,
               MAX(week_id) AS period_end
        FROM df_raw
    """).df().to_string(index=False))

    print(f"\n📊 event_type 분포")
    print(con.execute("""
        SELECT event_type, COUNT(*) AS events, SUM(fatalities) AS deaths
        FROM df_raw
        GROUP BY event_type ORDER BY events DESC
    """).df().to_string(index=False))

    df_weekly = con.execute(
        build_weekly_agg_query(lat_col, lon_col)
    ).df()
    con.close()

    print(f"\n✅ 주간 집계: {len(df_weekly):,}행")

    # ── 롤링 피처 계산 ────────────────────────────────────────────────────────
    print(f"\n🔢 롤링 피처 계산 중 (CII_roll7 / CII_roll90)...")
    df_weekly = compute_rolling_features(df_weekly)
    print(f"✅ 롤링 피처 완료")

    # ── Spillover 계산 ────────────────────────────────────────────────────────
    print(f"\n🔢 Spillover 계산 중 (인접국 raw CII 기반)...")
    print(f"   ※ 행 수에 따라 시간이 걸릴 수 있습니다.")
    df_weekly = compute_spillover(df_weekly)
    print(f"✅ Spillover 완료")

    # ── 요약 ─────────────────────────────────────────────────────────────────
    print_summary(df_weekly)

    # ── 컬럼 순서 정리 ───────────────────────────────────────────────────────
    col_order = [
        "country_std", "year", "week_num", "week_id", "group",
        "total_events", "weekly_fatalities",
        "battle_count", "vac_count", "nonstate_count",
        "battle_ratio", "civilian_targeting_ratio", "geographic_spread",
        "CII", "CII_roll7", "CII_roll90",
        "spillover",
    ]
    df_weekly = df_weekly[[c for c in col_order if c in df_weekly.columns]]

    # ── 저장 ─────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_pq  = os.path.join(output_dir, "acled_weekly.parquet")
    out_csv = os.path.join(output_dir, "acled_weekly.csv")

    pq.write_table(pa.Table.from_pandas(df_weekly), out_pq, compression="snappy")
    df_weekly.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n✅ 저장 완료")
    print(f"   {out_pq}  ({os.path.getsize(out_pq)/1024/1024:.2f} MB)")
    print(f"   {out_csv}  ({os.path.getsize(out_csv)/1024/1024:.2f} MB)")

    return out_pq


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ACLED CSV → 국가+주(week) 단위 raw 집계 Parquet"
    )
    parser.add_argument("file",  nargs="?", help="입력 CSV 경로 (grouped)")
    parser.add_argument("--out", default=None, help="출력 폴더 (기본: 입력파일 폴더)")
    args = parser.parse_args()

    input_path = args.file
    if not input_path:
        print("ACLED grouped CSV 파일 경로를 입력하세요:")
        input_path = input("  >> ").strip().strip('"').strip("'")

    if not os.path.exists(input_path):
        print(f"❌ 파일 없음: {input_path}")
        return

    output_dir = args.out or os.path.dirname(os.path.abspath(input_path))
    aggregate(input_path, output_dir)


if __name__ == "__main__":
    main()
