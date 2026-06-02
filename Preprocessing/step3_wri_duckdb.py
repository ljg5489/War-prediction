"""
step3_wri_duckdb.py
──────────────────────────────────────────────────────────────────────────────
WRI 계산 (DuckDB 온디맨드 JOIN) — 전체 데이터셋 통합 최종본

*_z.parquet 파일들을 SQL JOIN으로 합쳐 WRI를 계산합니다.

지원 데이터셋:
  ucdp          : CII / ESC 기준선 (필수)
  qog           : SFI → 민족분열지수 EF
  vdem          : SFI → Anocracy / 부패 | Polit → 선거폭력
  wb            : SFI → 군사비 | Econ → GDP / CPI
  gdelt         : Spill → 미디어 긴장도 (ACLED 없을 때 fallback)
  acled_weekly  : CII / ESC / Spill 우선 소스 (step1b 출력)
  reign         : Polit → PolitShock / irregular_change_flag (step1c)
  unhcr         : Spill → 난민 유출 압력 (step1c)
  fao           : Econ → 식량가격 충격 FoodShock (step1c)
  emdat         : 보조 컬럼 (재해 플래그, WRI 가중치 외)

서브인덱스 수식 (v3.1 최종):
  CII   = ACLED Z_CII_90d  OR  0.50×ucdp_deaths + 0.30×ucdp_events + ...
  ESC   = ACLED (Z7d-Z90d)/(|Z90d|+ε)  OR  ucdp_esc_z
  SFI   = 0.25×EF + 0.30×Anocracy + 0.25×Corr + 0.20×MilExp
  Econ  = -0.40×GDP + 0.25×CPI + 0.25×FoodShock + 0.10×FoodPrice
  Spill = (ACLED 50% OR 0) + (GDELT 25%) + (UNHCR refugee 25%)
  Polit = 0.40×ElecViol + 0.40×PolitShock + 0.20×IrregChange

WRI = 0.35×CII + 0.20×ESC + 0.15×SFI + 0.10×Econ + 0.12×Spill + 0.08×Polit

사용법:
    python step3_wri_duckdb.py <입력폴더> <출력폴더>
    python step3_wri_duckdb.py   (인자 없으면 경로 입력 프롬프트)

입력:  *_z.parquet 파일들이 있는 폴더 (step2 출력)
출력:  wri.parquet  +  wri.csv
──────────────────────────────────────────────────────────────────────────────
"""

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import glob
import os
import argparse

# ── WRI 가중치 ────────────────────────────────────────────────────────────────
W = {"CII": 0.35, "ESC": 0.20, "SFI": 0.15, "Econ": 0.10, "Spill": 0.12, "Polit": 0.08}

# ── 분석 연도 범위 ────────────────────────────────────────────────────────────
YEAR_MIN = 2013
YEAR_MAX = 2024

# ── 파일명 키워드 → 내부 키 매핑 ─────────────────────────────────────────────
KEYWORDS = {
    "ucdp":         "ucdp",
    "qog":          "qog",
    "vdem":         "vdem",
    "v-dem":        "vdem",
    "worldbank":    "wb",
    "world_bank":   "wb",
    "wb":           "wb",
    "gdelt":        "gdelt",
    "acled_weekly": "acled",   # step1b 출력 (주간 집계)
    "acled":        "acled",   # step1 출력 (연간 집계) ← 추가
    "reign":        "reign",   # step1c 출력
    "unhcr":        "unhcr",   # step1c 출력
    "fao":          "fao",     # step1c 출력
    "emdat":        "emdat",   # step1c 출력
}

# ── UN 지역 그룹 (regcluster 계산용) ─────────────────────────────────────────
UN_REGIONS = {
    "Syria": "MENA", "Yemen": "MENA", "Iraq": "MENA", "Lebanon": "MENA",
    "Somalia": "SSA", "Ethiopia": "SSA", "South Sudan": "SSA", "Mali": "SSA",
    "DR Congo": "SSA", "Nigeria": "SSA", "Sudan": "SSA",
    "Central African Republic": "SSA", "Botswana": "SSA",
    "Pakistan": "SA", "Afghanistan": "SA",
    "Myanmar": "SEA",
    "Japan": "EA", "South Korea": "EA", "Taiwan": "EA", "Mongolia": "EA",
    "Ukraine": "Europe", "Norway": "Europe", "Switzerland": "Europe",
    "Portugal": "Europe", "Germany": "Europe",
    "Venezuela": "Americas", "Colombia": "Americas", "Ecuador": "Americas",
    "Uruguay": "Americas", "Canada": "Americas", "Haiti": "Americas",
}


# ─────────────────────────────────────────────────────────────────────────────
# 파일 탐색
# ─────────────────────────────────────────────────────────────────────────────

def find_parquet_files(input_dir: str) -> dict:
    """폴더에서 *_z.parquet 파일을 찾아 키워드로 매핑."""
    all_pq  = sorted(glob.glob(os.path.join(input_dir, "*_z.parquet")))
    keys    = ["ucdp", "qog", "vdem", "wb", "gdelt",
               "acled", "reign", "unhcr", "fao", "emdat"]
    parquet = {k: None for k in keys}

    for fpath in all_pq:
        fname_lower = os.path.basename(fpath).lower()
        for keyword, key in KEYWORDS.items():
            if keyword in fname_lower:
                parquet[key] = fpath
                break

    print("📋 파일 매핑 결과")
    for key, fpath in parquet.items():
        if fpath:
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"   ✅ {key:<10}: {os.path.basename(fpath)}  ({size_mb:.2f} MB)")
        else:
            status = "필수" if key == "ucdp" else "선택"
            print(f"   ❌ {key:<10}: 파일 없음 ({status})")

    if not parquet["ucdp"]:
        raise FileNotFoundError("ucdp_z.parquet 는 필수입니다.")

    return parquet


# ─────────────────────────────────────────────────────────────────────────────
# 컬럼 유틸
# ─────────────────────────────────────────────────────────────────────────────

def get_cols(key: str, parquet: dict, con) -> list:
    fpath = parquet.get(key)
    if not fpath:
        return []
    return con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{fpath}')"
    ).df()["column_name"].tolist()


def col_expr(key, colname, alias, parquet, con, fallback="0.0") -> str:
    if colname in get_cols(key, parquet, con):
        return f"{alias}.{colname}"
    print(f"   ⚠️  {key}.{colname} 없음 → {fallback} 대체")
    return fallback


def make_join(key: str, alias: str, parquet: dict, on_year: bool = True) -> str:
    fpath = parquet.get(key)
    if not fpath:
        return f"-- {key}: 파일 없음, 스킵"

    # GDELT 전용: SQLDATE(YYYYMMDD) → year 추출 + country 컬럼 자동 감지
    if key == "gdelt":
        return f"LEFT JOIN gdelt_annual {alias} ON u.country_std = {alias}.country_std AND u.year = {alias}.year"

    # 시간 고정값(예: QOG al_ethnic2000) → 연도 무관, 국가 기준으로만 JOIN.
    # DISTINCT 로 국가당 1행 보장(연도 fan-out 방지).
    if not on_year:
        return (
            f"LEFT JOIN (SELECT DISTINCT * FROM read_parquet('{fpath}')) {alias} "
            f"ON u.country_std = {alias}.country_std"
        )

    return (
        f"LEFT JOIN read_parquet('{fpath}') {alias} "
        f"ON u.country_std = {alias}.country_std AND u.year = {alias}.year"
    )


def make_gdelt_cte(parquet: dict, con) -> str:
    """
    GDELT 전용 CTE: SQLDATE(YYYYMMDD) → year 추출 + 국가+연도 집계
    country_std 없으면 ActionGeo_CountryCode 사용
    """
    fpath = parquet.get("gdelt")
    if not fpath:
        return ""

    gdelt_cols = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{fpath}')"
    ).df()["column_name"].tolist()

    # country 컬럼 선택
    if "country_std" in gdelt_cols:
        ctry_src = "country_std"
    elif "ActionGeo_CountryCode" in gdelt_cols:
        ctry_src = "ActionGeo_CountryCode"
        print("   ⚠️  GDELT: country_std 없음 → ActionGeo_CountryCode 사용")
    else:
        print("   ⚠️  GDELT: 국가 컬럼 없음 → GDELT 스킵")
        return ""

    # year 컬럼 선택
    if "year" in gdelt_cols:
        year_src = "year"
    elif "SQLDATE" in gdelt_cols:
        year_src = "CAST(SUBSTR(CAST(SQLDATE AS VARCHAR), 1, 4) AS INTEGER)"
    else:
        print("   ⚠️  GDELT: 날짜 컬럼 없음 → GDELT 스킵")
        return ""

    # _z 컬럼 우선, 없으면 원본
    tone_src = "AvgTone_z"      if "AvgTone_z"      in gdelt_cols else "AvgTone"
    gold_src = "AvgGoldstein_z" if "AvgGoldstein_z" in gdelt_cols else "AvgGoldstein"

    return f"""
gdelt_annual AS (
    -- SQLDATE(YYYYMMDD) → year 추출, 국가+연도 평균 집계
    SELECT
        country_std,
        year,
        AVG({tone_src}) AS AvgTone_z,
        AVG({gold_src}) AS AvgGoldstein_z
    FROM (
        SELECT
            {ctry_src}   AS country_std,
            {year_src}   AS year,
            {tone_src},
            {gold_src}
        FROM read_parquet('{fpath}')
    ) t
    GROUP BY country_std, year
),
"""


# ─────────────────────────────────────────────────────────────────────────────
# regcluster 사전 계산 (ACLED Z_CII_90d 기반)
# ─────────────────────────────────────────────────────────────────────────────

def compute_regcluster(acled_path: str | None, con) -> str | None:
    if not acled_path:
        return None

    acled_cols = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{acled_path}')"
    ).df()["column_name"].tolist()

    if "CII_roll90_z" not in acled_cols:
        print("   ⚠️  CII_roll90_z 없음 → regcluster 스킵")
        return None

    print("   regcluster 사전 계산 중...")

    df_z = con.execute(f"""
        SELECT country_std, year, AVG(CII_roll90_z) AS Z_CII_90d
        FROM read_parquet('{acled_path}')
        GROUP BY country_std, year
    """).df()

    df_z["region"] = df_z["country_std"].map(UN_REGIONS)

    rows = []
    for (country, year), _ in df_z.set_index(["country_std", "year"]).iterrows():
        region = UN_REGIONS.get(country)
        if not region:
            rows.append({"country_std": country, "year": year, "regcluster": 0.0})
            continue
        region_df   = df_z[(df_z["region"] == region) & (df_z["country_std"] != country)]
        year_subset = region_df[region_df["year"] == year]
        if len(year_subset) == 0:
            rows.append({"country_std": country, "year": year, "regcluster": 0.0})
        else:
            high = (year_subset["Z_CII_90d"] > 1.5).sum()
            rows.append({"country_std": country, "year": year,
                         "regcluster": round(high / len(year_subset), 6)})

    df_reg = pd.DataFrame(rows)
    con.register("regcluster_tbl", df_reg)
    print(f"   ✅ regcluster 완료: {len(df_reg):,}행")
    return "regcluster_tbl"


# ─────────────────────────────────────────────────────────────────────────────
# WRI 쿼리 빌드
# ─────────────────────────────────────────────────────────────────────────────

def build_wri_query(parquet: dict, con, regcluster_tbl: str | None) -> str:
    ucdp_path  = parquet["ucdp"]
    acled_path = parquet.get("acled")

    # ── ACLED 컬럼 가용성 ─────────────────────────────────────────────────
    acled_cols     = get_cols("acled", parquet, con)
    has_roll7      = "CII_roll7_z"  in acled_cols
    has_roll90     = "CII_roll90_z" in acled_cols
    has_spillover  = "spillover_z"  in acled_cols

    # ────────────────────────────────────────────────────────────────────────
    # CII (ACLED Z_CII_90d 우선 → UCDP fallback)
    # ────────────────────────────────────────────────────────────────────────
    cii_ucdp = (
        f"0.50*COALESCE({col_expr('ucdp','total_deaths_z',   'u',parquet,con)},0.0)"
        f"+0.30*COALESCE({col_expr('ucdp','event_count_z',    'u',parquet,con)},0.0)"
        f"+0.10*COALESCE({col_expr('ucdp','deaths_state_z',   'u',parquet,con)},0.0)"
        f"+0.10*COALESCE({col_expr('ucdp','deaths_onesided_z','u',parquet,con)},0.0)"
    )
    cii_expr = (
        f"COALESCE(a.Z_CII_90d, {cii_ucdp})"
        if has_roll90 else cii_ucdp
    )
    if not has_roll90:
        print("   ⚠️  ACLED CII_roll90_z 없음 → CII UCDP 사용")

    # ────────────────────────────────────────────────────────────────────────
    # ESC (ACLED 우선 → UCDP fallback)
    # ────────────────────────────────────────────────────────────────────────
    if has_roll7 and has_roll90:
        esc_expr = (
            "COALESCE("
            "  (a.Z_CII_7d - a.Z_CII_90d) / (ABS(a.Z_CII_90d) + 1e-8),"
            "  u.esc_z"
            ")"
        )
    else:
        esc_expr = "COALESCE(u.esc_z, 0.0)"
        print("   ⚠️  ACLED ESC 없음 → UCDP fallback")

    # ────────────────────────────────────────────────────────────────────────
    # SFI = 0.25×EF + 0.30×Anocracy + 0.25×Corr + 0.20×MilExp
    # ────────────────────────────────────────────────────────────────────────
    sfi_ef   = col_expr("qog",  "al_ethnic2000_z",     "q", parquet, con)
    sfi_corr = col_expr("vdem", "v2x_corr_z",            "v", parquet, con)
    sfi_mil  = col_expr("wb",   "Military_Expenditure_z", "w", parquet, con)
    anoc     = (
        "4.0 * v.v2x_libdem * (1.0 - v.v2x_libdem)"
        if "v2x_libdem" in get_cols("vdem", parquet, con) else "0.0"
    )
    sfi_expr = (
        f"0.25*COALESCE({sfi_ef},   0.0)"
        f"+0.30*COALESCE({anoc},    0.0)"
        f"+0.25*COALESCE({sfi_corr},0.0)"
        f"+0.20*COALESCE({sfi_mil}, 0.0)"
    )

    # ────────────────────────────────────────────────────────────────────────
    # Econ = -0.40×GDP + 0.25×CPI + 0.25×FoodShock + 0.10×FoodPrice
    #   FAO 없으면 → -0.60×GDP + 0.40×CPI (기존 비율 유지)
    # ────────────────────────────────────────────────────────────────────────
    econ_gdp   = col_expr("wb",  "GDP_Growth_z", "w", parquet, con)
    econ_cpi   = col_expr("wb",  "Inflation_z",  "w", parquet, con)
    fao_shock  = col_expr("fao", "FoodShock_z",          "f", parquet, con)
    fao_price  = col_expr("fao", "food_price_index_z",   "f", parquet, con)
    has_fao    = parquet.get("fao") is not None

    if has_fao:
        econ_expr = (
            f"-0.40*COALESCE({econ_gdp},  0.0)"
            f"+0.25*COALESCE({econ_cpi},  0.0)"
            f"+0.25*COALESCE({fao_shock}, 0.0)"
            f"+0.10*COALESCE({fao_price}, 0.0)"
        )
        print("   Econ: GDP(-0.40) + CPI(0.25) + FoodShock(0.25) + FoodPrice(0.10)")
    else:
        econ_expr = (
            f"-0.60*COALESCE({econ_gdp}, 0.0)"
            f"+0.40*COALESCE({econ_cpi}, 0.0)"
        )
        print("   Econ: FAO 없음 → GDP(-0.60) + CPI(0.40)")

    # ── GDELT CTE 생성 (SQLDATE→year 추출 + 국가+연도 집계) ──────────────────
    gdelt_cte_str = make_gdelt_cte(parquet, con)
    if not gdelt_cte_str.strip():
        # GDELT 없거나 필수 컬럼 없음 → 더미 CTE
        gdelt_cte_str = """
gdelt_annual AS (
    SELECT NULL::VARCHAR AS country_std, NULL::INTEGER AS year,
           NULL::DOUBLE AS AvgTone_z, NULL::DOUBLE AS AvgGoldstein_z
    WHERE FALSE
),
"""

    # ────────────────────────────────────────────────────────────────────────
    # Spill = ACLED(50%) + GDELT(25%) + UNHCR refugee(25%)
    #   가용 소스에 따라 비율 자동 조정
    # ────────────────────────────────────────────────────────────────────────
    # GDELT는 CTE(gdelt_annual)를 통해 접근 → col_expr 대신 직접 참조
    if parquet.get("gdelt"):
        gdelt_raw = (
            "0.50*COALESCE(g.AvgTone_z, 0.0)"
            "+0.50*COALESCE(g.AvgGoldstein_z, 0.0)"
        )
    else:
        gdelt_raw = "0.0"
    unhcr_ref  = col_expr("unhcr", "refugee_outflow_z", "h", parquet, con)
    has_unhcr  = parquet.get("unhcr") is not None

    if has_spillover and has_unhcr:
        spill_expr = (
            f"0.50*COALESCE(a.spillover_z, 0.0)"
            f"+0.25*({gdelt_raw})"
            f"+0.25*COALESCE({unhcr_ref}, 0.0)"
        )
        print("   Spill: ACLED(50%) + GDELT(25%) + UNHCR(25%)")
    elif has_spillover:
        spill_expr = (
            f"0.60*COALESCE(a.spillover_z, 0.0)"
            f"+0.40*({gdelt_raw})"
        )
        print("   Spill: ACLED(60%) + GDELT(40%)")
    elif has_unhcr:
        spill_expr = (
            f"0.50*({gdelt_raw})"
            f"+0.50*COALESCE({unhcr_ref}, 0.0)"
        )
        print("   Spill: GDELT(50%) + UNHCR(50%)")
    else:
        spill_expr = gdelt_raw
        print("   Spill: GDELT(100%)")

    # ────────────────────────────────────────────────────────────────────────
    # Polit = 0.40×ElecViol + 0.40×PolitShock + 0.20×IrregChange
    #   REIGN 없으면 → ElecViol 100%
    # ────────────────────────────────────────────────────────────────────────
    elec_viol  = col_expr("vdem",  "v2elintim_z",             "v", parquet, con)
    polit_sh   = col_expr("reign", "PolitShock_z",             "r", parquet, con)
    irreg_flag = col_expr("reign", "irregular_change_flag_z",  "r", parquet, con)
    has_reign  = parquet.get("reign") is not None

    if has_reign:
        polit_expr = (
            f"0.40*COALESCE({elec_viol},  0.0)"
            f"+0.40*COALESCE({polit_sh},  0.0)"
            f"+0.20*COALESCE({irreg_flag},0.0)"
        )
        print("   Polit: ElecViol(0.40) + PolitShock(0.40) + IrregChange(0.20)")
    else:
        polit_expr = f"COALESCE({elec_viol}, 0.0)"
        print("   Polit: REIGN 없음 → ElecViol(100%)")

    # ── regcluster ───────────────────────────────────────────────────────────
    if regcluster_tbl:
        regcluster_expr = "COALESCE(rc.regcluster, 0.0)"
        regcluster_join = (
            f"LEFT JOIN {regcluster_tbl} rc "
            f"ON u.country_std = rc.country_std AND u.year = rc.year"
        )
    else:
        regcluster_expr = "0.0"
        regcluster_join = "-- regcluster 없음"

    # ── EM-DAT 보조 컬럼 ─────────────────────────────────────────────────────
    emdat_flag   = col_expr("emdat", "disaster_flag_z",   "e", parquet, con)
    emdat_deaths = col_expr("emdat", "disaster_deaths_z", "e", parquet, con)

    # ── UNHCR IDP 보조 컬럼 ──────────────────────────────────────────────────
    unhcr_idp  = col_expr("unhcr", "idp_count_z", "h", parquet, con)

    # ── ACLED 연간 집계 CTE ───────────────────────────────────────────────────
    if acled_path and (has_roll7 or has_roll90 or has_spillover):
        r7  = "AVG(CII_roll7_z)"  if has_roll7     else "NULL"
        r90 = "AVG(CII_roll90_z)" if has_roll90    else "NULL"
        sp  = "AVG(spillover_z)"  if has_spillover else "NULL"
        acled_cte = f"""
acled_annual AS (
    SELECT country_std, year,
           {r7}  AS Z_CII_7d,
           {r90} AS Z_CII_90d,
           {sp}  AS spillover_z
    FROM read_parquet('{acled_path}')
    GROUP BY country_std, year
),
"""
        acled_join = (
            "LEFT JOIN acled_annual a "
            "ON u.country_std = a.country_std AND u.year = a.year"
        )
    else:
        acled_cte  = """
acled_annual AS (
    SELECT NULL::VARCHAR AS country_std, NULL::INTEGER AS year,
           NULL::DOUBLE AS Z_CII_7d, NULL::DOUBLE AS Z_CII_90d,
           NULL::DOUBLE AS spillover_z
    WHERE FALSE
),
"""
        acled_join = "-- ACLED 없음"
        print("   ⚠️  ACLED 없음 → CII/ESC/Spill UCDP/GDELT 사용")

    # ── 전체 JOIN 목록 ────────────────────────────────────────────────────────
    joins = "\n    ".join([
        make_join("qog",   "q", parquet, on_year=False),
        make_join("vdem",  "v", parquet),
        make_join("wb",    "w", parquet),
        make_join("gdelt", "g", parquet),
        make_join("reign", "r", parquet),
        make_join("unhcr", "h", parquet),
        make_join("fao",   "f", parquet),
        make_join("emdat", "e", parquet),
        acled_join,
        regcluster_join,
    ])

    return f"""
WITH

-- ① UCDP ESC 변화율
ucdp_esc AS (
    SELECT *,
        COALESCE(
            (total_deaths
             - LAG(total_deaths) OVER (PARTITION BY country_std ORDER BY year))
            / NULLIF(LAG(total_deaths) OVER (PARTITION BY country_std ORDER BY year), 0),
        0.0) AS deaths_pct_change
    FROM read_parquet('{ucdp_path}')
),

-- ② UCDP ESC Z-score (ACLED ESC fallback용)
ucdp_final AS (
    SELECT *,
        ROUND(
            (deaths_pct_change - AVG(deaths_pct_change) OVER (PARTITION BY country_std))
            / (STDDEV_POP(deaths_pct_change) OVER (PARTITION BY country_std) + 1e-8),
        6) AS esc_z
    FROM ucdp_esc
),

-- ③ GDELT: SQLDATE → year 추출 + 국가+연도 집계
{gdelt_cte_str}
{acled_cte}
-- ③ JOIN + 서브인덱스 계산
sub_index AS (
    SELECT
        u.country_std,
        u.year,
        u."group",

        ROUND({cii_expr}, 6)   AS CII,
        ROUND({esc_expr}, 6)   AS ESC,
        ROUND({sfi_expr}, 6)   AS SFI,
        ROUND({econ_expr}, 6)  AS Econ,
        ROUND({spill_expr}, 6) AS Spill,
        ROUND({polit_expr}, 6) AS Polit,

        -- 보조 컬럼 (분석·모델 피처용)
        ROUND(COALESCE(a.Z_CII_7d,   0.0), 6) AS Z_CII_7d,
        ROUND(COALESCE(a.Z_CII_90d,  0.0), 6) AS Z_CII_90d,
        ROUND({regcluster_expr},           6) AS regcluster,
        ROUND(COALESCE({unhcr_idp},  0.0), 6) AS idp_count_z,
        ROUND(COALESCE({emdat_flag}, 0.0), 6) AS disaster_flag_z,
        ROUND(COALESCE({emdat_deaths},0.0),6) AS disaster_deaths_z

    FROM ucdp_final u
    {joins}
),

-- ④ WRI 최종 합산
wri_final AS (
    SELECT *,
        ROUND(
            {W['CII']}   * CII
          + {W['ESC']}   * ESC
          + {W['SFI']}   * SFI
          + {W['Econ']}  * Econ
          + {W['Spill']} * Spill
          + {W['Polit']} * Polit,
        6) AS WRI
    FROM sub_index
)

SELECT * FROM wri_final
WHERE year BETWEEN {YEAR_MIN} AND {YEAR_MAX}
ORDER BY country_std, year
"""


# ─────────────────────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────────────────────

def run(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    parquet = find_parquet_files(input_dir)
    con     = duckdb.connect()

    print("\n🔢 regcluster 계산 중...")
    regcluster_tbl = compute_regcluster(parquet.get("acled"), con)

    print("\n컬럼 가용성 확인 중...")
    wri_query = build_wri_query(parquet, con, regcluster_tbl)

    print("\n🔢 WRI 계산 중...")
    df_wri = con.execute(wri_query).df()
    print(f"✅ 계산 완료: {len(df_wri):,}행 × {len(df_wri.columns)}컬럼")

    print("\n📊 그룹별 WRI 통계")
    print(df_wri.groupby("group")["WRI"].describe().round(4).to_string())

    print("\n📊 서브인덱스 요약")
    sub_cols = ["CII","ESC","SFI","Econ","Spill","Polit","WRI"]
    print(df_wri[[c for c in sub_cols if c in df_wri.columns]].describe().round(4).to_string())

    print("\n🔥 WRI 최고 위험 상위 20개")
    print(
        df_wri.nlargest(20, "WRI")[
            ["country_std","year","group","CII","ESC","SFI","Econ","Spill","Polit","WRI"]
        ].reset_index(drop=True).to_string(index=False)
    )

    out_pq  = os.path.join(output_dir, "wri.parquet")
    out_csv = os.path.join(output_dir, "wri.csv")

    pq.write_table(pa.Table.from_pandas(df_wri), out_pq, compression="snappy")
    df_wri.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n✅ 저장 완료")
    print(f"   {out_pq}  ({os.path.getsize(out_pq)/1024/1024:.2f} MB)")
    print(f"   {out_csv}  ({os.path.getsize(out_csv)/1024/1024:.2f} MB)")
    print(f"\n📌 다음 단계: python step4_label_target.py {out_csv}")

    con.close()


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="*_z.parquet → WRI 계산 (전체 데이터셋 통합)"
    )
    parser.add_argument("input_dir",  nargs="?")
    parser.add_argument("output_dir", nargs="?")
    args = parser.parse_args()

    input_dir  = args.input_dir
    output_dir = args.output_dir

    if not input_dir:
        print("입력 폴더 (*_z.parquet 파일들):")
        input_dir = input("  >> ").strip().strip('"').strip("'")

    if not output_dir:
        default_out = os.path.join(os.path.dirname(input_dir), "03_wri")
        print(f"출력 폴더 (엔터={default_out}):")
        output_dir = input("  >> ").strip().strip('"').strip("'") or default_out

    if not os.path.isdir(input_dir):
        print(f"❌ 폴더 없음: {input_dir}")
        return

    run(input_dir, output_dir)


if __name__ == "__main__":
    main()