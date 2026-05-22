"""
step1c_auxiliary_datasets.py
──────────────────────────────────────────────────────────────────────────────
보조 데이터셋 처리: REIGN / UNHCR / FAO FPMA / EM-DAT

각 파일을 자동 감지해 국가+연도 단위로 집계하고 Parquet으로 저장합니다.

출력 피처:
  REIGN  → reign.parquet
      PolitShock            : 정치충격 복합지수 (쿠데타/선거분쟁/지도자교체)
      irregular_change_flag : 비정상 권력교체 여부 (0/1)
      tenure_mean           : 연평균 집권 기간(월)

  UNHCR  → unhcr.parquet
      refugee_outflow       : 해당 국가 기원 난민 수 (타국 유출)
      idp_count             : 국내 실향민(IDP) 수
      asylum_seekers        : 망명 신청자 수

  FAO    → fao.parquet
      food_price_index      : 월별 식량가격종합지수(PFCS) 연평균
      FoodShock             : 분기 대비 식량가격 급등 (max(0, ΔQ/Q_prev))

  EM-DAT → emdat.parquet
      disaster_flag         : 해당 연도 자연재해 발생 여부 (0/1)
      disaster_deaths       : 연간 자연재해 사망자 합계
      disaster_count        : 연간 재해 건수

피처 → WRI 연결:
  REIGN  → 5.5 Polit (PolitShock, irregular_change_flag)
  UNHCR  → 5.4 Spill (refugee_outflow), 5.7 Humanitarian
  FAO    → 5.3 Econ  (FoodShock, food_price_index)
  EM-DAT → 5.7 Humanitarian (step3 보조 컬럼, WRI 가중치 외)

파이프라인 위치:
    normalize_and_group.py → step1c → step2 → step3

데이터셋 자동 감지 (컬럼 기준):
    REIGN  : 'irregular' + 'tenure_months' 컬럼
    UNHCR  : 'refugees' 또는 'coo_name' 컬럼
    FAO    : 'pfcs' / 'food_price' / 'price_index' 컬럼
    EM-DAT : 'disaster_type' + 'total_deaths' 컬럼

사용법:
    python step1c_auxiliary_datasets.py <CSV1> [<CSV2> ...]
    python step1c_auxiliary_datasets.py --out <출력폴더> <CSV1> ...
    python step1c_auxiliary_datasets.py   (인자 없으면 대화형 입력)

입력:  normalize_and_group.py 출력 CSV (country_std, group 포함)
       단, FAO가 전국 집계(global)인 경우 country_std 없어도 처리
출력:  01_parquet/{reign,unhcr,fao,emdat}.parquet
──────────────────────────────────────────────────────────────────────────────
"""

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import argparse
import os

EPS = 1e-8

# ── 재해 유형 필터 (자연재해만) ───────────────────────────────────────────────
NATURAL_DISASTER_TYPES = [
    "Earthquake", "Flood", "Drought", "Storm", "Wildfire",
    "Volcanic activity", "Mass movement (dry)", "Mass movement (wet)",
    "Extreme temperature", "Fog", "Glacial lake outburst flood",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. 데이터셋 자동 감지
# ─────────────────────────────────────────────────────────────────────────────

def detect_dataset(cols: list) -> str:
    """컬럼 목록으로 데이터셋 종류 자동 감지."""
    lower = {c.lower() for c in cols}

    if "irregular" in lower and "tenure_months" in lower:
        return "REIGN"
    # UNHCR: 난민 수 컬럼이 있거나, origin/destination 구조 + date 패턴
    if ("refugees" in lower or "coo_name" in lower or "coo" in lower
            or ("idps" in lower and "date" in lower)
            or ("asylum_seekers" in lower and "date" in lower)):
        return "UNHCR"
    if any(k in lower for k in ["pfcs", "food_price", "price_index", "fpcs"]):
        return "FAO"
    # FAO FPMA 다운로드 형식: Year + Months(월이름) + Area + Value
    if "months" in lower and "value" in lower and "area" in lower:
        return "FAO"
    if "disaster_type" in lower or "disastertype" in lower:
        return "EMDAT"
    return "UNKNOWN"


def col(df: pd.DataFrame, *candidates) -> str | None:
    """대소문자 무시 컬럼명 검색. 첫 번째 매칭 반환."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. REIGN 처리
# ─────────────────────────────────────────────────────────────────────────────

def process_reign(df: pd.DataFrame) -> pd.DataFrame:
    """
    REIGN 월별 → 국가+연도 집계

    PolitShock 시간 윈도우:
      쿠데타(irregular=1)    : ±3개월 → 1.0
      선거(election=1)       : ±1개월 → 0.7
      지도자 교체(tenure=1)  : ±1개월 → 0.3
    연간 집계: max(PolitShock) per year
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # ── 컬럼 매핑 ─────────────────────────────────────────────────────────
    year_col   = col(df, "year")
    month_col  = col(df, "month")
    ctry_col   = col(df, "country_std", "country", "cname")
    irreg_col  = col(df, "irregular")
    elect_col  = col(df, "election", "elect")
    tenure_col = col(df, "tenure_months", "tenuremont", "tenure")

    if not all([year_col, ctry_col, irreg_col]):
        raise ValueError("REIGN 필수 컬럼(year, country, irregular) 없음")

    df = df.rename(columns={ctry_col: "country_std"})
    if month_col:
        df["month"] = df[month_col].fillna(1).astype(int)
    else:
        print("  ⚠️  month 컬럼 없음 → month=1 가정")
        df["month"] = 1

    df["year"]      = df[year_col].astype(int)
    df["irregular"] = pd.to_numeric(df[irreg_col], errors="coerce").fillna(0).astype(int)

    if elect_col:
        df["election"] = pd.to_numeric(df[elect_col], errors="coerce").fillna(0).astype(int)
    else:
        df["election"] = 0

    if tenure_col:
        df["tenure_months"] = pd.to_numeric(df[tenure_col], errors="coerce").fillna(np.nan)
    else:
        df["tenure_months"] = np.nan

    # ── 월별 PolitShock 계산 ─────────────────────────────────────────────
    results = []

    for country, grp in df.groupby("country_std", sort=False):
        g = grp.sort_values(["year", "month"]).copy().reset_index(drop=True)
        g["month_index"] = g["year"] * 12 + g["month"]
        g["PolitShock"]  = 0.0

        # 지도자 교체 감지 (tenure_months == 1 또는 첫 달)
        if tenure_col and "tenure_months" in g.columns:
            g["leader_change"] = (g["tenure_months"] == 1).astype(int)
        else:
            g["leader_change"] = 0

        coup_months    = g.loc[g["irregular"] == 1, "month_index"].tolist()
        elect_months   = g.loc[g["election"]  == 1, "month_index"].tolist()
        change_months  = g.loc[g["leader_change"] == 1, "month_index"].tolist()

        # 윈도우 적용 (우선순위: 쿠데타 > 선거 > 교체)
        for idx, row in g.iterrows():
            mi = row["month_index"]
            shock = 0.0
            if any(abs(mi - cm) <= 3 for cm in coup_months):
                shock = 1.0
            elif any(abs(mi - em) <= 1 for em in elect_months):
                shock = 0.7
            elif any(abs(mi - lm) <= 1 for lm in change_months):
                shock = 0.3
            g.at[idx, "PolitShock"] = shock

        results.append(g)

    df_monthly = pd.concat(results, ignore_index=True)

    # ── 연간 집계 ────────────────────────────────────────────────────────
    agg = (
        df_monthly
        .groupby(["country_std", "year"])
        .agg(
            PolitShock            = ("PolitShock",    "max"),
            irregular_change_flag = ("irregular",     "max"),
            tenure_mean           = ("tenure_months", "mean"),
        )
        .reset_index()
    )

    if "group" in df.columns:
        grp_map = df.drop_duplicates("country_std").set_index("country_std")["group"]
        agg["group"] = agg["country_std"].map(grp_map).fillna("OTHER")
    else:
        agg["group"] = "OTHER"

    print(f"  ✅ REIGN: {len(agg):,}행  "
          f"국가 {agg['country_std'].nunique()}개  "
          f"연도 {int(agg['year'].min())}~{int(agg['year'].max())}")
    print(f"     PolitShock > 0 비율: {(agg['PolitShock'] > 0).mean():.1%}")
    print(f"     irregular 발생 연도: {agg['irregular_change_flag'].sum():,}건")

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 3. UNHCR 처리
# ─────────────────────────────────────────────────────────────────────────────

def _parse_unhcr_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    UNHCR 날짜 컬럼 파싱 → year, month, date_parsed 생성.

    지원 형식:
      date 컬럼  : YYYY-MM-DD / DD/MM/YYYY / YYYY/MM/DD / YYYY-MM 등
      year 컬럼  : 연도만 있는 경우 (fallback)
    """
    date_col = col(df, "date", "date_start", "start_date",
                   "period", "year_month", "reference_date")
    year_col = col(df, "year")

    if date_col:
        raw = df[date_col].astype(str).str.strip()

        # YYYY-MM 형식 (일자 없음) → 01일 보완
        mask_ym = raw.str.match(r"^\d{4}-\d{2}$")
        raw = raw.where(~mask_ym, raw + "-01")

        parsed = pd.to_datetime(raw, errors="coerce", dayfirst=False)

        # dayfirst 재시도 (DD/MM/YYYY 등)
        still_null = parsed.isna()
        if still_null.any():
            parsed[still_null] = pd.to_datetime(
                raw[still_null], errors="coerce", dayfirst=True
            )

        df = df.copy()
        df["date_parsed"] = parsed
        df["year"]        = parsed.dt.year.astype("Int64")
        df["month"]       = parsed.dt.month.astype("Int64")

        null_cnt = df["year"].isna().sum()
        if null_cnt:
            print(f"  ⚠️  날짜 파싱 실패 {null_cnt:,}행 → 제외")
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)

        print(f"  ℹ️  날짜 컬럼 '{date_col}' 파싱 완료 "
              f"({df['year'].min()}~{df['year'].max()})")

    elif year_col:
        df = df.copy()
        df["year"]        = df[year_col].astype(int)
        df["month"]       = 12          # 연간 데이터 → 12월로 가정
        df["date_parsed"] = pd.to_datetime(
            df["year"].astype(str) + "-12-31", errors="coerce"
        )
        print(f"  ℹ️  year 컬럼 사용 (일별 데이터 아님)")

    else:
        raise ValueError(
            "UNHCR: 날짜 컬럼 없음.\n"
            "  필요 컬럼: date / date_start / start_date / year_month / year"
        )

    return df


def _annual_stock_agg(df: pd.DataFrame, ctry_col: str,
                      val_col: str, out_col: str) -> pd.DataFrame:
    """
    일별 재고(stock) 데이터 → 연간 집계.

    전략:
      1) 연말 스냅샷 우선 (12월 행)
         → 없으면 해당 연도 마지막 날 행

    합산(sum) 대신 스냅샷을 쓰는 이유:
      UNHCR 난민·IDP 수치는 특정 시점의 재고(stock)이므로
      일별 합산하면 실제값의 수백 배로 과장됩니다.

    ※ rename-then-select 대신 새 DataFrame 직접 구성
      → 컬럼 중복으로 인한 'not 1-dimensional' 에러 원천 차단
    """
    # ── 필요 컬럼만 꺼내 새 DataFrame 구성 (중복 컬럼 문제 차단) ──────────
    try:
        grp = pd.DataFrame({
            "country_std": df[ctry_col].values,
            "year":        df["year"].values,
            "month":       df["month"].values,
            "date_parsed": df["date_parsed"].values,
            "value":       pd.to_numeric(df[val_col], errors="coerce").values,
        })
    except KeyError as e:
        print(f"  ⚠️  _annual_stock_agg: 컬럼 없음 {e} → 스킵")
        return pd.DataFrame(columns=["country_std", "year", out_col])

    grp = grp.dropna(subset=["value"])
    grp["value"] = grp["value"].fillna(0)

    if grp.empty:
        return pd.DataFrame(columns=["country_std", "year", out_col])

    results = []
    for (country, year), sub in grp.groupby(["country_std", "year"], sort=False):
        # 우선순위 1: 12월 데이터
        dec = sub[sub["month"] == 12]
        if not dec.empty:
            val = dec.sort_values("date_parsed").iloc[-1]["value"]
        else:
            # 우선순위 2: 해당 연도 마지막 날짜 행
            val = sub.sort_values("date_parsed").iloc[-1]["value"]
        results.append({"country_std": country, "year": int(year), out_col: val})

    if not results:
        return pd.DataFrame(columns=["country_std", "year", out_col])
    return pd.DataFrame(results)


def process_unhcr(df: pd.DataFrame) -> pd.DataFrame:
    """
    UNHCR 일별(또는 연간) 데이터 → 국가+연도 집계

    입력 형식 자동 감지:
      - date / date_start / start_date / year_month 컬럼 → 일별 데이터
      - year 컬럼만 있음 → 연간 데이터

    집계 방식:
      UNHCR 수치는 재고(stock)이므로 합산 대신
      연말 스냅샷(12월 마지막 행) 또는 연평균을 사용합니다.

    출력 피처:
      refugee_outflow : origin 기준 연말 난민 수
      idp_count       : 국내 실향민(IDP) 연말 수
      asylum_seekers  : 망명신청자 연말 수
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # ── 날짜 파싱 ────────────────────────────────────────────────────────
    df = _parse_unhcr_date(df)

    # ── 컬럼 감지 ─────────────────────────────────────────────────────────
    # country_name / country_iso 도 후보에 포함 (UNHCR API 다운로드 형식 대응)
    ctry_col   = col(df, "country_std", "coo_name", "country_of_origin",
                      "country_origin", "origin", "country",
                      "country_name", "country_iso")
    dest_col   = col(df, "coa_name", "country_of_asylum",
                      "country_asylum", "asylum_country", "destination",
                      "country_name", "country_iso")
    refug_col  = col(df, "refugees", "total_refugees", "ref", "refugee")
    idp_col    = col(df, "idps", "idp", "internally_displaced",
                     "total_idps", "id_ps")
    asylum_col = col(df, "asylum_seekers", "asylum", "asy", "asylum_seeker")

    print(f"  감지된 컬럼: country={ctry_col}, dest={dest_col}, "
          f"ref={refug_col}, idp={idp_col}, asy={asylum_col}")

    if ctry_col is None:
        raise ValueError(
            "UNHCR: 국가 컬럼을 찾을 수 없습니다.\n"
            f"  현재 컬럼: {list(df.columns)}\n"
            "  필요 컬럼 (하나라도): country_std / coo_name / country_name / "
            "country_iso / origin / country\n"
            "  ※ normalize_and_group.py 를 먼저 실행하면 country_std 가 생깁니다."
        )

    # ── refugee_outflow: origin(기원국) 기준 연말 스냅샷 ──────────────────
    if ctry_col and refug_col:
        ref_agg = _annual_stock_agg(df, ctry_col, refug_col, "refugee_outflow")
    else:
        print("  ⚠️  origin/refugees 컬럼 없음 → refugee_outflow = 0")
        ref_agg = pd.DataFrame(columns=["country_std", "year", "refugee_outflow"])

    # ── idp_count: 목적지(destination) 또는 origin 기준 연말 스냅샷 ────────
    idp_ctry = dest_col or ctry_col
    if idp_ctry and idp_col:
        idp_agg = _annual_stock_agg(df, idp_ctry, idp_col, "idp_count")
    else:
        print("  ⚠️  idp 컬럼 없음 → idp_count = 0")
        idp_agg = pd.DataFrame(columns=["country_std", "year", "idp_count"])

    # ── asylum_seekers: 목적지 기준 연말 스냅샷 ──────────────────────────
    if idp_ctry and asylum_col:
        asy_agg = _annual_stock_agg(df, idp_ctry, asylum_col, "asylum_seekers")
    else:
        asy_agg = pd.DataFrame(columns=["country_std", "year", "asylum_seekers"])

    # ── 병합 ─────────────────────────────────────────────────────────────
    all_keys = pd.concat([
        ref_agg[["country_std", "year"]],
        idp_agg[["country_std", "year"]],
        asy_agg[["country_std", "year"]],
    ]).drop_duplicates()

    agg = all_keys.merge(ref_agg, on=["country_std", "year"], how="left")
    agg = agg.merge(idp_agg,     on=["country_std", "year"], how="left")
    agg = agg.merge(asy_agg,     on=["country_std", "year"], how="left")
    agg = agg.fillna(0)

    # group 복원
    if "group" in df.columns and ctry_col:
        grp_src = df.rename(columns={ctry_col: "country_std"})
        grp_map = grp_src.drop_duplicates("country_std").set_index("country_std")["group"]
        agg["group"] = agg["country_std"].map(grp_map).fillna("OTHER")
    else:
        agg["group"] = "OTHER"

    if agg.empty:
        print(f"  ⚠️  UNHCR: 집계 결과 없음. country 컬럼({ctry_col})과 "
              f"수치 컬럼을 확인하세요.")
    else:
        print(f"  ✅ UNHCR: {len(agg):,}행  "
              f"국가 {agg['country_std'].nunique()}개  "
              f"연도 {int(agg['year'].min())}~{int(agg['year'].max())}")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 4. FAO FPMA 처리
# ─────────────────────────────────────────────────────────────────────────────

def process_fao(df: pd.DataFrame, all_countries: list | None = None) -> pd.DataFrame:
    """
    FAO FPMA → 국가+연도 집계

    food_price_index : PFCS 월평균 → 연평균
    FoodShock        : max(0, (FPI_t - FPI_{t-3}) / FPI_{t-3}) 분기 충격 → 연간 max

    지원 형식:
      - Year + Months(월이름) + Area + Value  ← FAO FPMA 다운로드 기본 형식
      - date 컬럼 (YYYY-MM-DD)
      - year + month 컬럼 (숫자)
    """
    # 월이름 → 숫자 변환 테이블
    MONTH_MAP = {
        "january":1, "february":2, "march":3, "april":4,
        "may":5, "june":6, "july":7, "august":8,
        "september":9, "october":10, "november":11, "december":12,
        "jan":1, "feb":2, "mar":3, "apr":4,
        "jun":6, "jul":7, "aug":8, "sep":9,
        "oct":10, "nov":11, "dec":12,
    }

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # ── 컬럼 감지 ────────────────────────────────────────────────────────
    date_col  = col(df, "date", "month_date", "period", "time")
    year_col  = col(df, "year")
    month_col = col(df, "month", "months")   # ← "months"(복수) 추가

    price_col = col(df, "pfcs", "fpcs", "food_price_index",
                    "price_index", "value", "food_price", "index_value")
    ctry_col  = col(df, "country_std", "country", "area", "region")

    if not price_col:
        raise ValueError("FAO: 식량가격 컬럼 없음 (pfcs/food_price/value 등)")

    # ── 연도·월 파싱 ─────────────────────────────────────────────────────
    if date_col:
        # YYYY-MM-DD 또는 YYYY-MM 형식
        raw = df[date_col].astype(str).str.strip()
        mask_ym = raw.str.match(r"^\d{4}-\d{2}$")
        raw = raw.where(~mask_ym, raw + "-01")
        dt = pd.to_datetime(raw, errors="coerce")
        df["year"]  = dt.dt.year
        df["month"] = dt.dt.month

    elif year_col and month_col:
        df["year"] = df[year_col].astype(int)

        # 월이름(January…) 또는 숫자(1…12) 모두 처리
        raw_month = df[month_col].astype(str).str.strip().str.lower()
        if raw_month.str.match(r"^\d+$").all():
            df["month"] = raw_month.astype(int)
        else:
            df["month"] = raw_month.map(MONTH_MAP)
            unmapped = df["month"].isna().sum()
            if unmapped:
                print(f"  ⚠️  월 파싱 실패 {unmapped:,}행 → 제외")
            df = df.dropna(subset=["month"])
            df["month"] = df["month"].astype(int)

    elif year_col:
        df["year"]  = df[year_col].astype(int)
        df["month"] = 6
        print(f"  ℹ️  month 컬럼 없음 → month=6 가정 (연간 데이터)")

    else:
        raise ValueError("FAO: 날짜 컬럼(date/year/month) 없음")

    df["price"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["year", "price"])
    df["year"] = df["year"].astype(int)

    print(f"  ℹ️  파싱 완료: {len(df):,}행  "
          f"연도 {df['year'].min()}~{df['year'].max()}  "
          f"월 {sorted(df['month'].unique())}")

    # ── 국가 컬럼 처리 (rename 대신 직접 할당) ───────────────────────────
    is_global = (ctry_col is None)
    if is_global:
        print(f"  ℹ️  국가 컬럼 없음 → 전 세계 공통 지수로 처리")
        df["country_std"] = "__GLOBAL__"
    else:
        df["country_std"] = df[ctry_col]

    # ── group 컬럼 복원 ──────────────────────────────────────────────────
    if "group" in df.columns:
        grp_map = (
            df.drop_duplicates("country_std")
            .set_index("country_std")["group"]
        )
    else:
        grp_map = None

    # ── 월별 정렬 후 FoodShock 계산 ─────────────────────────────────────
    results = []
    for country, grp in df.groupby("country_std", sort=False):
        g = grp.sort_values(["year", "month"]).copy().reset_index(drop=True)
        g["price_lag3"] = g["price"].shift(3)
        g["FoodShock"]  = (
            (g["price"] - g["price_lag3"])
            / (g["price_lag3"].abs() + EPS)
        ).clip(lower=0)
        results.append(g)

    df_monthly = pd.concat(results, ignore_index=True)

    # ── 연간 집계 ────────────────────────────────────────────────────────
    agg = (
        df_monthly
        .groupby(["country_std", "year"])
        .agg(
            food_price_index = ("price",     "mean"),
            FoodShock        = ("FoodShock", "max"),
        )
        .reset_index()
    )
    agg["food_price_index"] = agg["food_price_index"].round(4)
    agg["FoodShock"]        = agg["FoodShock"].round(6)

    # ── 전국 집계 브로드캐스트 ────────────────────────────────────────────
    if is_global and all_countries:
        print(f"  ℹ️  {len(all_countries)}개국으로 브로드캐스트")
        global_df = agg[agg["country_std"] == "__GLOBAL__"].drop(columns=["country_std"])
        rows = []
        for c in all_countries:
            tmp = global_df.copy()
            tmp["country_std"] = c
            rows.append(tmp)
        agg = pd.concat(rows, ignore_index=True)

    # group 복원
    if grp_map is not None:
        agg["group"] = agg["country_std"].map(grp_map).fillna("OTHER")
    else:
        agg["group"] = "OTHER"

    print(f"  ✅ FAO: {len(agg):,}행  "
          f"국가 {agg['country_std'].nunique()}개  "
          f"연도 {int(agg['year'].min())}~{int(agg['year'].max())}")
    print(f"     FoodShock > 0 비율: {(agg['FoodShock'] > 0).mean():.1%}")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 5. EM-DAT 처리
# ─────────────────────────────────────────────────────────────────────────────

def process_emdat(df: pd.DataFrame) -> pd.DataFrame:
    """
    EM-DAT 이벤트 단위 → 국가+연도 집계

    disaster_flag   : 해당 연도 자연재해 발생 여부 (0/1)
    disaster_deaths : 연간 자연재해 총 사망자
    disaster_count  : 연간 재해 건수

    날짜 컬럼 자동 감지 (우선순위):
      1) date / start_date / event_date 등 날짜 컬럼 → 파싱 후 year 추출
      2) year / start_year 정수 컬럼 → 그대로 사용
      3) dis_no 컬럼 → 앞 4자리를 연도로 사용
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # ── 날짜/연도 컬럼 감지 ──────────────────────────────────────────────
    date_col = col(df, "date", "start_date", "event_date",
                   "date_start", "occurrence_date", "dis_date")
    year_col = col(df, "year", "start_year")
    ctry_col    = col(df, "country_std", "country", "iso",
                      "country_name", "country_iso")
    dtype_col   = col(df, "disaster_type", "disastertype", "type",
                      "disaster_subtype", "disaster subtype")
    deaths_col  = col(df, "total_deaths", "totaldeaths", "deaths",
                      "no_deaths", "death")
    affected_col = col(df, "total_affected", "totalaffected", "affected")

    # ── year 컬럼 생성 ───────────────────────────────────────────────────
    if date_col:
        # UNHCR와 동일한 날짜 파싱 로직
        raw = df[date_col].astype(str).str.strip()

        # YYYY-MM 형식 → 01일 보완
        mask_ym = raw.str.match(r"^\d{4}-\d{2}$")
        raw = raw.where(~mask_ym, raw + "-01")

        parsed = pd.to_datetime(raw, errors="coerce", dayfirst=False)

        # dayfirst 재시도 (DD/MM/YYYY 등)
        still_null = parsed.isna()
        if still_null.any():
            parsed[still_null] = pd.to_datetime(
                raw[still_null], errors="coerce", dayfirst=True
            )

        df["year"] = parsed.dt.year.astype("Int64")

        null_cnt = df["year"].isna().sum()
        if null_cnt:
            print(f"  ⚠️  날짜 파싱 실패 {null_cnt:,}행 → 제외")
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
        print(f"  ℹ️  날짜 컬럼 '{date_col}' 파싱 완료 "
              f"({df['year'].min()}~{df['year'].max()})")

    elif year_col:
        df["year"] = pd.to_numeric(df[year_col], errors="coerce")
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
        print(f"  ℹ️  year 컬럼 '{year_col}' 사용")

    else:
        # dis_no 형식: "2020-0001-AFG" → 앞 4자리가 연도
        dis_no_col = col(df, "dis_no", "disno", "disaster_no")
        if dis_no_col:
            df["year"] = df[dis_no_col].astype(str).str[:4]
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df.dropna(subset=["year"])
            df["year"] = df["year"].astype(int)
            print(f"  ℹ️  dis_no에서 연도 추출")
        else:
            raise ValueError(
                "EM-DAT: 날짜/연도 컬럼을 찾을 수 없습니다.\n"
                f"  현재 컬럼: {list(df.columns)}\n"
                "  필요 컬럼 (하나라도): date / start_date / year / start_year / dis_no"
            )

    # ── country 컬럼 처리 ────────────────────────────────────────────────
    if not ctry_col:
        raise ValueError(
            "EM-DAT: 국가 컬럼을 찾을 수 없습니다.\n"
            f"  현재 컬럼: {list(df.columns)}\n"
            "  필요 컬럼 (하나라도): country_std / country / iso / country_name"
        )
    # rename 대신 직접 할당 (중복 컬럼 방지)
    df["country_std"] = df[ctry_col]

    # ── 자연재해만 필터 (dtype_col 있을 경우) ────────────────────────────
    if dtype_col:
        df["disaster_type_clean"] = df[dtype_col].astype(str).str.strip()
        natural = df["disaster_type_clean"].isin(NATURAL_DISASTER_TYPES)
        print(f"  ℹ️  자연재해 필터: {natural.sum():,}건 / 전체 {len(df):,}건")
        df = df[natural].copy()
    else:
        print(f"  ⚠️  disaster_type 컬럼 없음 → 전체 이벤트 사용")

    # ── 사망자 숫자 처리 ─────────────────────────────────────────────────
    if deaths_col:
        df["deaths"] = pd.to_numeric(df[deaths_col], errors="coerce").fillna(0)
    else:
        df["deaths"] = 0

    # ── 연간 집계 ────────────────────────────────────────────────────────
    agg = (
        df.groupby(["country_std", "year"])
        .agg(
            disaster_count  = ("country_std", "count"),
            disaster_deaths = ("deaths",       "sum"),
        )
        .reset_index()
    )
    agg["disaster_flag"]   = 1
    agg["disaster_deaths"] = agg["disaster_deaths"].round(0).astype(int)

    # 재해 없는 국가+연도는 없으므로 0 행은 없음 (step3에서 COALESCE(0) 처리)

    if "group" in df.columns:
        grp_map = df.drop_duplicates("country_std").set_index("country_std")["group"]
        agg["group"] = agg["country_std"].map(grp_map).fillna("OTHER")
    else:
        agg["group"] = "OTHER"

    print(f"  ✅ EM-DAT: {len(agg):,}행  "
          f"국가 {agg['country_std'].nunique()}개  "
          f"연도 {int(agg['year'].min())}~{int(agg['year'].max())}")
    print(f"     재해 건수 총합: {int(agg['disaster_count'].sum()):,}")
    print(f"     재해 사망자 총합: {int(agg['disaster_deaths'].sum()):,}")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 6. 파일 처리 메인
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_NAMES = {
    "REIGN": "reign",
    "UNHCR": "unhcr",
    "FAO":   "fao",
    "EMDAT": "emdat",
}

STEP2_FEATURE_COLS = {
    "reign": ["PolitShock", "irregular_change_flag", "tenure_mean"],
    "unhcr": ["refugee_outflow", "idp_count", "asylum_seekers"],
    "fao":   ["food_price_index", "FoodShock"],
    "emdat": ["disaster_flag", "disaster_deaths", "disaster_count"],
}


def process_file(input_path: str, output_dir: str, all_countries: list | None = None):
    print(f"\n{'='*60}")
    print(f"  처리 중: {os.path.basename(input_path)}")

    df_raw = pd.read_csv(input_path, low_memory=False)
    print(f"  행 수: {len(df_raw):,}  /  컬럼: {list(df_raw.columns[:10])}...")

    dataset = detect_dataset(df_raw.columns.tolist())
    if dataset == "UNKNOWN":
        print(f"  ❌ 데이터셋 감지 실패. 지원 형식:")
        print(f"     REIGN  : irregular + tenure_months 컬럼 필요")
        print(f"     UNHCR  : refugees 또는 coo_name 컬럼 필요")
        print(f"     FAO    : pfcs / food_price / price_index 컬럼 필요")
        print(f"     EM-DAT : disaster_type + total_deaths 컬럼 필요")
        return None

    print(f"  🔍 감지된 데이터셋: {dataset}")

    try:
        if dataset == "REIGN":
            df_agg = process_reign(df_raw)
        elif dataset == "UNHCR":
            df_agg = process_unhcr(df_raw)
        elif dataset == "FAO":
            df_agg = process_fao(df_raw, all_countries=all_countries)
        elif dataset == "EMDAT":
            df_agg = process_emdat(df_raw)
        else:
            return None
    except Exception as e:
        print(f"  ❌ 처리 오류: {e}")
        return None

    out_name = OUTPUT_NAMES[dataset]
    out_path = os.path.join(output_dir, f"{out_name}.parquet")
    csv_path = os.path.join(output_dir, f"{out_name}_aggregated.csv")

    pq.write_table(pa.Table.from_pandas(df_agg), out_path, compression="snappy")
    df_agg.to_csv(csv_path, index=False, encoding="utf-8-sig")

    mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  💾 저장: {out_path}  ({mb:.2f} MB)")

    return dataset, out_name, df_agg


def print_step2_guide(processed: list):
    """step2 FEATURE_COLS 추가 안내."""
    if not processed:
        return
    print(f"\n{'='*60}")
    print("📌 step2_zscore_duckdb.py FEATURE_COLS 추가 항목:")
    print()
    for dataset, out_name, _ in processed:
        cols = STEP2_FEATURE_COLS.get(out_name, [])
        print(f'    "{out_name}": {cols},')
    print()
    print("📌 step3_wri_duckdb.py 는 자동으로 아래 파일을 인식합니다:")
    for dataset, out_name, _ in processed:
        print(f"    {out_name}_z.parquet → {dataset} 피처")


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="REIGN / UNHCR / FAO / EM-DAT → 국가+연도 Parquet 집계"
    )
    parser.add_argument("files", nargs="*", help="입력 CSV 파일 경로 (여러 개 가능)")
    parser.add_argument("--out", default=None, help="출력 폴더 (기본: ./01_parquet)")
    parser.add_argument(
        "--countries", default=None,
        help="FAO 전국 집계 브로드캐스트용 국가명 파일 (줄 구분 txt)"
    )
    args = parser.parse_args()

    file_list  = args.files
    output_dir = args.out or "./01_parquet"

    # 대화형 입력
    if not file_list:
        print("처리할 CSV 파일 경로를 입력하세요 (쉼표로 구분):")
        raw = input("  >> ").strip()
        if not raw:
            print("❌ 파일 경로 없음")
            return
        file_list = [p.strip().strip('"').strip("'")
                     for p in raw.replace(",", " ").split()]

    # FAO 브로드캐스트용 국가 목록
    all_countries = None
    if args.countries and os.path.exists(args.countries):
        with open(args.countries) as f:
            all_countries = [line.strip() for line in f if line.strip()]
        print(f"  ℹ️  FAO 브로드캐스트 국가 {len(all_countries)}개 로드")

    os.makedirs(output_dir, exist_ok=True)
    processed = []

    for fpath in file_list:
        if not os.path.exists(fpath):
            print(f"\n❌ 파일 없음: {fpath}")
            continue
        result = process_file(fpath, output_dir, all_countries=all_countries)
        if result:
            processed.append(result)

    print(f"\n{'='*60}")
    print(f"  완료: {len(processed)}개 파일 처리")
    print_step2_guide(processed)
    print(f"\n  다음 단계: python step2_zscore_duckdb.py {output_dir} 02_normalized/")


if __name__ == "__main__":
    main()