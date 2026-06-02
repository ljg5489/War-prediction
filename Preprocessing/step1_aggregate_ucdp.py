"""
step1_aggregate_ucdp.py
──────────────────────────────────────────────────────────────────────────────
이벤트 단위 분쟁 데이터 집계 (event → country+year)

UCDP 와 ACLED 모두 지원합니다. 파일 컬럼을 보고 자동으로 판별합니다.

컬럼 매핑:
  UCDP  : best → total_deaths / type_of_violence → 분류 / date_start → active_months
  ACLED : fatalities → total_deaths / event_type → 분류 / event_date → active_months

ACLED event_type 분류:
  deaths_state    ← Battles
  deaths_onesided ← Violence against civilians
  deaths_nonstate ← Riots / Explosions·Remote violence / Strategic developments / Protests

사용법:
    python step1_aggregate_ucdp.py <grouped.csv>
    python step1_aggregate_ucdp.py   (인자 없으면 경로 입력 프롬프트)

입력:  normalize_and_group.py 출력 CSV (country_std, group 컬럼 포함)
출력:  ucdp.parquet  +  ucdp_aggregated.csv
──────────────────────────────────────────────────────────────────────────────
"""

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import argparse
import os

# ── 데이터셋 감지 ─────────────────────────────────────────────────────────────
def detect_dataset(cols: list) -> str:
    """컬럼 목록으로 UCDP / ACLED 판별 (대소문자 무시)."""
    lower = [c.lower() for c in cols]
    if "best" in lower and "type_of_violence" in lower:
        return "UCDP"
    if "fatalities" in lower and "event_type" in lower:
        return "ACLED"
    return "UNKNOWN"


# ── UCDP 집계 쿼리 ───────────────────────────────────────────────────────────
UCDP_QUERY = """
SELECT
    year,
    country_std,
    COALESCE("group", 'OTHER')                                             AS "group",

    COUNT(*)                                                               AS event_count,
    COALESCE(SUM(best), 0)                                                 AS total_deaths,

    COALESCE(SUM(CASE WHEN type_of_violence = 1 THEN best ELSE 0 END), 0) AS deaths_state,
    COALESCE(SUM(CASE WHEN type_of_violence = 2 THEN best ELSE 0 END), 0) AS deaths_nonstate,
    COALESCE(SUM(CASE WHEN type_of_violence = 3 THEN best ELSE 0 END), 0) AS deaths_onesided,

    COUNT(CASE WHEN type_of_violence = 1 THEN 1 END)                      AS events_state,
    COUNT(CASE WHEN type_of_violence = 2 THEN 1 END)                      AS events_nonstate,
    COUNT(CASE WHEN type_of_violence = 3 THEN 1 END)                      AS events_onesided,

    COUNT(DISTINCT MONTH(CAST(date_start AS DATE)))                        AS active_months

FROM df_raw
GROUP BY year, country_std, "group"
ORDER BY country_std, year
"""

def build_acled_query(cols: list) -> str:
    """ACLED 집계 쿼리 생성. 컬럼 존재 여부에 따라 active_months 표현식 결정."""
    if "event_date" in cols:
        active_months_expr = "COUNT(DISTINCT MONTH(CAST(event_date AS DATE)))"
    elif "week" in cols:
        # WEEK 형식 "YYYY-WW" → 주차를 월로 근사 (4.33주 = 1개월)
        active_months_expr = (
            "COUNT(DISTINCT CAST(CEIL("
            "CAST(SUBSTRING(CAST(week AS VARCHAR), 6, 2) AS DOUBLE) / 4.33"
            ") AS INTEGER))"
        )
    else:
        active_months_expr = "NULL"

    return f"""
SELECT
    year,
    country_std,
    COALESCE("group", 'OTHER')                                                    AS "group",

    COUNT(*)                                                                      AS event_count,
    COALESCE(SUM(fatalities), 0)                                                  AS total_deaths,

    COALESCE(SUM(CASE WHEN event_type = 'Battles'
                      THEN fatalities ELSE 0 END), 0)                            AS deaths_state,

    COALESCE(SUM(CASE WHEN event_type IN (
                          'Riots',
                          'Explosions/Remote violence',
                          'Strategic developments',
                          'Protests'
                      ) THEN fatalities ELSE 0 END), 0)                          AS deaths_nonstate,

    COALESCE(SUM(CASE WHEN event_type = 'Violence against civilians'
                      THEN fatalities ELSE 0 END), 0)                            AS deaths_onesided,

    COUNT(CASE WHEN event_type = 'Battles' THEN 1 END)                           AS events_state,
    COUNT(CASE WHEN event_type IN (
                   'Riots','Explosions/Remote violence',
                   'Strategic developments','Protests'
               ) THEN 1 END)                                                     AS events_nonstate,
    COUNT(CASE WHEN event_type = 'Violence against civilians' THEN 1 END)        AS events_onesided,

    {active_months_expr}                                                         AS active_months

FROM df_raw
GROUP BY year, country_std, "group"
ORDER BY country_std, year
"""


# ── 데이터 분포 확인 ──────────────────────────────────────────────────────────
def print_distribution(con, dataset: str):
    if dataset == "UCDP":
        print("\n📊 type_of_violence 분포")
        print(con.execute("""
            SELECT type_of_violence,
                   COUNT(*) AS events,
                   SUM(best) AS deaths
            FROM df_raw
            GROUP BY type_of_violence
            ORDER BY type_of_violence
        """).df().to_string(index=False))

    elif dataset == "ACLED":
        print("\n📊 event_type 분포")
        print(con.execute("""
            SELECT event_type,
                   COUNT(*) AS events,
                   SUM(fatalities) AS deaths
            FROM df_raw
            GROUP BY event_type
            ORDER BY events DESC
        """).df().to_string(index=False))


# ── 메인 처리 ────────────────────────────────────────────────────────────────
def aggregate(input_path: str):
    print(f"\n📂 로딩: {input_path}")
    df_raw = pd.read_csv(input_path, low_memory=False)
    print(f"   행 수: {len(df_raw):,}  /  컬럼: {list(df_raw.columns)}")

    # 데이터셋 판별
    dataset = detect_dataset(df_raw.columns.tolist())
    if dataset == "UNKNOWN":
        print("❌ UCDP / ACLED 판별 실패.")
        print("   UCDP 필수 컬럼: best, type_of_violence")
        print("   ACLED 필수 컬럼: fatalities, event_type")
        return
    print(f"\n🔍 데이터셋 감지: {dataset}")

    # ── 컬럼명 소문자 통일 ────────────────────────────────────────────────────
    df_raw.columns = [c.lower() for c in df_raw.columns]

    # ── year 컬럼 없으면 추출 ─────────────────────────────────────────────────
    if "year" not in df_raw.columns:
        if "week" in df_raw.columns:
            # WEEK 형식: "2020-01" 또는 "2020-W01" → 앞 4자리가 연도
            df_raw["year"] = df_raw["week"].astype(str).str[:4].astype(int)
            print(f"   ℹ️  year 컬럼 없음 → WEEK 컬럼에서 추출")
        elif "event_date" in df_raw.columns:
            df_raw["year"] = pd.to_datetime(
                df_raw["event_date"], errors="coerce"
            ).dt.year
            print(f"   ℹ️  year 컬럼 없음 → event_date 에서 추출")
        else:
            print("❌ year / week / event_date 컬럼을 찾을 수 없습니다.")
            return

    con = duckdb.connect()
    con.register("df_raw", df_raw)   # pandas DataFrame 명시적 등록

    # ── 구조 확인 ─────────────────────────────────────────────────────────────
    print("\n📊 전체 행 수 / 연도 범위")
    print(con.execute("""
        SELECT
            COUNT(*)                    AS total_events,
            COUNT(DISTINCT country_std) AS countries,
            MIN(year)                   AS year_min,
            MAX(year)                   AS year_max
        FROM df_raw
    """).df().to_string(index=False))

    print_distribution(con, dataset)

    print("\n📊 그룹별 국가 수")
    print(con.execute("""
        SELECT
            COALESCE("group", 'OTHER') AS grp,
            COUNT(DISTINCT country_std) AS country_count,
            COUNT(*) AS events
        FROM df_raw
        GROUP BY grp
        ORDER BY grp
    """).df().to_string(index=False))

    # ── 집계 ─────────────────────────────────────────────────────────────────
    print("\n🔢 집계 중...")
    query  = UCDP_QUERY if dataset == "UCDP" else build_acled_query(df_raw.columns.tolist())
    df_agg = con.execute(query).df()

    print(f"✅ 집계 완료: {len(df_agg):,}행 (나라+연도 단위)")
    print(f"   국가 수: {df_agg['country_std'].nunique()}개")
    print(f"   연도 범위: {int(df_agg['year'].min())}~{int(df_agg['year'].max())}")

    # ── 결과 확인 ─────────────────────────────────────────────────────────────
    print("\n📄 앞 10행")
    print(df_agg.head(10).to_string(index=False))

    print("\n📊 그룹별 집계 요약")
    print(df_agg.groupby("group")[
        ["event_count", "total_deaths", "deaths_state", "deaths_nonstate", "deaths_onesided"]
    ].sum().reset_index().to_string(index=False))

    print("\n📊 사망자 최다 상위 10개 (나라+연도)")
    print(df_agg.nlargest(10, "total_deaths")[
        ["country_std", "year", "group", "event_count", "total_deaths"]
    ].reset_index(drop=True).to_string(index=False))

    # ── 저장 ─────────────────────────────────────────────────────────────────
    out_dir     = os.path.dirname(os.path.abspath(input_path))
    out_parquet = os.path.join(out_dir, "ucdp.parquet")
    out_csv     = os.path.join(out_dir, "ucdp_aggregated.csv")

    pq.write_table(pa.Table.from_pandas(df_agg), out_parquet, compression="snappy")
    df_agg.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n✅ 저장 완료")
    print(f"   {out_parquet}  ({os.path.getsize(out_parquet)/1024:.1f} KB)")
    print(f"   {out_csv}  ({os.path.getsize(out_csv)/1024:.1f} KB)")
    print(f"\n📌 다음 단계: ucdp.parquet 를 step2 입력 폴더에 넣고 step2 실행")

    con.close()
    return out_parquet


# ── 진입점 ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="UCDP / ACLED 이벤트 단위 CSV → 나라+연도 단위 Parquet 집계"
    )
    parser.add_argument("file", nargs="?", help="입력 CSV 파일 경로")
    args = parser.parse_args()

    input_path = args.file
    if not input_path:
        print("분쟁 데이터 CSV 파일 경로를 입력하세요 (UCDP 또는 ACLED):")
        input_path = input("  >> ").strip().strip('"').strip("'")

    if not os.path.exists(input_path):
        print(f"❌ 파일 없음: {input_path}")
        return

    aggregate(input_path)


if __name__ == "__main__":
    main()
