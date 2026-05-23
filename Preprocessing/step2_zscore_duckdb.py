"""
step2_zscore_duckdb.py
──────────────────────────────────────────────────────────────────────────────
국가별 Z-score 정규화 (DuckDB)

수식: x̃_c,t = (x_c,t − μ_c) / (σ_c + ε)

v3.1 추가:
  - acled_weekly : ACLED 주간 피처 (step1b 출력)
  - reign        : PolitShock, irregular_change_flag (step1c 출력)
  - unhcr        : refugee_outflow, idp_count (step1c 출력)
  - fao          : food_price_index, FoodShock (step1c 출력)
  - emdat        : disaster_flag, disaster_deaths (step1c 출력)

사용법:
    python step2_zscore_duckdb.py <입력폴더> <출력폴더>
    python step2_zscore_duckdb.py   (인자 없으면 경로 입력 프롬프트)

입력:  Parquet 파일들이 있는 폴더 (ucdp.parquet, qog.parquet 등)
출력:  *_z.parquet 파일들 (출력폴더에 저장)
──────────────────────────────────────────────────────────────────────────────
"""

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import os
import argparse

EPS = 1e-8

# ── 데이터셋별 정규화 컬럼 정의 ──────────────────────────────────────────────
FEATURE_COLS = {
    # ── 기존 데이터셋 ─────────────────────────────────────────────────────
    "ucdp": [
        "event_count", "total_deaths",
        "deaths_state", "deaths_nonstate", "deaths_onesided",
    ],
    "qog": [
        "al_ethnic2000",
    ],
    "vdem": [
        "v2x_libdem", "v2elintim", "v2x_corr",
    ],
    "v-dem": [
        "v2x_libdem", "v2elintim", "v2x_corr",
    ],
    "worldbank": [
        "GDP_Growth", "Inflation", "Military_Expenditure", "Unemployment",
    ],
    "wb": [
        "GDP_Growth", "Inflation", "Military_Expenditure", "Unemployment",
    ],
    "gdelt": [
        "AvgTone", "AvgGoldstein", "TotalArticles", "TotalMentions",
    ],

    # ── v3.1 추가 데이터셋 ────────────────────────────────────────────────

    # ACLED 주간 집계 (step1b 출력)
    # Z_CII / ESC 는 step3에서 계산하므로 raw 값만 정규화
    "acled_weekly": [
        "weekly_fatalities", "CII", "CII_roll7", "CII_roll90",
        "battle_ratio", "civilian_targeting_ratio",
        "geographic_spread", "spillover",
    ],

    # REIGN 정치충격 (step1c 출력)
    "reign": [
        "PolitShock", "irregular_change_flag", "tenure_mean",
    ],

    # UNHCR 난민·실향민 (step1c 출력)
    "unhcr": [
        "refugee_outflow", "idp_count", "asylum_seekers",
    ],

    # FAO FPMA 식량가격 (step1c 출력)
    "fao": [
        "food_price_index", "FoodShock",
    ],

    # EM-DAT 자연재해 (step1c 출력)
    "emdat": [
        "disaster_flag", "disaster_deaths", "disaster_count",
    ],
}


def get_feature_cols(parquet_path: str, con) -> list:
    """파일명 키워드로 정규화 대상 컬럼 결정. 실제 존재하는 컬럼만 반환."""
    fname = os.path.basename(parquet_path).lower()
    actual_cols = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')"
    ).df()["column_name"].tolist()

    for keyword, cols in FEATURE_COLS.items():
        if keyword in fname:
            found   = [c for c in cols if c in actual_cols]
            missing = [c for c in cols if c not in actual_cols]
            if missing:
                print(f"   ⚠️  없는 컬럼 스킵: {missing}")
            return found

    # 매핑 실패 → 숫자형 컬럼 자동 선택
    schema  = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')"
    ).df()
    exclude = {"year", "country_id", "type_of_violence", "week_num"}
    num_cols = schema[
        schema["column_type"].str.contains("INT|FLOAT|DOUBLE|DECIMAL", case=False)
    ]["column_name"].tolist()
    return [c for c in num_cols if c not in exclude]


def build_zscore_query(parquet_path: str, feature_cols: list) -> str:
    z_exprs = ",\n        ".join([
        f"ROUND((\n"
        f"    ({col} - AVG({col}) OVER (PARTITION BY country_std))\n"
        f"    / (STDDEV_POP({col}) OVER (PARTITION BY country_std) + {EPS})"
        f"), 6) AS {col}_z"
        for col in feature_cols
    ])
    return f"""
    SELECT
        *,
        {z_exprs}
    FROM read_parquet('{parquet_path}')
    """


def zscore_parquet(parquet_path: str, output_dir: str, con) -> str | None:
    fname    = os.path.basename(parquet_path)
    out_name = fname.replace(".parquet", "_z.parquet")
    out_path = os.path.join(output_dir, out_name)

    print(f"\n{'='*55}")
    print(f"  입력: {fname}")

    feat_cols = get_feature_cols(parquet_path, con)
    if not feat_cols:
        print("  ⚠️  정규화할 컬럼 없음 — 스킵")
        return None
    print(f"  정규화 컬럼: {feat_cols}")

    query = build_zscore_query(parquet_path, feat_cols)

    con.execute(f"""
        COPY ({query})
        TO '{out_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)

    # 검증
    chk = con.execute(f"""
        SELECT country_std,
               ROUND(AVG({feat_cols[0]}_z), 4)    AS mean_z,
               ROUND(STDDEV({feat_cols[0]}_z), 4) AS std_z
        FROM read_parquet('{out_path}')
        GROUP BY country_std
        LIMIT 5
    """).df()

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  ✅ 저장 완료: {out_name}  ({size_mb:.2f} MB)")
    print(f"  검증 (mean≈0, std≈1 이면 정상):")
    print(chk.to_string(index=False))
    return out_path


def run(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    pq_paths = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not pq_paths:
        print(f"❌ '{input_dir}' 에 Parquet 파일이 없습니다.")
        return

    print(f"✅ Parquet 파일 {len(pq_paths)}개 발견")
    for p in pq_paths:
        size_mb = os.path.getsize(p) / 1024 / 1024
        print(f"   {os.path.basename(p):<40} {size_mb:.2f} MB")

    con     = duckdb.connect()
    z_paths = []

    for pq_path in pq_paths:
        if pq_path.endswith("_z.parquet"):
            continue
        out = zscore_parquet(pq_path, output_dir, con)
        if out:
            z_paths.append(out)

    print(f"\n✅ 전체 완료: {len(z_paths)}개 파일")
    print(f"   저장 위치: {output_dir}")
    print(f"\n📌 다음 단계: step3 실행 시 출력 폴더 경로 사용")

    con.close()


# ── 진입점 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parquet 파일들을 국가별 Z-score 정규화"
    )
    parser.add_argument("input_dir",  nargs="?", help="입력 폴더 (Parquet 파일들)")
    parser.add_argument("output_dir", nargs="?", help="출력 폴더 (*_z.parquet 저장)")
    args = parser.parse_args()

    input_dir  = args.input_dir
    output_dir = args.output_dir

    if not input_dir:
        print("입력 폴더 경로를 입력하세요 (Parquet 파일들이 있는 폴더):")
        input_dir = input("  >> ").strip().strip('"').strip("'")

    if not output_dir:
        default_out = os.path.join(os.path.dirname(input_dir), "02_normalized")
        print(f"출력 폴더 경로를 입력하세요 (엔터 = {default_out}):")
        output_dir = input("  >> ").strip().strip('"').strip("'")
        if not output_dir:
            output_dir = default_out

    if not os.path.isdir(input_dir):
        print(f"❌ 폴더 없음: {input_dir}")
        return

    run(input_dir, output_dir)


if __name__ == "__main__":
    main()
