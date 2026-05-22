"""
step0_csv_to_parquet.py
──────────────────────────────────────────────────────────────────────────────
CSV → Parquet 변환 (QOG / V-Dem / WorldBank / GDELT 전용)

UCDP는 step1에서 집계 후 자동으로 Parquet 저장됩니다.
나머지 데이터셋은 이 스크립트로 변환 후 step2에 입력하세요.

파일명 규칙 (step2/step3 자동 감지용):
    qog_*.csv        → qog.parquet
    vdem_*.csv       → vdem.parquet
    v-dem_*.csv      → vdem.parquet
    worldbank_*.csv  → wb.parquet
    wb_*.csv         → wb.parquet
    gdelt_*.csv      → gdelt.parquet
    그 외             → 원본파일명.parquet

사용법:
    python step0_csv_to_parquet.py <CSV1> [<CSV2> ...] --out <출력폴더>
    python step0_csv_to_parquet.py   (인자 없으면 경로 입력 프롬프트)

출력:  01_parquet/*.parquet  (Snappy 압축)
──────────────────────────────────────────────────────────────────────────────
"""

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import argparse
import os

# ── 파일명 키워드 → 표준 출력명 매핑 ─────────────────────────────────────────
NAME_MAP = {
    "qog":        "qog",
    "vdem":       "vdem",
    "v-dem":      "vdem",
    "worldbank":  "wb",
    "world_bank": "wb",
    "wb":         "wb",
    "gdelt":      "gdelt",
}

# ── 데이터셋별 필요 컬럼 (없는 컬럼은 경고만 출력) ────────────────────────────
EXPECTED_COLS = {
    "qog":   ["year", "country_std", "group", "al_ethnic2000"],
    "vdem":  ["year", "country_std", "group", "v2x_libdem", "v2elintim", "v2x_corr"],
    "wb":    ["year", "country_std", "group",
              "ny_gdp_pcap_kd_zg", "fp_cpi_totl_zg", "ms_mil_xpnd_gd_zs"],
    "gdelt": ["year", "country_std", "group", "avg_tone", "goldstein_avg"],
}


def detect_output_name(csv_path: str) -> str:
    """파일명 키워드로 표준 출력명 결정."""
    fname = os.path.basename(csv_path).lower()
    for keyword, name in NAME_MAP.items():
        if keyword in fname:
            return name
    # 매핑 실패 → 원본 파일명 그대로 사용
    return os.path.splitext(os.path.basename(csv_path))[0]


def validate(parquet_path: str, dataset_key: str, con):
    """변환된 Parquet 파일 검증."""
    rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')"
    ).fetchone()[0]

    cols = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')"
    ).df()["column_name"].tolist()

    print(f"  검증: {rows:,}행  /  {len(cols)}컬럼")

    # 연도 범위 & 국가 수
    summary = con.execute(f"""
        SELECT
            COUNT(DISTINCT country_std) AS countries,
            MIN(year) AS year_min,
            MAX(year) AS year_max
        FROM read_parquet('{parquet_path}')
    """).df()
    print(f"  국가 수: {summary['countries'].iloc[0]}  "
          f"/ 연도: {int(summary['year_min'].iloc[0])}~{int(summary['year_max'].iloc[0])}")

    # 필요 컬럼 존재 여부
    expected = EXPECTED_COLS.get(dataset_key, [])
    missing  = [c for c in expected if c not in cols]
    if missing:
        print(f"  ⚠️  없는 컬럼: {missing}")
        print(f"      → normalize_and_group.py 를 먼저 실행했는지 확인하세요.")
    else:
        print(f"  ✅ 필수 컬럼 모두 확인")


def convert(csv_path: str, output_dir: str, con) -> str:
    print(f"\n{'='*55}")
    print(f"  입력: {os.path.basename(csv_path)}")

    # 출력 파일명 결정
    dataset_key = detect_output_name(csv_path)
    out_path    = os.path.join(output_dir, f"{dataset_key}.parquet")

    # 이미 존재하면 경고
    if os.path.exists(out_path):
        print(f"  ⚠️  이미 존재: {os.path.basename(out_path)} → 덮어씁니다")

    # CSV 로딩
    print(f"  로딩 중...")
    df = pd.read_csv(csv_path, low_memory=False)
    before_mb = os.path.getsize(csv_path) / 1024 / 1024
    print(f"  CSV: {len(df):,}행  ×  {len(df.columns)}컬럼  ({before_mb:.1f} MB)")

    # Parquet 저장
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path, compression="snappy")
    after_mb = os.path.getsize(out_path) / 1024 / 1024

    print(f"  Parquet: {after_mb:.1f} MB  "
          f"(압축률 {(1 - after_mb/before_mb)*100:.0f}% 감소)")
    print(f"  저장: {out_path}")

    # 검증
    validate(out_path, dataset_key, con)

    return out_path


def run(csv_paths: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    con = duckdb.connect()

    converted = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"\n❌ 파일 없음: {csv_path}")
            continue
        if not csv_path.lower().endswith(".csv"):
            print(f"\n❌ CSV 파일만 지원합니다: {csv_path}")
            continue
        try:
            out = convert(csv_path, output_dir, con)
            converted.append(out)
        except Exception as e:
            print(f"\n❌ 오류 ({os.path.basename(csv_path)}): {e}")

    con.close()

    print(f"\n{'='*55}")
    print(f"  변환 완료: {len(converted)}개 파일")
    print(f"  저장 위치: {output_dir}")
    if converted:
        print(f"\n📌 다음 단계: python step2_zscore_duckdb.py {output_dir} <출력폴더>")


# ── 진입점 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CSV 파일을 Parquet(Snappy)으로 변환 (QOG/V-Dem/WB/GDELT)"
    )
    parser.add_argument(
        "files", nargs="*",
        help="변환할 CSV 파일 경로 (여러 개 가능)"
    )
    parser.add_argument(
        "--out", default=None,
        help="출력 폴더 (기본: ./01_parquet)"
    )
    args = parser.parse_args()

    csv_paths  = args.files
    output_dir = args.out

    # 인자 없으면 대화형 입력
    if not csv_paths:
        print("변환할 CSV 파일 경로를 입력하세요 (여러 개면 쉼표로 구분):")
        raw = input("  >> ").strip()
        if not raw:
            print("❌ 파일 경로가 입력되지 않았습니다.")
            return
        csv_paths = [p.strip().strip('"').strip("'")
                     for p in raw.replace(",", " ").split()]

    if not output_dir:
        # 첫 번째 파일 기준으로 기본 출력 폴더 설정
        default_out = os.path.join(os.path.dirname(csv_paths[0]) or ".", "01_parquet")
        print(f"출력 폴더를 입력하세요 (엔터 = {default_out}):")
        output_dir = input("  >> ").strip().strip('"').strip("'")
        if not output_dir:
            output_dir = default_out

    run(csv_paths, output_dir)


if __name__ == "__main__":
    main()
