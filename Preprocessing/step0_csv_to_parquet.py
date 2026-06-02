"""
step0_csv_to_parquet.py
──────────────────────────────────────────────────────────────────────────────
CSV → Parquet 변환 (QOG / V-Dem / WorldBank / GDELT 전용) - 대용량 OOM 방지 완료 버전

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
              "GDP_Growth", "Inflation", "Military_Expenditure"],
    "gdelt": ["SQLDATE", "ActionGeo_CountryCode",
              "AvgTone", "AvgGoldstein", "TotalArticles"],
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

    # normalize_and_group 실행 여부 확인
    if "country_std" not in cols:
        print(f"  ⚠️  country_std 컬럼 없음 → normalize_and_group.py 를 먼저 실행하세요.")
        print(f"      올바른 순서: normalize_and_group.py → step0 → step2 → step3")
        return

    # 연도 범위 — year 또는 week 컬럼 자동 감지
    has_year = "year" in cols
    has_week = any(c.lower() in ("week", "week_id") for c in cols)

    if has_year:
        summary = con.execute(f"""
            SELECT COUNT(DISTINCT country_std) AS countries,
                   MIN(year) AS t_min, MAX(year) AS t_max
            FROM read_parquet('{parquet_path}')
        """).df()
        print(f"  국가 수: {summary['countries'].iloc[0]}  "
              f"/ 연도: {int(summary['t_min'].iloc[0])}~{int(summary['t_max'].iloc[0])}")
    elif has_week:
        week_col = next(c for c in cols if c.lower() in ("week", "week_id"))
        summary  = con.execute(f"""
            SELECT COUNT(DISTINCT country_std) AS countries,
                   MIN("{week_col}") AS t_min, MAX("{week_col}") AS t_max
            FROM read_parquet('{parquet_path}')
        """).df()
        print(f"  국가 수: {summary['countries'].iloc[0]}  "
              f"/ 기간: {summary['t_min'].iloc[0]}~{summary['t_max'].iloc[0]}")
    else:
        n = con.execute(
            f"SELECT COUNT(DISTINCT country_std) FROM read_parquet('{parquet_path}')"
        ).fetchone()[0]
        print(f"  국가 수: {n}  / 시간 컬럼 없음")

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

    before_mb = os.path.getsize(csv_path) / 1024 / 1024
    print(f"  CSV 크기: {before_mb:.1f} MB")

    # ── DuckDB 스트리밍으로 변환 (메모리 문제 없음) ───────────────────────
    print(f"  변환 중 (DuckDB 스트리밍)...")
    try:
        con.execute(f"""
            COPY (
                SELECT * FROM read_csv(
                    '{csv_path}',
                    ignore_errors = true,
                    null_padding  = true
                )
            )
            TO '{out_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)
    except Exception as e:
        # 🔴 핵심 수정: DuckDB 실패 시 pandas 청크 스트리밍 방식으로 안정적인 전환 (OOM 원천 차단)
        print(f"  ⚠️  DuckDB 변환 실패 ({e})")
        print(f"  pandas + PyArrow 스트리밍 방식으로 안전하게 재시도 중...")
        
        writer = None
        # 데이터 유실 방지를 위해 chunksize는 유지하되 파일에 직접 스트리밍 기록
        for chunk in pd.read_csv(csv_path, chunksize=500_000,
                                  low_memory=True, on_bad_lines="skip"):
            # pandas DataFrame을 PyArrow Table로 변환
            table = pa.Table.from_pandas(chunk)
            
            # 첫 청크에서 스키마를 정의하여 라이터 인스턴스 생성
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            
            # 청크 단위 데이터를 실시간으로 디스크 파일에 추가 저장 (Append)
            writer.write_table(table)
            
        # 작업 종료 후 안전하게 스트림 닫기
        if writer is not None:
            writer.close()

    after_mb = os.path.getsize(out_path) / 1024 / 1024
    rows     = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_path}')"
    ).fetchone()[0]
    print(f"  Parquet: {rows:,}행  /  {after_mb:.1f} MB  "
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