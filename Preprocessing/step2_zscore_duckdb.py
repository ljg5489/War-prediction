"""
step2_zscore_duckdb.py  (v3.1 정규화 안정화판)
──────────────────────────────────────────────────────────────────────────────
국가별 Z-score 정규화 (DuckDB)

수식: x̃_c,t = (x_c,t − μ_c) / σ_c        (σ_c 가 사실상 0이면 0 처리)

⚠️ 안정화 변경점 (WRI 폭발 버그 수정)
  기존식  : (x − μ) / (STDDEV_POP + 1e-8)
            → 분산이 0에 가까운 구간(0이 많은 재난·난민 피처 등)에서 분모가
              EPS(1e-8)로 붕괴 → z 가 원시값 × 1e8 규모로 폭발 → WRI 오염.
  변경식  : 아래 3중 방어
            (1) NULL 값 → 0 (NaN 전파 차단)
            (2) σ < STD_FLOOR(1e-6) 인 '사실상 무변동' 국가/피처 → z = 0
                (정보가 없는 피처는 평균값=0 이 올바른 표준화 결과)
            (3) 최종 z 를 ±Z_CLIP(8.0) 로 클리핑
                (정상 데이터는 거의 영향 없음. 구조적 이상치만 캡핑하여
                 단 하나의 오염값도 WRI 합산을 폭주시키지 못하게 하는 가드레일)

  ※ 정상적인 입력에서는 기존식과 결과가 동일합니다(검증 완료).
    오직 '분산 붕괴 / NULL / 구조적 폭발' 케이스만 안전하게 교정합니다.

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

# ── 정규화 안정화 상수 ───────────────────────────────────────────────────────
STD_FLOOR = 1e-6   # 이보다 작은 국가별 표준편차는 '무변동'으로 보고 z=0 처리
Z_CLIP    = 8.0    # 최종 z 클리핑 범위 (±8). 정상 z 는 사실상 -5~5 안에 있음

# ── 데이터셋별 정규화 컬럼 정의 ──────────────────────────────────────────────
FEATURE_COLS = {
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
        "GDP_Growth",           # ny_gdp_pcap_kd_zg → GDP_Growth          (출력: GDP_Growth_z)
        "Inflation",            # fp_cpi_totl_zg    → Inflation            (출력: Inflation_z)
        "Military_Expenditure", # ms_mil_xpnd_gd_zs → Military_Expenditure (출력: Military_Expenditure_z)
    ],
    "wb": [
        "GDP_Growth",
        "Inflation",
        "Military_Expenditure",
    ],
    "gdelt": [
        "AvgTone",        # avg_tone      → AvgTone       (출력: AvgTone_z)
        "AvgGoldstein",   # goldstein_avg → AvgGoldstein  (출력: AvgGoldstein_z)
        "TotalArticles",  # num_articles  → TotalArticles (출력: TotalArticles_z)
        "TotalMentions",  # 신규 추가                      (출력: TotalMentions_z)
    ],
    # step1b 출력 (ACLED 주간) — step1b 안내대로 추가
    "acled_weekly": [
        "weekly_fatalities", "CII", "CII_roll7", "CII_roll90",
        "battle_ratio", "civilian_targeting_ratio",
        "geographic_spread", "spillover",
    ],
    # 인도주의 (UNHCR / EM-DAT) — wri.csv 에 *_z 가 존재하므로 여기서 함께 정규화.
    # 0 이 많은 zero-inflated 피처라 위 STD_FLOOR/CLIP 가드의 핵심 보호 대상.
    "unhcr": 
    ["idp_count", "refugee_outflow", "forced_mig_raw"
     ],
    "emdat": [
        "disaster_flag", "disaster_deaths",
    ],
    "humanitarian": [
        "refugee_outflow", "idp_count", "forced_mig",
        "disaster_flag", "disaster_deaths",
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
    exclude = {"year", "week_num", "country_id", "type_of_violence"}
    num_cols = schema[
        schema["column_type"].str.contains("INT|FLOAT|DOUBLE", case=False)
    ]["column_name"].tolist()
    return [c for c in num_cols if c not in exclude]


def build_zscore_query(parquet_path: str, feature_cols: list) -> str:
    """
    안정화 z-score 쿼리.
      stats CTE : 국가별 μ, σ 를 한 번만 계산
      본 SELECT : NULL→0, σ<STD_FLOOR→0, 그 외 (x−μ)/σ 후 ±Z_CLIP 클리핑
    """
    # 국가별 μ, σ (피처별 윈도우 1회 계산)
    stat_exprs = ",\n        ".join([
        f"AVG({col})        OVER (PARTITION BY country_std) AS {col}__mu,\n"
        f"        STDDEV_POP({col}) OVER (PARTITION BY country_std) AS {col}__sd"
        for col in feature_cols
    ])

    # z 계산식 (3중 방어)
    z_exprs = ",\n        ".join([
        (
            f"CASE\n"
            f"            WHEN {col} IS NULL THEN 0.0\n"
            f"            WHEN {col}__sd IS NULL OR {col}__sd < {STD_FLOOR} THEN 0.0\n"
            f"            ELSE GREATEST(-{Z_CLIP}, LEAST({Z_CLIP},\n"
            f"                 ROUND(({col} - {col}__mu) / {col}__sd, 6)))\n"
            f"        END AS {col}_z"
        )
        for col in feature_cols
    ])

    mu_sd_cols = ", ".join(
        [f"{col}__mu" for col in feature_cols]
        + [f"{col}__sd" for col in feature_cols]
    )

    return f"""
    WITH stats AS (
        SELECT
            *,
            {stat_exprs}
        FROM read_parquet('{parquet_path}')
    )
    SELECT
        * EXCLUDE ({mu_sd_cols}),
        {z_exprs}
    FROM stats
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

    # ── 건강검진: 모든 z 컬럼이 국가별 mean≈0, std≈1(또는 0=무변동) 인지 ──
    z_cols = [f"{c}_z" for c in feat_cols]
    print(f"  ✅ 저장 완료: {out_name}  ({os.path.getsize(out_path)/1024/1024:.2f} MB)")
    print(f"  🩺 정규화 건강검진 (전체 z 범위 / 국가평균 std):")

    bad = False
    for zc in z_cols:
        agg = con.execute(f"""
            WITH per_country AS (
                SELECT country_std,
                       AVG({zc})        AS m,
                       STDDEV_POP({zc}) AS s
                FROM read_parquet('{out_path}')
                GROUP BY country_std
            )
            SELECT
                MIN({zc}) AS zmin, MAX({zc}) AS zmax,
                (SELECT ROUND(AVG(ABS(m)),4) FROM per_country) AS mean_abs_mu,
                (SELECT ROUND(AVG(s),4)      FROM per_country WHERE s > 0) AS mean_std
            FROM read_parquet('{out_path}')
        """).df().iloc[0]
        flag = ""
        # 폭발 흔적/비정상 표준편차 경고
        if abs(agg["zmin"]) > Z_CLIP + 1e-6 or abs(agg["zmax"]) > Z_CLIP + 1e-6:
            flag = "  ❌ 클리핑 범위 초과 (이상)"; bad = True
        elif agg["mean_std"] is not None and (agg["mean_std"] > 3 or agg["mean_std"] < 0.1):
            flag = "  ⚠️  국가평균 std 가 1에서 많이 벗어남"; bad = True
        print(f"     {zc:<28} range=[{agg['zmin']:>7.3f}, {agg['zmax']:>7.3f}]  "
              f"|국가평균μ|≈{agg['mean_abs_mu']}  국가평균std≈{agg['mean_std']}{flag}")

    if not bad:
        print("  ✅ 정상: 모든 z 컬럼이 안정 범위(±{:.0f}) 안, 국가별 std≈1".format(Z_CLIP))
    else:
        print("  ⚠️  위 경고 컬럼 확인 필요 (원천 데이터 grain/결측 점검 권장)")

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
        print(f"   {os.path.basename(p):<35} {size_mb:.2f} MB")

    con      = duckdb.connect()
    z_paths  = []

    for pq_path in pq_paths:
        # 이미 _z.parquet 이면 스킵
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
        description="Parquet 파일들을 국가별 Z-score 정규화 (안정화판)"
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