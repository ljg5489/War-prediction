"""
step1d_aggregate_unhcr.py
──────────────────────────────────────────────────────────────────────────────
UNHCR → 국가+연도 단위 인도주의 피처 정리

이 UNHCR 파일은 이미 국가-연도 단위라(이벤트 집계 불필요), 컬럼 이름만
파이프라인 표준으로 맞추고 중복 국가-연도만 합산한다.

생성 피처:
  idp_count        = idps        (국내 실향민 수)
  refugee_outflow  = refugees    (해당 국가 출신 난민 수 = 유출)
  asylum_outflow   = asylum_seekers (참고)
  forced_mig_raw   = ForcedMig   (참고; F15 = (IDP+Refugee)/Pop 계산 시 분자)

파이프라인 위치:  normalize_and_group.py → step1d → step0(또는 직접 step2) → step3

사용법:
    python step1d_aggregate_unhcr.py <unhcr_grouped.csv> [--out 폴더]
    python step1d_aggregate_unhcr.py   (인자 없으면 경로 입력 프롬프트)

출력:  unhcr.parquet  +  unhcr_aggregated.csv
  → step2 FEATURE_COLS 에 추가:
       "unhcr": ["idp_count", "refugee_outflow"],
──────────────────────────────────────────────────────────────────────────────
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import argparse
import os

# 원본 컬럼 → 표준 피처명
RENAME = {
    "idps":           "idp_count",
    "refugees":       "refugee_outflow",
    "asylum_seekers": "asylum_outflow",
    "ForcedMig":      "forced_mig_raw",
}


def aggregate(input_path: str, output_dir: str):
    print(f"\n📂 로딩: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"   행 수: {len(df):,}  /  컬럼: {list(df.columns)}")

    if "country_std" not in df.columns:
        print("❌ country_std 없음 → normalize_and_group.py 를 먼저 실행하세요.")
        return
    if "year" not in df.columns:
        print("❌ year 컬럼 없음.")
        return

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]); df["year"] = df["year"].astype(int)

    # 표준 컬럼만 추출 (존재하는 것만)
    value_cols = [src for src in RENAME if src in df.columns]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=0).fillna(0.0)

    # 국가-연도 중복 합산 (보통 1행/국가-연도라 사실상 그대로)
    agg = (df.groupby(["country_std", "year"], as_index=False)
             .agg({**{c: "sum" for c in value_cols},
                   "group": "first"}))
    agg = agg.rename(columns=RENAME)
    keep = ["year", "country_std", "group"] + [RENAME[c] for c in value_cols]
    agg = agg[keep].sort_values(["country_std", "year"])

    print(f"\n✅ 정리 완료: {len(agg):,}행 (국가+연도)")
    print(f"   국가 수: {agg['country_std'].nunique()}  "
          f"/ 연도: {agg['year'].min()}~{agg['year'].max()}")
    print("\n📊 그룹별 합계")
    sum_cols = [RENAME[c] for c in value_cols]
    print(agg.groupby("group")[sum_cols].sum().reset_index().to_string(index=False))
    print("\n🔥 IDP 최다 상위 8 (국가+연도)")
    if "idp_count" in agg.columns:
        print(agg.nlargest(8, "idp_count")[
            ["country_std", "year", "group", "idp_count", "refugee_outflow"]
        ].reset_index(drop=True).to_string(index=False))

    os.makedirs(output_dir, exist_ok=True)
    out_pq  = os.path.join(output_dir, "unhcr.parquet")
    out_csv = os.path.join(output_dir, "unhcr_aggregated.csv")
    pq.write_table(pa.Table.from_pandas(agg), out_pq, compression="snappy")
    agg.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 저장: {out_pq} ({os.path.getsize(out_pq)/1024:.1f} KB)")
    print('\n📌 step2 FEATURE_COLS 에 추가: "unhcr": ["idp_count", "refugee_outflow"],')
    return out_pq


def main():
    p = argparse.ArgumentParser(description="UNHCR 국가-연도 피처 정리")
    p.add_argument("file", nargs="?", help="UNHCR (grouped) CSV")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    fp = args.file
    if not fp:
        print("UNHCR grouped CSV 경로:")
        fp = input("  >> ").strip().strip('"').strip("'")
    if not os.path.exists(fp):
        print(f"❌ 파일 없음: {fp}"); return
    out = args.out or os.path.dirname(os.path.abspath(fp))
    aggregate(fp, out)


if __name__ == "__main__":
    main()
