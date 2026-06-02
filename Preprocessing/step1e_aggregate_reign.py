"""
step1e_aggregate_reign.py
──────────────────────────────────────────────────────────────────────────────
REIGN 국가-월-지도자 단위 → 국가+연도 단위 집계 (정치충격 피처)

⚠️  이 REIGN 파일의 제약
  - 표준 REIGN 의 쿠데타 컬럼(pt_suc / pt_attempt)이 없음
  - 'irregular' 컬럼이 0/1 플래그가 아니라 로그변환 연속값(0, ln2, ln3 …)이라
    비정상교체 플래그로 사용 불가
  → 그래서 살아있는 신호(선거 발생 election_now, 지도자 이름 변화)로 구성하고
    irregular_change_flag 는 '선거 없이 일어난 지도자 교체 = 비정상'의 근사치로 만든다.
  → 진짜 쿠데타 등급(1.0)이 필요하면 pt_suc/pt_attempt 가 포함된 원본 REIGN 필요.

생성 피처 (국가-연도):
  leader_change_flag      1 if 그 해 지도자 교체 (이름 변화로 감지)
  election_flag           1 if 그 해 선거 발생 (election_now=1)
  irregular_change_flag   1 if 선거 없이 일어난 지도자 교체 (근사 = 비정상/쿠데타성)
  PolitShock              교체·선거 등급 (아래 규칙)

PolitShock 규칙 (설계 문서 근사):
    1.0  비정상(선거 없는) 지도자 교체     ← 쿠데타 등급 근사
    0.7  선거 발생                          ← '선거 분쟁' 근사
    0.3  (선거 동반) 지도자 교체
    0.0  변화 없음

파이프라인 위치:  normalize_and_group.py → step1e → step0(또는 직접 step2) → step3

사용법:
    python step1e_aggregate_reign.py <reign_grouped.csv> [--out 폴더]
출력:  reign.parquet  +  reign_aggregated.csv
  → step2 FEATURE_COLS 에 추가:
       "reign": ["PolitShock", "irregular_change_flag"],
──────────────────────────────────────────────────────────────────────────────
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import argparse
import os


def aggregate(input_path: str, output_dir: str, proxy_irregular: bool = False):
    print(f"\n📂 로딩: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"   행 수: {len(df):,}  /  컬럼: {list(df.columns)}")

    need = ["country_std", "year", "month", "leader"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        print(f"❌ 필수 컬럼 없음: {miss}")
        return

    df["year"]  = pd.to_numeric(df["year"],  errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "month"])
    df["year"] = df["year"].astype(int); df["month"] = df["month"].astype(int)
    df["election_now"] = (pd.to_numeric(df.get("election_now", 0),
                                        errors="coerce").fillna(0) > 0).astype(int)
    df["anticipation"] = (pd.to_numeric(df.get("anticipation", 0),
                                        errors="coerce").fillna(0) > 0).astype(int)

    # ── 지도자 교체 감지 (국가 내 월 순서로 이름 변화) ────────────────────────
    df = df.sort_values(["country_std", "year", "month"]).reset_index(drop=True)
    df["leader_prev"] = df.groupby("country_std")["leader"].shift(1)
    df["leader_change"] = (
        (df["leader"] != df["leader_prev"]) & df["leader_prev"].notna()
    ).astype(int)
    # 교체가 일어난 월에 선거(또는 선거 예고)가 동반됐는지
    df["change_with_election"] = (
        (df["leader_change"] == 1) &
        ((df["election_now"] == 1) | (df["anticipation"] == 1))
    ).astype(int)
    # 비정상(선거 없이) 교체 근사
    df["irregular_change"] = (
        (df["leader_change"] == 1) & (df["change_with_election"] == 0)
    ).astype(int)

    print(f"   감지: 지도자교체 {int(df['leader_change'].sum()):,}건 "
          f"(선거동반 {int(df['change_with_election'].sum()):,} / "
          f"비정상근사 {int(df['irregular_change'].sum()):,}) | "
          f"선거 {int(df['election_now'].sum()):,}건")

    # ── 국가-연도 집계 ───────────────────────────────────────────────────────
    g = (df.groupby(["country_std", "year"], as_index=False)
           .agg(leader_change_flag=("leader_change", "max"),
                election_flag=("election_now", "max"),
                irregular_change_flag=("irregular_change", "max"),
                group=("group", "first")))

    # ── PolitShock 등급 ──────────────────────────────────────────────────────
    # 기본: 신뢰 가능한 신호만 사용 (선거 0.7 / 지도자교체 0.3). 쿠데타 1.0 등급은
    # 이 파일로 신뢰성 있게 판별 불가하므로 제외.
    ps = np.zeros(len(g))
    ps = np.where(g["leader_change_flag"] == 1, 0.3, ps)
    ps = np.where(g["election_flag"] == 1,      np.maximum(ps, 0.7), ps)
    if proxy_irregular:
        # opt-in: '선거 없는 교체'를 쿠데타성(1.0)으로 근사 — 안정 민주국에서 과대계상 주의
        ps = np.where(g["irregular_change_flag"] == 1, 1.0, ps)
        print("   ⚠️  --proxy-irregular 활성: 비정상교체 근사를 PolitShock=1.0 으로 반영(노이즈 큼)")
    else:
        # 기본은 비정상 플래그를 0 으로 둔다(신뢰 불가). 컬럼은 유지하되 값 비움.
        g["irregular_change_flag"] = 0
        print("   ℹ️  irregular_change_flag = 0 (이 파일로 쿠데타/비정상교체 판별 불가)")
    g["PolitShock"] = ps

    g = g[["year", "country_std", "group", "PolitShock",
           "irregular_change_flag", "leader_change_flag", "election_flag"]]
    g = g.sort_values(["country_std", "year"])

    print(f"\n✅ 집계 완료: {len(g):,}행 (국가+연도)")
    print(f"   국가 수: {g['country_std'].nunique()} / 연도: {g['year'].min()}~{g['year'].max()}")
    print("\n📊 PolitShock 분포")
    print(g["PolitShock"].value_counts().sort_index().to_string())
    print("\n📊 그룹별 비정상교체(근사) 합계")
    print(g.groupby("group")[["irregular_change_flag", "election_flag",
                              "leader_change_flag"]].sum().reset_index().to_string(index=False))

    os.makedirs(output_dir, exist_ok=True)
    out_pq  = os.path.join(output_dir, "reign.parquet")
    out_csv = os.path.join(output_dir, "reign_aggregated.csv")
    pq.write_table(pa.Table.from_pandas(g), out_pq, compression="snappy")
    g.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 저장: {out_pq} ({os.path.getsize(out_pq)/1024:.1f} KB)")
    print('\n📌 step2 FEATURE_COLS 에 추가: "reign": ["PolitShock", "irregular_change_flag"],')
    return out_pq


def main():
    p = argparse.ArgumentParser(description="REIGN 월간 → 국가-연도 정치충격 집계")
    p.add_argument("file", nargs="?", help="REIGN (grouped) CSV")
    p.add_argument("--proxy-irregular", action="store_true",
                   help="선거없는 지도자교체를 쿠데타성(PolitShock=1.0)으로 근사 (노이즈 큼, 기본 off)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    fp = args.file
    if not fp:
        print("REIGN grouped CSV 경로:")
        fp = input("  >> ").strip().strip('"').strip("'")
    if not os.path.exists(fp):
        print(f"❌ 파일 없음: {fp}"); return
    out = args.out or os.path.dirname(os.path.abspath(fp))
    aggregate(fp, out, proxy_irregular=args.proxy_irregular)


if __name__ == "__main__":
    main()
