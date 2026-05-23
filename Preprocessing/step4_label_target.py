"""
step4_label_target.py
──────────────────────────────────────────────────────────────────────────────
타겟 변수 Y 레이블 생성 (전 세계)

θ_c = Percentile25(WRI_c,t | 전쟁 확인 기간)
Y_c,t = 1  if WRI_c,t >= θ_c
        0  otherwise

전쟁 확인 기간 정의:
  - A/B/OTHER 그룹: 연간 사망자 > WAR_DEATH_THRESHOLD(25명)인 연도
  - C 그룹 (평화국): 항상 0 → θ_c = WRI 75번째 백분위수

사용법:
    python step4_label_target.py <wri.csv>
    python step4_label_target.py   (인자 없으면 경로 입력 프롬프트)

입력: wri.csv (step3 출력)
출력: wri_labeled.csv
──────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import argparse
import os

# ── 설정 ─────────────────────────────────────────────────────────────────────
WAR_DEATH_THRESHOLD      = 25   # UCDP 전쟁 정의 최소 기준 (연 25명 이상)
PEACE_COUNTRY_PERCENTILE = 75   # C그룹: 75번째 백분위수 이상이면 이상 징후


def define_war_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 행에 is_war_year(0/1) 플래그 부여.
    - A/B/OTHER: 연간 사망자 > WAR_DEATH_THRESHOLD
    - C: 항상 0 (전쟁 없음)
    """
    df = df.copy()

    if "total_deaths" in df.columns:
        annual_deaths = (
            df.groupby(["country_std", "year"])["total_deaths"]
            .sum()
            .reset_index()
            .rename(columns={"total_deaths": "annual_deaths"})
        )
        df = df.merge(annual_deaths, on=["country_std", "year"], how="left")
        df["is_war_year"] = (
            (df["group"] != "C") &                              # C그룹 제외
            (df["annual_deaths"].fillna(0) > WAR_DEATH_THRESHOLD)
        ).astype(int)
    else:
        print("⚠️  total_deaths 컬럼 없음 → A/B/OTHER 전 기간을 전쟁 기간으로 처리")
        df["is_war_year"] = (df["group"] != "C").astype(int)

    war_counts = df.groupby("country_std")["is_war_year"].sum()
    print("📋 국가별 전쟁 확인 연도 수 (상위 30개):")
    print(war_counts[war_counts > 0].sort_values(ascending=False).head(30).to_string())
    return df


def compute_theta(df: pd.DataFrame) -> pd.Series:
    """
    국가별 임계값 θ_c 계산.
    θ_c = Percentile25(WRI | is_war_year == 1)
    전쟁 기간 없으면 WRI 75번째 백분위수 사용.
    """
    def threshold_for_country(sub):
        war_wri = sub.loc[sub["is_war_year"] == 1, "WRI"]
        if len(war_wri) >= 4:
            return war_wri.quantile(0.25)
        elif len(war_wri) > 0:
            return war_wri.median()
        else:
            return sub["WRI"].quantile(PEACE_COUNTRY_PERCENTILE / 100)

    theta = df.groupby("country_std").apply(threshold_for_country)
    print("\n📋 국가별 θ_c 요약 (그룹별 상위 5개)")
    for grp in sorted(df["group"].unique()):
        countries = df[df["group"] == grp]["country_std"].unique()
        subset    = theta[theta.index.isin(countries)].sort_values(ascending=False).head(5)
        print(f"\n  [{grp}]")
        print(subset.round(4).to_string())
    return theta


def assign_label(df: pd.DataFrame, theta: pd.Series) -> pd.DataFrame:
    """
    θ_c 기준으로 Y 레이블 부여.
    Y = 1: 전쟁 발생 위험 HIGH
    Y = 0: 위험 LOW
    """
    df = df.copy()
    df["theta_c"] = df["country_std"].map(theta)
    df["Y"]       = (df["WRI"] >= df["theta_c"]).astype(int)

    print("\n📊 레이블 분포 (전체)")
    dist = df["Y"].value_counts(normalize=True).round(3)
    print(f"   Y=0 (저위험): {dist.get(0, 0):.1%}")
    print(f"   Y=1 (고위험): {dist.get(1, 0):.1%}")

    print("\n📊 그룹별 Y=1 비율")
    print(df.groupby("group")["Y"].mean().round(3).rename("Y=1 비율").to_string())

    return df


def main():
    parser = argparse.ArgumentParser(
        description="WRI CSV에 타겟 변수 Y 레이블 추가"
    )
    parser.add_argument("file", nargs="?", help="입력 wri.csv 파일 경로")
    args = parser.parse_args()

    input_path = args.file
    if not input_path:
        print("wri.csv 파일 경로를 입력하세요:")
        input_path = input("  >> ").strip().strip('"').strip("'")

    if not os.path.exists(input_path):
        print(f"❌ 파일 없음: {input_path}")
        return

    print(f"📂 로딩: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"   행 수: {len(df):,}  /  국가 수: {df['country_std'].nunique()}")

    print("\n── STEP 1: 전쟁 확인 기간 정의 ─────────────────────────────")
    df = define_war_years(df)

    print("\n── STEP 2: 국가별 θ_c 계산 ──────────────────────────────────")
    theta = compute_theta(df)

    print("\n── STEP 3: Y 레이블 부여 ─────────────────────────────────────")
    df_out = assign_label(df, theta)

    out_path = input_path.replace(".csv", "_labeled.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 저장: {out_path}")
    print(f"   추가된 컬럼: theta_c, Y")
    print(f"   최종 행 수: {len(df_out):,}")


if __name__ == "__main__":
    main()
