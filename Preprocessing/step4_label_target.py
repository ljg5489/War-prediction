"""
step4_label_target.py (v5.5 - Smart Time-Unit & Absolute Target Edition)
──────────────────────────────────────────────────────────────────────────────
데이터의 시간 단위(연도 vs 주차)를 자동 감지하여, 
국가별 꼼수 임계값을 제거하고 '실제 미래의 절대적 분쟁 여부'를 타깃 Y로 생성합니다.

메커니즘:
  1. 시간 축이 'year'이면 -> 1~2년 뒤(t+1 또는 t+2)의 미래를 예측
  2. 시간 축이 'week_end' 등 주차 단위이면 -> 13주 뒤(t+13)의 미래를 예측
  3. 타깃 기준: 글로벌 절대 기준 (WRI 점수가 0보다 크거나 사망자가 존재하는 절대 위험 상태)
──────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import argparse
import os

# ── 글로벌 예측 설정 ────────────────────────────────────────────────────────
GLOBAL_WRI_THRESHOLD = -0.5  # WRI 기준 절대 임계값 (0.0보다 크면 리스크 상태로 정의)


def detect_and_set_shift(df: pd.DataFrame) -> tuple:
    """
    데이터셋의 시간 단위를 분석하여 최적의 정렬 기준 컬럼과 미래 시프트(Shift) 간격을 결정합니다.
    """
    df = df.copy()
    
    # 1. 시간 컬럼 감지
    if "week_end" in df.columns:
        time_col = "week_end"
        shift_size = 13  # 주차 단위 데이터 -> 13주 뒤 미래 예측
        unit_name = "주(Weeks)"
    elif "date" in df.columns:
        time_col = "date"
        shift_size = 13
        unit_name = "주(Weeks)"
    elif "year" in df.columns:
        time_col = "year"
        shift_size = 1   # 연도 단위 데이터 -> 1년 뒤 미래 예측 (t+1)
        unit_name = "년(Years)"
    else:
        raise KeyError("❌ 'year', 'week_end', 'date' 중 어떤 시간 축 컬럼도 찾을 수 없습니다.")
        
    print(f"⏰ [시간 축 감지] '{time_col}' 컬럼 발견 -> 데이터 단위: {unit_name}")
    print(f"🔁 [시프트 설정] 미래 예측 타깃 스텝 크기 = -{shift_size} ({unit_name} 뒤의 미래)")

    # 2. 국가 및 시간순 정렬
    if "country_std" in df.columns:
        df = df.sort_values(by=["country_std", time_col]).reset_index(drop=True)
    else:
        df = df.sort_values(by=[time_col]).reset_index(drop=True)
        
    return df, time_col, shift_size


def build_absolute_target(df: pd.DataFrame, time_col: str, shift_size: int) -> pd.DataFrame:
    df = df.copy()

    # [STEP 1] 현재 시점(t)의 실제 상태 정의
    if "is_active_conflict" not in df.columns:
        df["is_active_conflict"] = (df["WRI"] > GLOBAL_WRI_THRESHOLD).astype(int)

    # --- [여기에 Onset 로직 추가] ---
    # 1. 13주(또는 1년) 뒤의 미래 상태를 가져옴
    if "country_std" in df.columns:
        df["future_conflict"] = df.groupby("country_std")["is_active_conflict"].shift(-shift_size)
    else:
        df["future_conflict"] = df["is_active_conflict"].shift(-shift_size)

    # 2. 현재는 0(평화)인데 미래에는 1(분쟁)인 경우만 1로 표시 (이것이 Onset)
    df["Y"] = ((df["is_active_conflict"] == 0) & (df["future_conflict"] == 1)).astype(int)
    # ----------------------------

    # [STEP 3] 정제 및 마무리
    df = df.dropna(subset=["future_conflict"]).reset_index(drop=True)
    df = df.drop(columns=["future_conflict"]) # 중간 계산용 컬럼 제거
    
    # ... 이후 진단 출력 로직은 그대로 유지 ...
    return df


def main():
    parser = argparse.ArgumentParser(description="WRI 데이터셋에 절대적 미래 타깃 Y 주입")
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
    print(f"   최초 행 수: {len(df):,}  /  국가 수: {df['country_std'].nunique()}")

    print("\n── STEP 1: 시간 단위 분석 및 시계열 정렬 ────────────────────────")
    df, time_col, shift_size = detect_and_set_shift(df)

    print("\n── STEP 2: 절대 미래 지표 타깃(Y) 빌드 ──────────────────────────")
    df_out = build_absolute_target(df, time_col, shift_size)

    out_path = input_path.replace(".csv", "_labeled.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 타겟 주입 완료 및 저장 성공: {out_path}")
    print("\n--- [데이터 실체 확인] ---")
    print(df_out['Y'].value_counts())
    print(f"전체 샘플 대비 1의 비율: {df_out['Y'].mean():.2%}")


if __name__ == "__main__":
    main()

