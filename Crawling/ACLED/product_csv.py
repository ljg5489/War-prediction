import pandas as pd
import os

# 1. 파일 경로 설정
master_file = "master_conflict_data.csv"
new_data_file = "Germany_data_2026_formming.csv"

# 2. 파일 존재 여부 확인 후 병합
if os.path.exists(master_file) and os.path.exists(new_data_file):
    # 신규 데이터 불러오기
    df_new = pd.read_csv(new_data_file)
    
    # 기존 마스터 파일에 이어 쓰기 (header=False로 컬럼명이 중복되지 않게 함)
    df_new.to_csv(master_file, mode='a', index=False, header=False)
    
    print(f"✅ 병합 완료! '{new_data_file}'의 데이터가 '{master_file}'에 성공적으로 추가되었습니다.")
    
    # 3. 전체 데이터 확인 (검증)
    df_master = pd.read_csv(master_file)
    print(f"\n[마스터 파일 최종 리포트]")
    print(f"- 전체 데이터 행 수: {len(df_master)}건")
    print(f"- 수집 기간: {df_master['year'].min()}년 ~ {df_master['year'].max()}년")
    print(f"- 포함된 국가: {df_master['country'].unique()}")
else:
    print("❌ 파일을 찾을 수 없습니다. 파일 이름을 확인해 주세요.")
