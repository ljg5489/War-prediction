import pandas as pd
import os

def preprocess_acled_data(input_file, output_file):
    print("데이터 로딩 중...")
    # 1. 데이터 로드
    df = pd.read_csv(input_file)
    print(f"초기 데이터 크기: {df.shape}")

    # 2. 시계열 필터링 (2013 ~ 2024년)
    # WEEK 컬럼을 datetime 객체로 변환하여 연도 추출
    df['WEEK'] = pd.to_datetime(df['WEEK'])
    df_filtered = df[(df['WEEK'].dt.year >= 2013) & (df['WEEK'].dt.year <= 2024)].copy()
    print(f"2013~2024년 시계열 필터링 완료: {df_filtered.shape}")

    # 3. 전처리 전 결측치 확인 (명세서 내용 재현)
    print("\n========================================")
    print(" ACLED 결측치 통계 (전처리 전)")
    print("========================================")
    missing_before = df_filtered.isnull().sum()
    if missing_before['POPULATION_EXPOSURE'] > 0:
         print(f"[POPULATION_EXPOSURE] 전체 결측치 개수: {missing_before['POPULATION_EXPOSURE']}")
    
    # 4. 핵심 전처리: POPULATION_EXPOSURE 변수 영구 배제
    # 행(dropna)을 삭제하지 않고 열(drop)만 삭제하여 타 변수(Fatalities 등) 보존
    if 'POPULATION_EXPOSURE' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['POPULATION_EXPOSURE'])
        print("\n=> POPULATION_EXPOSURE 변수를 분석 대상에서 영구 배제 완료.")

    # 5. 최종 결측치 검증
    print("\n========================================")
    print(" ACLED 결측치 통계 (전처리 후)")
    print("========================================")
    missing_after = df_filtered.isnull().sum()
    
    if missing_after.sum() == 0:
        print(" -> 발견된 결측치 없음. (결측률 0% 완전성 확보)")
    else:
        print(" -> 남은 결측치 내역:\n", missing_after[missing_after > 0])

    # 6. 정제된 데이터를 최종 파일로 저장
    df_filtered.to_csv(output_file, index=False)
    print(f"\n최종 정제된 데이터가 저장되었습니다: {output_file}")

# 실행부
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 실제 환경에 맞게 파일 경로를 수정해주세요.
    INPUT_FILE_PATH = os.path.join(base_dir, 'ACLED_All_Countries_.csv')  
    OUTPUT_FILE_PATH = os.path.join(base_dir, 'ACLED_cleaned_2013_2024_All_Countries.csv')
    
    # 함수 실행
    preprocess_acled_data(INPUT_FILE_PATH, OUTPUT_FILE_PATH)