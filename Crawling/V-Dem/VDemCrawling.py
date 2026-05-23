import pandas as pd

# V-Dem 원본 파일명
VDEM_FILE = r"C:\Users\hk100\Desktop\War-prediction\Crawling\V-Dem\VDemRawData.csv"
OUTPUT_FILE = "VDemData_All_Countries.csv"

# 1. ⭐️ v2elintim(선거 폭력 지수)이 추가된 14개 핵심 변수 리스트
COLUMNS_TO_KEEP = [
    # 기본 식별자
    
    'country_name', 'country_text_id', 'year', 
    # 기존 타겟 및 정치 지표
    'v2elpeace', 'v2x_rule', 'v2x_clphy', 'e_pt_coup', 'e_civil_war',
    'v2x_libdem', 'v2x_corr', 'v2x_veracc', 'v2xcs_ccsi', 'v2x_polyarchy',
    # ⭐️ PDF 가이드 피처 그룹 5 필수 변수 추가
    'v2elintim' 
]

def process_vdem_data():
    print("⏳ V-Dem 거대 원본 데이터 처리 중... (선거 폭력 지수 포함 전체 국가 추출 중)")
    
    # 데이터 로드
    df = pd.read_csv(VDEM_FILE, low_memory=False)
    
    # 2013년 이후의 '모든 국가' 데이터 필터링
    filtered_df = df[df['year'] >= 2013]
    
    # 2. ⭐️ 변수가 존재하는지 확인 후 안전하게 추출
    # (원본 파일에 따라 대소문자가 다를 수 있으니 확인이 필요할 수 있습니다)
    existing_cols = [col for col in COLUMNS_TO_KEEP if col in df.columns]
    final_df = filtered_df[existing_cols]
    
    # 정렬
    final_df = final_df.sort_values(by=['year', 'country_text_id'])
    
    # 저장
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ 추출 완료! '{OUTPUT_FILE}'에 선거 폭력 지수(v2elintim)가 포함되었습니다.")
    print(f"총 데이터 행 수: {len(final_df):,}개")

if __name__ == "__main__":
    process_vdem_data()