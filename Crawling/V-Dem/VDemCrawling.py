import pandas as pd

# V-Dem 원본 파일명 (다운로드 받으신 파일명에 맞게 수정하세요)
VDEM_FILE = "VDemRawData.csv" 
OUTPUT_FILE = "VDemData_All_Countries.csv"

# 1. ⭐️ 하경 님의 요청으로 최종 확장된 핵심 변수 13개 리스트 (유지)
COLUMNS_TO_KEEP = [
    # 기본 식별자
    'country_name', 'country_text_id', 'year', 
    # 기존 타겟 변수
    'v2elpeace', 'v2x_rule', 'v2x_clphy', 'e_pt_coup', 'e_civil_war',
    # ⭐️ 필수 추가/유지 변수 5종
    'v2x_libdem', 'v2x_corr', 'v2x_veracc', 'v2xcs_ccsi', 'v2x_polyarchy'
]

def process_vdem_data():
    print("⏳ V-Dem 거대 원본 데이터 처리 중... (전체 국가 데이터 추출 중, 잠시만 기다려주세요)")
    
    # low_memory=False로 메모리 경고 방지
    df = pd.read_csv(VDEM_FILE, low_memory=False)
    
    # 2. ⭐️ 수정 포인트: 특정 국가 필터링 제거! 2013년 이후의 '모든 국가' 데이터만 남깁니다.
    filtered_df = df[df['year'] >= 2013]
    
    # 3. 확정된 13개 변수만 잘라내기
    final_df = filtered_df[COLUMNS_TO_KEEP]
    
    # 4. 연도(year) 오름차순 먼저, 그 다음 국가(country_text_id) 알파벳순 정렬
    final_df = final_df.sort_values(by=['year', 'country_text_id'])
    
    # 5. 결과 저장
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ V-Dem 전체 국가 추출 완벽하게 끝났습니다! 총 {len(final_df):,}행의 데이터가 '{OUTPUT_FILE}'로 저장되었습니다.")

if __name__ == "__main__":
    process_vdem_data()
