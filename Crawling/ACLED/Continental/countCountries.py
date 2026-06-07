# 파일 위치: C:\Users\hk100\Desktop\War-prediction\Crawling\ACLED\Continental\
# 파일 명: countCountries.py

import pandas as pd
import os

# ── 타겟 30개국 그룹 매핑 사전 ──
GROUP_MAPPING = {
    # Group A (고위험군)
    "Syria": "A", "Yemen": "A", "Somalia": "A", "Myanmar": "A",
    "Ethiopia": "A", "South Sudan": "A", "Mali": "A", 
    "Democratic Republic of Congo": "A", "DR Congo": "A", # ACLED 표기 고려
    "Ukraine": "A", "Iraq": "A",

    # Group B (중위험군)
    "Pakistan": "B", "Nigeria": "B", "Venezuela": "B", "Sudan": "B",
    "Central African Republic": "B", "Taiwan": "B", "Haiti": "B",
    "Lebanon": "B", "Colombia": "B", "Ecuador": "B",

    # Group C (저위험군)
    "Norway": "C", "Switzerland": "C", "Japan": "C", "South Korea": "C",
    "Portugal": "C", "Uruguay": "C", "Botswana": "C", "Mongolia": "C",
    "Canada": "C", "Germany": "C"
}

def count_acled_countries(input_file):
    print("데이터 로딩 중...")
    
    # 1. 파일 존재 여부 확인 및 로드
    if not os.path.exists(input_file):
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
        return
        
    # ACLED 데이터는 'COUNTRY' 컬럼만 읽어옵니다.
    df = pd.read_csv(input_file, usecols=['COUNTRY'])
    print(f"📌 총 스캔된 ACLED 데이터: {len(df):,} 행\n")

    # 2. 결측치 안전 처리 및 그룹 동적 매핑
    df['COUNTRY'] = df['COUNTRY'].fillna('UNKNOWN')
    
    # map 함수를 사용하여 사전(GROUP_MAPPING)에 있으면 A,B,C를, 없으면 'OTHER'를 부여합니다.
    df['group'] = df['COUNTRY'].map(GROUP_MAPPING).fillna('OTHER')

    # 3. 결과 출력
    print("========================================")
    print(" ACLED 그룹별 인식된 국가 목록 및 데이터 건수")
    print("========================================")
    
    # 출력할 그룹 순서 지정
    group_order = ['A', 'B', 'C', 'OTHER', 'UNKNOWN']
    
    for g in group_order:
        # 해당 그룹의 데이터만 필터링
        group_df = df[df['group'] == g]
        
        if not group_df.empty:
            print(f"\n[{g} Group]")
            
            # 국가별로 개수를 세고 내림차순 정렬
            country_counts = group_df['COUNTRY'].value_counts()
            
            group_total = 0
            for country, count in country_counts.items():
                print(f"    - {country} : {count:,} 행")
                group_total += count
                
            print("    ------------------------------------")
            print(f"    >> {g} 그룹 소계: {group_total:,} 행")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 🎯 읽어올 ACLED 파일 이름 (필요 시 알맞게 수정)
    INPUT_FILE_PATH = os.path.join(base_dir, 'ACLED_cleaned_2013_2024_All_Countries.csv')
    
    count_acled_countries(INPUT_FILE_PATH)