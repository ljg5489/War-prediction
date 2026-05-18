import pandas as pd

# 1. 2026년 데이터 불러오기
df = pd.read_csv("Germany_data_2026.csv")

# 2. 열(Column) 이름 매핑 딕셔너리 생성 (엑셀 열 이름 -> API 열 이름)
rename_dict = {
    'WEEK': 'event_date',
    'EVENT_TYPE': 'event_type',
    'SUB_EVENT_TYPE': 'sub_event_type',
    'FATALITIES': 'fatalities',
    'CENTROID_LATITUDE': 'latitude',
    'CENTROID_LONGITUDE': 'longitude',
    'COUNTRY': 'country'
}

# 열 이름 싹 바꾸기
df = df.rename(columns=rename_dict)

# 3. 엑셀 데이터에 없는 빈 필수 열(Column) 채워 넣기
# (Aggregated 데이터라서 빠져있는 interaction이나 개별 사건 ID를 임의로 채워줍니다)
# ID 열이 중복될 수 있으므로 행 번호(index)를 붙여 고유 ID로 만듭니다.
df['event_id_cnty'] = "AGG_" + df['ID'].astype(str) + "_" + df.index.astype(str)
df['interaction'] = 'Unknown' # 집계 데이터라 행위자 상호작용 코드가 없으므로 보완

# 4. 우리가 마스터 파일에서 쓰기로 했던 정확한 10개 열 리스트
COLUMNS_TO_KEEP = [
    'event_id_cnty', 'event_date', 'year', 'event_type', 
    'sub_event_type', 'interaction', 'fatalities', 
    'latitude', 'longitude', 'country'
]

# 5. 순서에 맞게 열을 추출하여 새로운 데이터프레임 완성
final_df = df[COLUMNS_TO_KEEP]

# 6. 최종 파일로 저장 (master 파일에 바로 붙여도 되지만, 우선 확인용으로 따로 저장)
final_df.to_csv("Germany_data_2026_formming.csv", index=False)

print("✅ 마스터 파일 형식으로 변환 완료!")
print("\n[변환된 데이터 5줄 미리보기]")
print(final_df.head())
