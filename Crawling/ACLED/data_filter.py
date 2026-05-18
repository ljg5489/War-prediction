import pandas as pd

# 1. 파일 불러오기
#df = pd.read_excel("Asia-Pacific_aggregated_data_up_to_week_of-2026-04-25.xlsx")
#df = pd.read_excel("Africa_aggregated_data_up_to_week_of-2026-04-25.xlsx")
#df = pd.read_excel("Latin-America-the-Caribbean_aggregated_data_up_to_week_of-2026-04-25.xlsx")
df = pd.read_excel("Europe-Central-Asia_aggregated_data_up_to_week_of-2026-04-25.xlsx")
#df = pd.read_excel("US-and-Canada_aggregated_data_up_to_week_of-2026-04-25.xlsx")
#df = pd.read_excel("Middle-East_aggregated_data_up_to_week_of-2026-04-25.xlsx")

# 2. 'WEEK' 열의 텍스트("28-February-2026")를 진짜 날짜 데이터(datetime)로 변환
df['WEEK'] = pd.to_datetime(df['WEEK'])

# 3. 날짜로 변환된 'WEEK' 열에서 연도(.dt.year)만 추출하여 'year'라는 새로운 열 만들기!
df['year'] = df['WEEK'].dt.year

# 4. 이제 'year' 열이 생겼으므로, 원래 하려고 했던 2026년 필터링이 가능해집니다.
filtered_df = df[df['year'] == 2026]

# 4-1. 핵심 수정: 2026년이면서(AND) 국가가 'Somalia'인 데이터만 필터링
filtered_df = df[(df['year'] == 2026) & (df['COUNTRY'] == 'Germany')]

# 5. 결과 저장하기
filtered_df.to_csv("Germany_data_2026.csv", index=False)

print(f"추출 완료! 총 {len(filtered_df)}건의 데이터가 저장되었습니다.")
