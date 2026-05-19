import pandas as pd
import os

# 현재 실행 중인 파이썬 파일(merge.py)이 위치한 폴더의 절대 경로를 가져옵니다.
current_dir = os.path.dirname(os.path.abspath(__file__))

# 합칠 엑셀 파일(.xlsx) 이름 또는 전체 경로를 입력하세요.
file_names = [
    "Africa_aggregated_data_up_to_week_of-2026-04-25.xlsx",
    "Asia-Pacific_aggregated_data_up_to_week_of-2026-04-25.xlsx",  # 2번째 xlsx
    "Europe-Central-Asia_aggregated_data_up_to_week_of-2026-04-25.xlsx",  # 3번째 xlsx
    "Latin-America-the-Caribbean_aggregated_data_up_to_week_of-2026-04-25.xlsx",  # 4번째 xlsx
    "Middle-East_aggregated_data_up_to_week_of-2026-04-25.xlsx",  # 5번째 xlsx
    "US-and-Canada_aggregated_data_up_to_week_of-2026-04-25.xlsx"   # 6번째 xlsx
]

dataframes = []

for file in file_names:
    if file.strip():
        # 폴더 경로와 파일 이름을 합쳐서 완벽한 절대 경로를 생성합니다.
        file_path = os.path.join(current_dir, file)
        
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            print(f"❌ [Path Error] Could not find '{file}'.")
            print(f"   Searched at: {file_path}")
            continue
            
        print(f"Reading '{file}'...")
        df = pd.read_excel(file_path)
        dataframes.append(df)

# 병합 진행
if len(dataframes) > 0:
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # 결과물도 같은 폴더에 저장되도록 경로 지정
    output_path = os.path.join(current_dir, "ACLED_All_Countries.csv")
    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Merge complete! Saved as '{output_path}'.")
else:
    print("\n⚠️ No files to merge. Please check the file names.")