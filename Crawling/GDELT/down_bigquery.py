import os
from google.cloud import bigquery
import pandas as pd

# ---------------------------------------------------------
# 🔑 1. 방금 발급받은 완벽한 출입증 경로
# ---------------------------------------------------------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\hk100\AppData\Roaming\gcloud\application_default_credentials.json"

# ---------------------------------------------------------
# 🚀 2. 구글 클라우드 접속
# ---------------------------------------------------------
print("🔄 구글 클라우드에 접속 중...")
client = bigquery.Client(project="warprediction")
print("✅ 구글 클라우드 접속 완벽 성공!")

# ---------------------------------------------------------
# 📦 3. 데이터 다운로드 및 저장
# ---------------------------------------------------------
# BigQuery 웹에서 저장하셨던 결과 테이블의 주소를 적어주세요.
# (만약 데이터세트 이름이나 테이블 이름이 다르다면 꼭 수정해주세요!)
table_id = "warprediction.gdelt_processed_data.gdelt_daily_all"

print("\n⏳ 94GB에 달하는 구글 서버 데이터를 내 컴퓨터로 가져오는 중입니다...")
print("(데이터 크기와 인터넷 속도에 따라 1~5분 정도 소요될 수 있습니다)")

# 쿼리(계산) 없이 테이블 데이터만 쏙 빼오기
df = client.list_rows(table_id).to_dataframe()

print(f"\n📦 다운로드 완료! 총 {len(df):,}행의 데이터를 가져왔습니다.")
print("💾 로컬 CSV 파일로 쓰는 중...")

# 내 컴퓨터 하드디스크에 CSV 형식으로 저장 (한글 깨짐 방지 utf-8-sig)
df.to_csv("GDELT_Daily_All_Countries.csv", index=False, encoding='utf-8-sig')

print("🎉 완벽하게 저장되었습니다! 폴더에서 'GDELT_Daily_All_Countries.csv' 파일을 확인하세요.")