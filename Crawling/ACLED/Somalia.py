import requests
import pandas as pd
import time
import os

# 1. 환경 및 설정 변수
EMAIL = "hk100456@sunmoon.ac.kr"
PASSWORD = "Asdf1020300!"

TARGET_COUNTRY = "Somalia"
START_YEAR = 2006  # 기획안(PDF) 기준 반영 (예멘 2014~현재)
END_YEAR = 2026
CSV_FILENAME = "master_conflict_data.csv" # 모든 국가가 공유할 마스터 파일

# 유지할 필수 컬럼 리스트 (10개 핵심 변수 정확히 일치)
COLUMNS_TO_KEEP = [
    'event_id_cnty', 'event_date', 'year', 'event_type', 
    'sub_event_type', 'interaction', 'fatalities', 
    'latitude', 'longitude', 'country'
]

# 2. 토큰 발급 함수
def get_access_token(username, password):
    url = "https://acleddata.com/oauth/token"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'username': username,
        'password': password,
        'grant_type': "password",
        'client_id': "acled",
        'scope': "authenticated"
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()['access_token']
    else:
        raise Exception(f"토큰 발급 실패: {response.text}")

# 3. 데이터 수집 메인 로직
def fetch_country_data():
    print(f"🚀 [{TARGET_COUNTRY}] 데이터 크롤링 시작 (기간: {START_YEAR} ~ {END_YEAR})")
    my_token = get_access_token(EMAIL, PASSWORD)
    
    total_events_collected = 0
    
    # 연도별 반복문
    for year in range(START_YEAR, END_YEAR + 1):
        year_data = []
        page = 1
        
        while True:
            # ⭐️ 핵심 수정 1: API 1년 치 제한 검열을 피하기 위해 year 파라미터로 복구
            parameters = {
                "country": TARGET_COUNTRY,
                "year": year,
                "limit": 5000,
                "page": page
            }
            
            response = requests.get(
                "https://acleddata.com/api/acled/read?_format=json",
                params=parameters,
                headers={"Authorization": f"Bearer {my_token}", "Content-Type": "application/json"},
            )
            
            if response.status_code == 200:
                data = response.json().get("data", [])
                
                if not data:
                    if page == 1: # 1페이지부터 데이터가 아예 없으면 이상한 것이므로 사유 출력
                        print(f"  [디버그] {year}년 데이터 0건. 서버 응답 메시지: {response.json()}")
                    break
                
                # ⭐️ 핵심 수정 2: 받아온 데이터를 year_data에 누적시키고 페이지를 넘기는 로직 복구
                year_data.extend(data)
                print(f"  └ {year}년 - {page}페이지 수집 완료 ({len(data)}건)")
                
                # 받아온 데이터가 5000건 미만이면 해당 연도의 마지막 페이지라는 뜻이므로 루프 탈출
                if len(data) < 5000:
                    break
                
                page += 1     # 다음 페이지 번호로 올림
                time.sleep(1) # 서버 과부하 방지용 1초 대기
                
            else:
                print(f"❌ 데이터 요청 에러 ({year}년): {response.status_code}")
                break
                
        # 해당 연도의 모든 페이지 수집이 끝나면 CSV로 묶어서 저장
        if year_data:
            df = pd.DataFrame(year_data)
            df_core = df[COLUMNS_TO_KEEP].copy()
            
            # 데이터 정제
            df_core['event_date'] = pd.to_datetime(df_core['event_date'])
            df_core['fatalities'] = pd.to_numeric(df_core['fatalities'])
            
            # mode='a'를 통해 마스터 CSV 파일에 데이터 이어 쓰기 (Append)
            file_exists = os.path.isfile(CSV_FILENAME)
            df_core.to_csv(CSV_FILENAME, mode='a', index=False, header=not file_exists)
            
            total_events_collected += len(df_core)
            print(f"✅ {year}년 데이터 {len(df_core)}건 누적 저장 완료!\n")
            
    print(f"🎉 [{TARGET_COUNTRY}] 크롤링 최종 완료! 총 {total_events_collected}건 누적됨.")

if __name__ == "__main__":
    fetch_country_data()
