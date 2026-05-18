import pandas as pd
import os
import time

def convert_acled_to_parquet():
    csv_file = "UCDP_GED_filtered_grouped.csv"
    parquet_file = "UCDP_GED_filtered_pa.parquet"
    
    print(f"⏳ '{csv_file}' 파일을 불러오는 중...")
    
    # 1. CSV 파일 읽기 (시간 측정 시작)
    # ACLED는 수집 기간에 따라 행이 많을 수 있으므로 low_memory=False를 사용합니다.
    start_time = time.time()
    try:
        df = pd.read_csv(csv_file, low_memory=False)
    except FileNotFoundError:
        print(f"❌ 에러: '{csv_file}' 파일을 찾을 수 없습니다. 파일명을 확인해 주세요.")
        return
    csv_load_time = time.time() - start_time
    
    # 2. Parquet 파일로 저장
    print(f"📦 ACLED 충돌 데이터를 Parquet 형식으로 압축 저장하는 중...")
    # ACLED 데이터는 인덱스 없이 저장하는 것이 깔끔합니다.
    df.to_parquet(parquet_file, engine='pyarrow', index=False)
    
    # 3. Parquet 파일 읽기 (속도 비교용 측정)
    start_time = time.time()
    df_parquet = pd.read_parquet(parquet_file, engine='pyarrow')
    parquet_load_time = time.time() - start_time
    
    # 4. 파일 크기 비교
    csv_size = os.path.getsize(csv_file) / (1024 * 1024) # MB 단위
    parquet_size = os.path.getsize(parquet_file) / (1024 * 1024)
    
    # 5. 결과 출력
    print("\n" + "="*50)
    print("✅ ACLED 데이터 Parquet 변환 완료!")
    print("="*50)
    print(f"📊 [파일 크기 비교]")
    print(f" - CSV 용량     : {csv_size:.2f} MB")
    print(f" - Parquet 용량 : {parquet_size:.2f} MB (약 {csv_size/parquet_size:.1f}배 압축!)")
    print(f"\n⚡ [읽기 속도 비교]")
    print(f" - CSV 로드     : {csv_load_time:.4f} 초")
    print(f" - Parquet 로드 : {parquet_load_time:.4f} 초")
    print("="*50)
    print(f"💡 이제 모든 데이터(V-Dem, GDELT, ACLED)가 Parquet으로 준비되었습니다.")

if __name__ == "__main__":
    convert_acled_to_parquet()
