import pandas as pd
import os
import time

def convert_vdem_to_parquet():
    csv_file = "VDemData_grouped.csv"
    parquet_file = "VDemData_pa.parquet"
    
    print(f"⏳ '{csv_file}' 파일을 불러오는 중...")
    
    # 1. CSV 파일 읽기 (시간 측정 시작)
    start_time = time.time()
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ 에러: '{csv_file}' 파일을 찾을 수 없습니다.")
        return
    csv_load_time = time.time() - start_time
    
    # 2. Parquet 파일로 저장
    print(f"📦 데이터를 Parquet 형식으로 압축 저장하는 중...")
    df.to_parquet(parquet_file, engine='pyarrow', index=False)
    
    # 3. Parquet 파일 읽기 (속도 비교용 측정)
    start_time = time.time()
    df_parquet = pd.read_parquet(parquet_file, engine='pyarrow')
    parquet_load_time = time.time() - start_time
    
    # 4. 파일 크기 비교
    csv_size = os.path.getsize(csv_file) / (1024 * 1024) # MB로 변환
    parquet_size = os.path.getsize(parquet_file) / (1024 * 1024)
    
    # 5. 결과 출력
    print("\n" + "="*50)
    print("✅ V-Dem 데이터 Parquet 변환 완료!")
    print("="*50)
    print(f"📊 [파일 크기 비교]")
    print(f" - CSV 용량     : {csv_size:.2f} MB")
    print(f" - Parquet 용량 : {parquet_size:.2f} MB (약 {csv_size/parquet_size:.1f}배 압축!)")
    print(f"\n⚡ [읽기 속도 비교]")
    print(f" - CSV 로드     : {csv_load_time:.4f} 초")
    print(f" - Parquet 로드 : {parquet_load_time:.4f} 초")
    print("="*50)
    print(f"💡 앞으로 Streamlit에서는 '{parquet_file}'을 사용하시면 훨씬 빠릅니다!")

if __name__ == "__main__":
    convert_vdem_to_parquet()
