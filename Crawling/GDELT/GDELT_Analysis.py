import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 1. Streamlit 전역 설정
# ==========================================
st.set_page_config(page_title="GDELT RAW Data Explorer", page_icon="🗞️", layout="wide")

# ==========================================
# 2. 시각화 디자인 설정 (한글 깨짐 방지 및 테마)
# ==========================================
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white", font='Malgun Gothic')

# 그룹별 컬러: A(Red), B(Orange), C(Teal)
my_palette = ["#C0392B", "#E67E22", "#16A085"]

# ==========================================
# 3. 데이터 로딩 (정형화 없이 로드)
# ==========================================
@st.cache_data
def load_gdelt_raw():
    # 데이터 로드 (결측치 제거 안 함)
    df = pd.read_csv("GDELT_data.csv")
    
    # 시각화 정렬을 위한 날짜 변환 (이 작업은 정형화가 아닌 인덱싱 용도)
    df['date_dt'] = pd.to_datetime(df['SQLDATE'].astype(str), format='%Y%m%d', errors='coerce')
    
    # README 기준 FIPS 국가 코드 매핑
    group_map = {
        # Group A
        'SY': 'A', 'YM': 'A', 'SO': 'A', 'BM': 'A', 'ET': 'A', 
        'OD': 'A', 'ML': 'A', 'CG': 'A', 'UP': 'A', 'IZ': 'A',
        # Group B
        'PK': 'B', 'NI': 'B', 'VE': 'B', 'SU': 'B', 'CT': 'B', 
        'TW': 'B', 'HA': 'B', 'LE': 'B', 'CO': 'B', 'EC': 'B',
        # Group C
        'NO': 'C', 'SZ': 'C', 'JA': 'C', 'KS': 'C', 'PO': 'C', 
        'UY': 'C', 'BC': 'C', 'MG': 'C', 'CA': 'C', 'GM': 'C'
    }
    df['group'] = df['ActionGeo_CountryCode'].map(group_map)
    return df

# 데이터 로드 실행
try:
    df_raw = load_gdelt_raw()
except FileNotFoundError:
    st.error("❌ 'GDELT_data.csv' 파일을 찾을 수 없습니다.")
    st.stop()

# ==========================================
# 4. 사이드바 설정 (인터랙티브 컨트롤)
# ==========================================
with st.sidebar:
    st.header("⚙️ RAW 데이터 탐색")
    st.info("💡 본 데이터는 정형화되지 않은 상태로, 이상치와 결측치가 그대로 포함되어 있습니다.")
    
    # 분석할 메인 변수 선택
    target_var = st.selectbox(
        "📊 분석 지표 선택", 
        ['AvgTone', 'AvgGoldstein', 'TotalMentions', 'TotalSources', 'TotalArticles']
    )
    
    # 기간 필터
    st.divider()
    all_dates = df_raw['date_dt'].dropna().sort_values()
    start_date, end_date = st.select_slider(
        "📅 분석 기간 선택",
        options=all_dates.unique(),
        value=(all_dates.min(), all_dates.max())
    )

# 필터링된 데이터 (날짜 기준만 필터링, 값 보정은 없음)
filtered_df = df_raw[(df_raw['date_dt'] >= start_date) & (df_raw['date_dt'] <= end_date)]

# ==========================================
# 5. 메인 대시보드 UI
# ==========================================
st.title("🗞️ GDELT 언론 심리 및 사건 파급력 분석 (RAW)")
st.markdown(f"**현재 지표:** {target_var} | **선택 기간:** {start_date.date()} ~ {end_date.date()}")

# 명세서 정보 아코디언
with st.expander("📖 GDELT 데이터셋 README(명세서) 확인"):
    st.markdown("""
    - **AvgGoldstein**: 사건의 이론적 영향력 (-10 ~ +10)
    - **AvgTone**: 뉴스 기사의 어조/감성 (-100 ~ +100)
    - **TotalMentions**: 뉴스 언급 횟수 (이슈의 화제성)
    - **TotalSources**: 보도 매체(출처) 수 (정보의 확산성)
    - **TotalArticles**: 발행 기사 총수 (보도 집중도)
    """)

st.divider()

# 상단 그래프 열 배치 (1:1 비율)
col1, col2 = st.columns(2)

with col1:
    # [그래프 1] 그룹별 데이터 분포 (Box + Strip Plot)
    st.subheader(f"1️⃣ 그룹별 {target_var} 분포")
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    sns.boxplot(x='group', y=target_var, data=filtered_df, order=['A', 'B', 'C'],
                palette=my_palette, width=0.5, ax=ax1, linewidth=2, 
                fliersize=0, hue='group', legend=False, boxprops=dict(alpha=0.4))
    sns.stripplot(x='group', y=target_var, data=filtered_df, order=['A', 'B', 'C'],
                  palette=my_palette, size=2, hue='group', ax=ax1,
                  jitter=0.3, alpha=0.4, legend=False)
    ax1.set_title(f"Raw {target_var} Distribution by Group", fontsize=15)
    sns.despine()
    st.pyplot(fig1)

with col2:
    # [그래프 2] 그룹별 일일 평균 추세 (Line Plot)
    st.subheader(f"2️⃣ {target_var} 시계열 평균 추세")
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    sns.lineplot(x='date_dt', y=target_var, hue='group', data=filtered_df, 
                 errorbar=None, palette=my_palette, linewidth=2.5)
    ax2.set_title(f"Daily Mean Trend of {target_var}", fontsize=15)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine()
    st.pyplot(fig2)

st.divider()

# 하단 그래프 열 배치
col3, col4 = st.columns([1.2, 0.8])

with col3:
    # [그래프 3] 보도 화제성 vs 확산성 (Scatter Plot)
    st.subheader("3️⃣ 보도 파급력 분석: 화제성 vs 확산성")
    st.caption("Mentions(언급수)와 Sources(매체수)의 관계를 통해 뉴스 폭발력을 확인합니다.")
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    # 정형화되지 않은 이상치가 그대로 점으로 나타납니다.
    sns.scatterplot(x='TotalSources', y='TotalMentions', hue='group', 
                    size='TotalArticles', sizes=(20, 400), alpha=0.4,
                    palette=my_palette, data=filtered_df, ax=ax3)
    ax3.set_title("Total Sources vs Total Mentions (Bubble: Articles)", fontsize=15)
    sns.despine()
    st.pyplot(fig3)

with col4:
    # [그래프 4] 변수 간 상관관계 (Heatmap)
    st.subheader("4️⃣ 지표 간 상관관계 (RAW)")
    fig4, ax4 = plt.subplots(figsize=(8, 7))
    num_cols = ['AvgGoldstein', 'AvgTone', 'TotalMentions', 'TotalSources', 'TotalArticles']
    corr_matrix = filtered_df[num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt=".2f", ax=ax4, cbar=False)
    ax4.set_title("Correlation Heatmap", fontsize=15)
    st.pyplot(fig4)
