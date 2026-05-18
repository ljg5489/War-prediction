import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Streamlit 전역 설정 (레이아웃 및 탭 이름)
# ==========================================
st.set_page_config(page_title="V-Dem 분쟁 예측 EDA", page_icon="🌍", layout="wide")

# ==========================================
# 2. 디자인 및 폰트 설정
# ==========================================
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Malgun Gothic')

# 커스텀 팔레트: Group A(Red), Group B(Orange), Group C(Teal)
my_palette = ["#C0392B", "#E67E22", "#16A085"]
sns.set_palette(sns.color_palette(my_palette))

# ==========================================
# 3. 데이터 로딩 (캐싱)
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("VDemData.csv")
    group_map = {
        'SYR':'A', 'YEM':'A', 'SOM':'A', 'MMR':'A', 'ETH':'A', 
        'SSD':'A', 'MLI':'A', 'COD':'A', 'UKR':'A', 'IRQ':'A',
        'PAK':'B', 'NGA':'B', 'VEN':'B', 'SDN':'B', 'CAF':'B', 
        'TWN':'B', 'HAT':'B', 'LBN':'B', 'COL':'B', 'ECU':'B',
        'NOR':'C', 'CHE':'C', 'JPN':'C', 'KOR':'C', 'PRT':'C', 
        'URY':'C', 'BWA':'C', 'MNG':'C', 'CAN':'C', 'DEU':'C'
    }
    df['group'] = df['country_text_id'].map(group_map)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ 'VDemData.csv' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    st.stop()

# ==========================================
# 4. 변수 사전 (README 내용 맵핑)
# ==========================================
var_dict = {
    'v2x_polyarchy': {'name': '선거 민주주의 지수', 'desc': '선거가 얼마나 자유롭고 공정하게 치러지는가를 측정. 지수가 급락하는 현상은 분쟁의 주요 전조 증상입니다.'},
    'v2x_libdem': {'name': '자유 민주주의 지수', 'desc': '선거를 넘어, 소수자 권리 보호와 권력 분립(행정부 견제)이 얼마나 잘 작동하는지 측정합니다.'},
    'v2x_rule': {'name': '법의 지배 지수', 'desc': '행정부가 법에 종속되는지, 사법부가 독립적인지 측정. 수치가 낮으면 국가 실패(State Failure)의 가능성이 큽니다.'},
    'v2x_corr': {'name': '정치 부패 지수', 'desc': '행정, 입법, 사법부 등 정부 전반의 부패 정도. 반군이나 폭동이 발생하는 가장 큰 사회적 명분입니다.'},
    'v2x_clphy': {'name': '신체적 폭력 금지 지수', 'desc': '정부가 자국민을 상대로 고문, 정치적 살인 등을 자행하지 않는 정도. 내전 직전 급락하는 핵심 지표입니다.'},
    'v2elpeace': {'name': '선거 평화 지수', 'desc': '선거 기간 중 비국가 행위자에 의한 폭력 사태가 없는 정도. 낮을수록 선거판이 폭력적임을 의미합니다.'},
    'v2x_veracc': {'name': '수직적 책임성 지수', 'desc': '국민이 합법적으로 정부를 통제할 수 있는 정도. 이 길이 막히면 폭력적 시위로 번질 확률이 높습니다.'},
    'v2xcs_ccsi': {'name': '핵심 시민사회 지수', 'desc': '시민사회 조직이 정부의 탄압 없이 자율적이고 활발하게 활동하는 정도를 측정합니다.'}
}

# ==========================================
# 5. 사이드바 UI (사용자 컨트롤)
# ==========================================
with st.sidebar:
    st.header("⚙️ 분석 설정")
    st.write("시각화할 데이터를 조작해보세요.")
    
    # 연도 슬라이더
    min_yr = int(df['year'].min())
    max_yr = int(df['year'].max())
    selected_years = st.slider("📅 분석 연도 범위", min_value=min_yr, max_value=max_yr, value=(1990, max_yr))
    
    # 시계열 차트용 변수 선택
    st.divider()
    selected_var = st.selectbox(
        "📈 시계열 차트 지표 선택", 
        options=list(var_dict.keys()), 
        format_func=lambda x: f"{var_dict[x]['name']} ({x})"
    )

# ==========================================
# 6. 메인 화면 UI
# ==========================================
st.title("🌍 글로벌 국가 그룹별 정치·제도적 구조 분석")
st.markdown("**Group A**: 활성 분쟁 (Red) | **Group B**: 고위험 잠재 (Orange) | **Group C**: 안정적 대조군 (Teal)")

# 설명서 아코디언 (접기/펼치기)
with st.expander("📖 데이터 명세서 및 지표 설명 보기", expanded=False):
    st.markdown("""
    **[데이터 출처]** V-Dem (Varieties of Democracy) v14  
    **[목적]** 분쟁 예측 AI 모델의 기저 질환(구조적, 정치적, 제도적 취약성) 패턴 학습  
    
    * **모델 타겟 변수:**
      * `e_civil_war`: 내전 발생 여부 (정답지 Label로 활용)
      * `e_pt_coup`: 군사 쿠데타 발생 여부 (강력한 트리거)
    """)
    # 딕셔너리에 있는 모든 변수 설명 출력
    for k, v in var_dict.items():
        st.write(f"- **{v['name']} (`{k}`)**: {v['desc']}")

st.divider()

# 필터링된 데이터
filtered_df = df[(df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])]

# 좌/우 화면 분할
col1, col2 = st.columns([1.1, 0.9], gap="large")

# --- [좌측: 박스플롯 (분포 요약)] ---
with col1:
    st.subheader("📊 주요 4대 지표 그룹별 분포 비교")
    st.caption(f"기준: {selected_years[0]}년 ~ {selected_years[1]}년 통합")
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
    box_features = ['v2x_polyarchy', 'v2x_corr', 'v2x_rule', 'v2x_clphy']
    
    for i, feat in enumerate(box_features):
        row, col = divmod(i, 2)
        ax = axes1[row, col]
        
        sns.boxplot(x='group', y=feat, data=filtered_df, order=['A', 'B', 'C'],
                    palette=my_palette, width=0.5, ax=ax, linewidth=1.5, 
                    fliersize=0, hue='group', legend=False, boxprops=dict(alpha=0.5))
        sns.stripplot(x='group', y=feat, data=filtered_df, order=['A', 'B', 'C'],
                      palette=my_palette, size=2.5, hue='group', ax=ax,
                      jitter=0.2, alpha=0.6, linewidth=0.5, edgecolor='gray', legend=False)
        
        ax.set_title(var_dict[feat]['name'], fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        sns.despine(ax=ax)
        
    plt.tight_layout()
    st.pyplot(fig1)

# --- [우측: 인터랙티브 시계열 차트] ---
with col2:
    st.subheader(f"📈 {var_dict[selected_var]['name']} 추세 변화")
    
    # 안내 문구 (Information Box)
    st.info(f"**💡 지표 설명:** {var_dict[selected_var]['desc']}")
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.lineplot(x='year', y=selected_var, hue='group', data=filtered_df,
                 errorbar=None, palette=my_palette, linewidth=3,
                 marker='o', markersize=5, ax=ax2)

    ax2.set_xlabel('연도', fontsize=11)
    ax2.set_ylabel('지수 값', fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    legend = ax2.legend(title='국가 그룹', facecolor='white', edgecolor='gray')
    legend.get_frame().set_linewidth(0.5)
    sns.despine(ax=ax2)
    
    st.pyplot(fig2)
