"""
⚡ WarPrediction 데이터 정형화 EDA 도구
- 4개 탭: 개요 / 분포·이상치 / 그룹 비교 / 상관관계
- Group A/B/C 국가 필터 및 연도 범위 슬라이더
- 데이터셋 자동 인식 + 변수 설명
- Plotly & Seaborn 하이브리드 시각화 (한국어 폰트 지원)
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ============================================================
# 0. 전역 설정
# ============================================================
st.set_page_config(
    page_title="WarPrediction EDA",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 커스텀 CSS ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans KR', 'Malgun Gothic', sans-serif; }
  .block-container { padding: 1.5rem 2rem 2rem; }

  /* 탭 스타일 */
  [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #1e293b; }
  [data-baseweb="tab"] {
    background: #0f172a; border-radius: 6px 6px 0 0;
    color: #64748b; font-size: 0.85rem; padding: 0.4rem 1rem;
  }
  [aria-selected="true"][data-baseweb="tab"] {
    background: #1e293b !important; color: #f1f5f9 !important;
    border-bottom: 2px solid #38bdf8;
  }

  /* 메트릭 카드 */
  [data-testid="metric-container"] {
    background: #0f172a; border: 1px solid #1e293b;
    border-radius: 8px; padding: 0.8rem 1rem;
  }
  [data-testid="metric-container"] label { color: #64748b; font-size: 0.72rem; }
  [data-testid="metric-container"] [data-testid="metric-value"] {
    color: #f1f5f9; font-size: 1.5rem; font-weight: 600;
  }

  /* 정보 박스 */
  .info-box {
    background: #0f172a; border-left: 3px solid #38bdf8;
    padding: 0.6rem 1rem; border-radius: 0 6px 6px 0;
    color: #94a3b8; font-size: 0.82rem; margin-bottom: 0.8rem;
  }
  .group-badge-a { color: #ef4444; font-weight: 600; }
  .group-badge-b { color: #f97316; font-weight: 600; }
  .group-badge-c { color: #16a085; font-weight: 600; }
  div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── 공통 레이아웃 & 팔레트 ───────────────────────────────────
PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0f1a",
    plot_bgcolor="#0a0f1a",
    font=dict(family="IBM Plex Sans KR, Malgun Gothic, sans-serif", color="#cbd5e1"),
    margin=dict(l=40, r=20, t=50, b=40),
)

# ============================================================
# 1. 국가-그룹 매핑
# ============================================================
ISO3_GROUP = {
    "SYR":"A","YEM":"A","SOM":"A","MMR":"A","ETH":"A",
    "SSD":"A","MLI":"A","COD":"A","UKR":"A","IRQ":"A",
    "PAK":"B","NGA":"B","VEN":"B","SDN":"B","CAF":"B",
    "TWN":"B","HTI":"B","LBN":"B","COL":"B","ECU":"B",
    "NOR":"C","CHE":"C","JPN":"C","KOR":"C","PRT":"C",
    "URY":"C","BWA":"C","MNG":"C","CAN":"C","DEU":"C",
}
FIPS_GROUP = {
    "SY":"A","YM":"A","SO":"A","BM":"A","ET":"A","OD":"A","ML":"A","CG":"A","UP":"A","IZ":"A",
    "PK":"B","NI":"B","VE":"B","SU":"B","CT":"B","TW":"B","HA":"B","LE":"B","CO":"B","EC":"B",
    "NO":"C","SZ":"C","JA":"C","KS":"C","PO":"C","UY":"C","BC":"C","MG":"C","CA":"C","GM":"C",
}
NAME_GROUP = {
    "Syria":"A","Yemen":"A","Somalia":"A","Myanmar":"A","Ethiopia":"A","South Sudan":"A",
    "Mali":"A","Democratic Republic of Congo":"A","Congo/Zaire":"A","DRC":"A","Ukraine":"A","Iraq":"A",
    "Pakistan":"B","Nigeria":"B","Venezuela":"B","Sudan":"B","Central African Republic":"B",
    "Cen African Rep":"B","Taiwan":"B","Haiti":"B","Lebanon":"B","Colombia":"B","Ecuador":"B",
    "Norway":"C","Switzerland":"C","Japan":"C","South Korea":"C","Korea South":"C","Korea, South":"C",
    "Portugal":"C","Uruguay":"C","Botswana":"C","Mongolia":"C","Canada":"C","Germany":"C",
}

def detect_group(val):
    v = str(val).strip()
    return ISO3_GROUP.get(v) or FIPS_GROUP.get(v) or NAME_GROUP.get(v) or "?"

# ============================================================
# 2. 데이터셋 프로파일
# ============================================================
PROFILES = {
    "acled": {
        "label": "ACLED – 물리적 분쟁·시위 이벤트",
        "country_col": "country",
        "date_col": "event_date",
        "key_nums": ["fatalities","latitude","longitude"],
        "key_cats": ["event_type","sub_event_type","interaction"],
        "target_hint": "`fatalities` → 분쟁 강도 타겟 변수. `event_type` 분포와 국가별 사망자 합산을 먼저 확인하세요.",
        "var_desc": {
            "fatalities": "사건당 사망자 수 — 분쟁 강도의 핵심 타겟 변수",
            "event_type": "사건 대분류 (Battles / Protests / Riots / Explosions …)",
            "sub_event_type": "사건 소분류 — 시위→무력충돌 에스컬레이션 패턴 학습에 활용",
            "interaction": "행위자 상호작용 코드 (10=정부군 단독, 12=정부군 vs 반군 …)",
        },
    },
    "vdem": {
        "label": "V‑Dem – 민주주의·구조적 취약성",
        "country_col": "country_text_id",
        "date_col": "year",
        "key_nums": ["v2x_polyarchy","v2x_libdem","v2x_rule","v2x_corr","v2x_clphy","v2elpeace","v2x_veracc","v2xcs_ccsi"],
        "key_cats": ["e_civil_war","e_pt_coup"],
        "target_hint": "`e_civil_war` · `e_pt_coup` → 모델 타겟(Label). `v2x_polyarchy` 급락은 분쟁 전조.",
        "var_desc": {
            "v2x_polyarchy": "선거 민주주의 지수 (0~1) — 급락 시 분쟁 전조",
            "v2x_libdem": "자유 민주주의 지수 (0~1) — 소수자 권리 + 권력분립",
            "v2x_rule": "법의 지배 지수 (0~1) — 낮으면 국가 실패 위험",
            "v2x_corr": "정치 부패 지수 (0~1) — 높을수록 부패 → 폭동 명분",
            "v2x_clphy": "신체적 폭력 금지 지수 (0~1) — 내전 직전 급락",
            "v2elpeace": "선거 평화 지수 — 낮을수록 선거 폭력 심각",
            "v2x_veracc": "수직적 책임성 (0~1) — 막히면 폭력 시위로 번짐",
            "v2xcs_ccsi": "핵심 시민사회 지수 (0~1) — 탄압받을수록 분쟁 위험",
            "e_civil_war": "내전 발생 여부 (0/1) — 타겟 Label",
            "e_pt_coup": "쿠데타 발생 여부 (0/1) — 트리거 변수",
        },
    },
    "gdelt": {
        "label": "GDELT – 일별 언론·이벤트 신호",
        "country_col": "ActionGeo_CountryCode",
        "date_col": "SQLDATE",
        "key_nums": ["AvgGoldstein","AvgTone","TotalMentions","TotalSources","TotalArticles"],
        "key_cats": ["EventCode"],
        "target_hint": "`AvgGoldstein` (−10~+10) 음수 → 불안정. `AvgTone` 낮을수록 부정 기사.",
        "var_desc": {
            "AvgGoldstein": "골드스타인 지수 평균 (−10~+10) — 안정성 이론 영향치",
            "AvgTone": "기사 평균 어조 — 낮을수록 부정적 보도",
            "TotalMentions": "일일 총 언급 횟수 — 화제성 지표",
            "TotalSources": "일일 총 정보원 수 — 확산성 지표",
            "TotalArticles": "일일 총 기사 수 — 보도 집중도 지표",
        },
    },
    "reign": {
        "label": "REIGN – 정권·정치 불안정",
        "country_col": "country",
        "date_col": "date",
        "key_nums": ["tenure_months","irregular","anticipation"],
        "key_cats": ["government","election","leader_change"],
        "target_hint": "`irregular` · `anticipation` 상승 → 정권 교체·쿠데타 전조.",
        "var_desc": {
            "irregular": "비정상적 정치 변화 지표 — 상승 시 쿠데타·폭동 전조",
            "anticipation": "정치 긴장·불안정 관련 지표",
            "tenure_months": "지도자 집권 기간(개월) — 짧을수록 정치 불안정",
            "leader_change": "지도자 교체 여부 (0/1)",
            "election": "선거 진행 여부 (0/1)",
        },
    },
    "ucdp": {
        "label": "UCDP GED – 지리 참조 폭력 이벤트",
        "country_col": "country",
        "date_col": "date_start",
        "key_nums": ["best"],
        "key_cats": ["type_of_violence","region","group"],
        "target_hint": "`best` (최적 추정 사망자)가 핵심 강도 변수.",
        "var_desc": {
            "best": "최적 추정 총 사망자 수 — 폭력 강도 핵심 지표",
            "type_of_violence": "폭력 유형 코드 (1=국가, 2=비국가, 3=일방적 폭력)",
        },
    },
    "disaster": {
        "label": "EM‑DAT – 자연재해",
        "country_col": "country",
        "date_col": "start_date",
        "key_nums": ["total_deaths","total_affected"],
        "key_cats": ["disaster_type"],
        "target_hint": "⚠️ −999 = 누락값 (필터 필수). `start_date`의 '0월0일'은 날짜 불명 이벤트.",
        "var_desc": {
            "total_deaths": "총 사망자 수 (−999 = 누락)",
            "total_affected": "총 피해자 수 (−999 = 누락)",
            "disaster_type": "재해 유형 (Drought / Flood / Storm / Epidemic …)",
        },
    },
    "fao": {
        "label": "FAO – 월별 식량가격지수",
        "country_col": "country",
        "date_col": "date",
        "key_nums": ["PFCS"],
        "key_cats": [],
        "target_hint": "`PFCS` (2015=100): 높을수록 식량 가격 상승. 시계열 추세 시각화 권장.",
        "var_desc": {"PFCS": "식량가격종합지수 (기준: 2015=100)"},
    },
    "unhcr": {
        "label": "UNHCR – 강제이주·난민",
        "country_col": "country_iso",
        "date_col": "year",
        "key_nums": ["refugees","idps","asylum_seekers","population","ForcedMig"],
        "key_cats": [],
        "target_hint": "`refugees + idps + asylum_seekers` → 분쟁 강도 간접 지표.",
        "var_desc": {
            "refugees": "타국으로 이동한 난민 수",
            "idps": "국내실향민 수",
            "asylum_seekers": "망명 신청자 수",
            "ForcedMig": "강제이주 종합 지표",
        },
    },
    "worldbank": {
        "label": "World Bank – 경제 지표",
        "country_col": "ISO3",
        "date_col": "Year",
        "key_nums": ["GDP_Per_Capita","GDP_Growth","Inflation","Unemployment","Military_Expenditure"],
        "key_cats": [],
        "target_hint": "`Military_Expenditure` 상승 + `GDP_Growth` 하락 → 복합 분쟁 신호.",
        "var_desc": {
            "GDP_Per_Capita": "1인당 GDP (불변가격)",
            "GDP_Growth": "연간 GDP 성장률 (%)",
            "Inflation": "소비자 물가 상승률 (%)",
            "Unemployment": "총 실업률 (%)",
            "Military_Expenditure": "GDP 대비 군사비 (%)",
        },
    },
}

def match_profile(fname: str):
    n = fname.lower()
    for k, p in PROFILES.items():
        if k in n:
            return p
    return None

# ============================================================
# 3. 캐시 함수들
# ============================================================
DATA_DIR = r"C:\Users\hk100\Desktop\WarPrediction\Crawling\Parquet_Unstructured"

@st.cache_data(show_spinner=False)
def load_file(path):
    return pd.read_parquet(path, engine="pyarrow") if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)

@st.cache_data(show_spinner=False)
def get_meta(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    total_miss = int(df.isna().sum().sum())
    miss_pct = (df.isna().mean() * 100).round(2)
    miss_df = miss_pct[miss_pct > 0].sort_values(ascending=False).reset_index()
    miss_df.columns = ["변수명", "결측 비율(%)"]
    miss_df["결측 건수"] = df.isna().sum()[miss_df["변수명"]].values
    return num_cols, cat_cols, total_miss, miss_df

@st.cache_data(show_spinner=False)
def iqr_outliers(series: pd.Series):
    arr = series.dropna().values.astype(float)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    mask = (arr < lo) | (arr > hi)
    return float(lo), float(hi), int(mask.sum()), float(mask.mean()*100)

@st.cache_data(show_spinner=False)
def fast_hist(series: pd.Series, nbins=80):
    arr = series.dropna().values.astype(float)
    counts, edges = np.histogram(arr, bins=nbins)
    return (edges[:-1]+edges[1:])/2, counts

@st.cache_data(show_spinner=False)
def corr_matrix(df: pd.DataFrame, cols: list):
    return df[cols].corr()

# ============================================================
# 4. 파일 목록 및 로드
# ============================================================
if not os.path.exists(DATA_DIR):
    st.error(f"❌ 폴더 없음: {DATA_DIR}")
    st.stop()

all_files = [f for f in os.listdir(DATA_DIR) if f.endswith((".parquet",".csv"))]
if not all_files:
    st.warning("⚠️ 파일이 없습니다.")
    st.stop()

# ============================================================
# 5. 사이드바 구성
# ============================================================
with st.sidebar:
    st.markdown("## 🌍 WarPrediction EDA")
    st.divider()

    selected_file = st.selectbox("📂 파일 선택", all_files)
    file_path = os.path.join(DATA_DIR, selected_file)
    profile = match_profile(selected_file)
    st.caption(f"💾 {os.path.getsize(file_path)/1e6:.1f} MB")

    if profile:
        st.markdown(f"**{profile['label']}**")
        st.markdown(f"<div class='info-box'>💡 {profile['target_hint']}</div>", unsafe_allow_html=True)

    with st.spinner("데이터 로드 중…"):
        raw = load_file(file_path)
    
    st.divider()
    st.markdown("### 📅 분석 연도 범위")
    date_col = profile["date_col"] if profile else None
    selected_years = (1990, 2025)
    if date_col and date_col in raw.columns:
        parsed_dates = pd.to_datetime(raw[date_col], format='mixed', errors='coerce')
        if parsed_dates.notna().sum() > 0:
            min_yr = int(parsed_dates.dt.year.min())
            max_yr = int(parsed_dates.dt.year.max())
        else:
            min_yr = int(raw[date_col].min())
            max_yr = int(raw[date_col].max())
            
        if min_yr < max_yr:
            selected_years = st.slider("연도 범위", min_yr, max_yr, (min_yr, max_yr))
        else:
            selected_years = (min_yr, max_yr)
            st.info(f"단일 연도 데이터입니다: {min_yr}년")

    st.divider()
    st.markdown("### 🗂️ 국가 그룹")
    show_a = st.checkbox("🔴 A – 활성 분쟁", True)
    show_b = st.checkbox("🟠 B – 고위험 잠재", True)
    show_c = st.checkbox("🟢 C – 안정 대조군", True)
    sel_groups = {g for g, s in zip("ABC",[show_a,show_b,show_c]) if s} or {"A","B","C"}

    st.divider()
    st.markdown("### ⚙️ 이상치 옵션")
    outlier_method = st.radio("이상치 감지 방식", ["IQR × 1.5 (기본)", "IQR × 3.0 (극단값만)"], index=0)
    iqr_k = 1.5 if "1.5" in outlier_method else 3.0

# ============================================================
# 6. 데이터 필터링 & 그룹 부착
# ============================================================
country_col = profile["country_col"] if profile else None
if country_col and country_col in raw.columns:
    raw = raw.copy()
    raw["__group__"] = raw[country_col].map(detect_group).fillna("?")
    df = raw[raw["__group__"].isin(sel_groups | {"?"})]
else:
    df = raw.copy()
    df["__group__"] = "?"

# 연도 필터링 적용
if date_col and date_col in df.columns:
    parsed = pd.to_datetime(df[date_col], format='mixed', errors='coerce')
    if parsed.notna().sum() > 0:
        df = df[(parsed.dt.year >= selected_years[0]) & (parsed.dt.year <= selected_years[1])]
    else:
        df = df[(df[date_col] >= selected_years[0]) & (df[date_col] <= selected_years[1])]

num_cols, cat_cols, total_miss, miss_df = get_meta(df)
all_cols = [c for c in df.columns if c != "__group__"]

# ============================================================
# 7. 메인 화면 헤더
# ============================================================
st.markdown(f"## 🌍 {selected_file.replace('_',' ').replace('.parquet','').replace('.csv','')}")
st.markdown(
    '<span class="group-badge-a">Group A 활성분쟁</span> &nbsp;|&nbsp; '
    '<span class="group-badge-b">Group B 고위험</span> &nbsp;|&nbsp; '
    '<span class="group-badge-c">Group C 안정</span>',
    unsafe_allow_html=True,
)
st.markdown("")

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("총 행수", f"{len(df):,}")
m2.metric("총 열수", f"{len(df.columns)-1:,}")
m3.metric("수치형 변수", f"{len(num_cols):,}")
m4.metric("범주형 변수", f"{len(cat_cols):,}")
m5.metric("전체 결측치", f"{total_miss:,}")

st.markdown("")

# ============================================================
# 8. 탭 레이아웃
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 데이터 개요",
    "📊 분포 & 이상치",
    "🗂️ 그룹 비교",
    "🔗 상관관계",
])

# ─────────────────────────────────────────────────────────────
# TAB 1 · 데이터 개요
# ─────────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### 결측치 비율 (상위 20개 변수)")
        if len(miss_df):
            top_miss = miss_df.head(20)
            fig_miss = go.Figure(go.Bar(
                x=top_miss["결측 비율(%)"],
                y=top_miss["변수명"],
                orientation="h",
                marker=dict(
                    color=top_miss["결측 비율(%)"],
                    colorscale=[[0,"#1e3a5f"],[0.5,"#1d6fa5"],[1,"#ef4444"]],
                    showscale=False,
                ),
                text=top_miss["결측 비율(%)"].map(lambda x: f"{x:.1f}%"),
                textposition="outside",
            ))
            fig_miss.update_layout(**PLOTLY_BASE)
            fig_miss.update_layout(
                height=max(280, len(top_miss)*28),
                xaxis=dict(title="결측 비율 (%)", range=[0, 105]),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=140, r=60, t=20, b=40),
            )
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("✅ 결측치가 없는 깨끗한 데이터입니다!")

    with col_right:
        st.markdown("#### 변수별 데이터 타입 & 기술통계")
        dtype_df = pd.DataFrame({
            "변수명": all_cols,
            "타입": [str(df[c].dtype) for c in all_cols],
            "비결측(%)": [(1 - df[c].isna().mean())*100 for c in all_cols],
            "고유값수": [df[c].nunique() for c in all_cols],
        })
        dtype_df["비결측(%)"] = dtype_df["비결측(%)"].round(1)
        st.dataframe(
            dtype_df.style.background_gradient(subset=["비결측(%)"], cmap="RdYlGn"),
            use_container_width=True,
            height=420,
        )

    st.divider()
    st.markdown("#### 수치형 변수 기술통계")
    if num_cols:
        desc = df[num_cols].describe().T.round(3)
        desc.index.name = "변수명"
        st.dataframe(
            desc.reset_index().style.background_gradient(subset=["mean","std"], cmap="Blues"),
            use_container_width=True,
        )

        if profile and profile.get("var_desc"):
            with st.expander("📖 변수 설명 보기", expanded=False):
                for k, v in profile["var_desc"].items():
                    st.markdown(f"- **`{k}`** : {v}")

# ─────────────────────────────────────────────────────────────
# TAB 2 · 분포 & 이상치
# ─────────────────────────────────────────────────────────────
with tab2:
    if not num_cols:
        st.warning("수치형 변수가 없습니다.")
    else:
        suggest = [c for c in (profile["key_nums"] if profile else []) if c in df.columns] or num_cols
        sel_var = st.selectbox("분석할 변수 선택", suggest, key="dist_var")

        series = df[sel_var].dropna()
        lo, hi, n_out, pct_out = iqr_outliers(series)
        if iqr_k != 1.5:
            arr = series.values.astype(float)
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - iqr_k*iqr, q3 + iqr_k*iqr
            n_out = int(((arr < lo) | (arr > hi)).sum())
            pct_out = float(n_out / len(arr) * 100)

        o1,o2,o3,o4,o5 = st.columns(5)
        o1.metric("평균", f"{series.mean():.3g}")
        o2.metric("중앙값", f"{series.median():.3g}")
        o3.metric("표준편차", f"{series.std():.3g}")
        o4.metric("이상치 건수", f"{n_out:,}")
        o5.metric("이상치 비율", f"{pct_out:.2f}%", delta=None)

        st.markdown("")
        dist_left, dist_right = st.columns([1.4, 1], gap="large")

        with dist_left:
            st.markdown(f"##### `{sel_var}` 분포")
            centers, counts = fast_hist(series)
            fig_h = go.Figure()
            mask_ok = (centers >= lo) & (centers <= hi)
            fig_h.add_trace(go.Bar(x=centers[mask_ok], y=counts[mask_ok],
                name="정상 범위", marker_color="#38bdf8", marker_line_width=0))
            fig_h.add_trace(go.Bar(x=centers[~mask_ok], y=counts[~mask_ok],
                name="이상치 범위", marker_color="#ef4444", marker_line_width=0))
            fig_h.add_vline(x=lo, line_dash="dash", line_color="#f97316",
                annotation_text=f"하한 {lo:.2g}", annotation_position="top right",
                annotation_font_size=11)
            fig_h.add_vline(x=hi, line_dash="dash", line_color="#f97316",
                annotation_text=f"상한 {hi:.2g}", annotation_position="top left",
                annotation_font_size=11)
            fig_h.update_layout(
                **PLOTLY_BASE, barmode="overlay", bargap=0.03,
                xaxis_title=sel_var, yaxis_title="count",
                legend=dict(orientation="h", y=1.08),
                height=320,
            )
            st.plotly_chart(fig_h, use_container_width=True)

        with dist_right:
            st.markdown(f"##### `{sel_var}` 박스플롯")
            arr_f = series.values.astype(float)
            q1v, med_v, q3v = np.percentile(arr_f, [25, 50, 75])
            iqrv = q3v - q1v
            out_vals = arr_f[(arr_f < lo) | (arr_f > hi)]
            if len(out_vals) > 50_000:
                rng = np.random.default_rng(42)
                out_vals = rng.choice(out_vals, 50_000, replace=False)

            fig_b = go.Figure()
            fig_b.add_trace(go.Box(
                q1=[q1v], median=[med_v], q3=[q3v],
                lowerfence=[lo], upperfence=[hi],
                name="IQR", marker_color="#38bdf8",
                line_color="#38bdf8", fillcolor="rgba(56,189,248,0.2)",
                boxpoints=False,
            ))
            if len(out_vals):
                fig_b.add_trace(go.Scatter(
                    x=out_vals, y=np.zeros(len(out_vals)),
                    mode="markers",
                    marker=dict(color="#ef4444", size=4, opacity=0.4),
                    name=f"이상치 {n_out:,}건",
                ))
            fig_b.update_layout(
                **PLOTLY_BASE, height=320,
                showlegend=True,
                legend=dict(orientation="h", y=1.08),
                xaxis_title=sel_var,
            )
            st.plotly_chart(fig_b, use_container_width=True)

        st.markdown("##### 이상치 샘플 (최대 100행)")
        out_df = df[df[sel_var].notna() & ((df[sel_var] < lo) | (df[sel_var] > hi))]
        show_cols = [c for c in ([country_col, sel_var] + (profile["key_cats"] if profile else [])) if c and c in df.columns]
        if len(out_df):
            st.dataframe(out_df[show_cols].head(100), use_container_width=True)
        else:
            st.info("이상치가 없습니다.")

# ─────────────────────────────────────────────────────────────
# TAB 3 · 그룹 비교 (⭐️ 요청하신 화이트 바탕의 Seaborn 감성 코드 이식)
# ─────────────────────────────────────────────────────────────
with tab3:
    if "__group__" not in df.columns or df["__group__"].nunique() <= 1:
        st.warning("국가 그룹 컬럼을 인식하지 못했습니다. 사이드바 파일 선택을 확인하세요.")
    else:
        grp_nums = [c for c in (profile["key_nums"] if profile else num_cols) if c in df.columns]
        if not grp_nums:
            st.warning("수치형 변수가 없습니다.")
        else:
            col1, col2 = st.columns([1.1, 0.9], gap="large")

            # Tab 3 전용 Seaborn 테마 강제 (이미지처럼 화이트 박스 형태 유지)
            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] = False
            sns.set_theme(style="whitegrid", font='Malgun Gothic')
            
            # 커스텀 팔레트 매핑 (A: Red, B: Orange, C: Teal)
            tab3_palette = {"A": "#C0392B", "B": "#E67E22", "C": "#16A085", "?": "#888888"}

            # --- [좌측: 박스플롯 (분포 요약)] ---
            with col1:
                st.subheader("📊 주요 지표 그룹별 분포 비교")
                if selected_years:
                    st.caption(f"기준: {selected_years[0]}년 ~ {selected_years[1]}년 통합")
                
                # 변수가 모자랄 수 있으므로 동적 처리 (최대 4개)
                box_features = grp_nums[:4]
                
                fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
                
                for i, feat in enumerate(box_features):
                    row, col = divmod(i, 2)
                    ax = axes1[row, col]
                    
                    # 프로파일에서 한글 이름 가져오기
                    title_text = profile["var_desc"].get(feat, feat) if profile and "var_desc" in profile else feat
                    short_title = title_text.split("—")[0].split("(")[0].strip()
                    
                    sns.boxplot(x='__group__', y=feat, data=df, order=sorted(sel_groups),
                                palette=tab3_palette, width=0.5, ax=ax, linewidth=1.5, 
                                fliersize=0, hue='__group__', legend=False, boxprops=dict(alpha=0.5))
                    sns.stripplot(x='__group__', y=feat, data=df, order=sorted(sel_groups),
                                  palette=tab3_palette, size=2.5, hue='__group__', ax=ax,
                                  jitter=0.2, alpha=0.6, linewidth=0.5, edgecolor='gray', legend=False)
                    
                    ax.set_title(short_title, fontsize=12, fontweight='bold')
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    sns.despine(ax=ax)
                
                # 변수가 4개 미만일 때 빈 차트 숨기기
                for j in range(len(box_features), 4):
                    r, c = divmod(j, 2)
                    fig1.delaxes(axes1[r, c])
                    
                plt.tight_layout()
                st.pyplot(fig1)

            # --- [우측: 인터랙티브 시계열 차트] ---
            with col2:
                selected_var = st.selectbox("📈 시계열 분석 변수 선택", grp_nums, key="ts_var")
                
                title_text = profile["var_desc"].get(selected_var, selected_var) if profile and "var_desc" in profile else selected_var
                short_title = title_text.split("—")[0].split("(")[0].strip()
                st.subheader(f"📈 {short_title} 추세 변화")
                
                # 안내 문구 (Information Box)
                if profile and profile.get("var_desc") and selected_var in profile["var_desc"]:
                    st.info(f"**💡 지표 설명:** {profile['var_desc'][selected_var]}")
                
                if date_col and date_col in df.columns:
                    # 그룹별 연도별 평균 집계
                    ts = df[[date_col, "__group__", selected_var]].copy()
                    try:
                        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
                        if ts[date_col].isna().mean() > 0.5:
                            ts[date_col] = pd.to_numeric(ts[date_col], errors="coerce")
                    except:
                        ts[date_col] = pd.to_numeric(ts[date_col], errors="coerce")

                    ts = ts.dropna(subset=[date_col, selected_var])
                    ts[selected_var] = pd.to_numeric(ts[selected_var], errors="coerce")
                    ts = ts.dropna(subset=[selected_var])

                    if pd.api.types.is_datetime64_any_dtype(ts[date_col]):
                        ts["__t__"] = ts[date_col].dt.year
                    else:
                        ts["__t__"] = ts[date_col]

                    agg_ts = ts.groupby(["__t__", "__group__"])[selected_var].mean().reset_index()

                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    sns.lineplot(x='__t__', y=selected_var, hue='__group__', data=agg_ts,
                                 errorbar=None, palette=tab3_palette, linewidth=3,
                                 marker='o', markersize=5, ax=ax2, hue_order=sorted(sel_groups))

                    ax2.set_xlabel('연도', fontsize=11)
                    ax2.set_ylabel('지수 값', fontsize=11)
                    ax2.grid(axis='y', linestyle='--', alpha=0.5)
                    
                    legend = ax2.legend(title='국가 그룹', facecolor='white', edgecolor='gray')
                    legend.get_frame().set_linewidth(0.5)
                    sns.despine(ax=ax2)
                    
                    st.pyplot(fig2)
                else:
                    st.warning("시계열을 그릴 수 있는 날짜/연도 컬럼이 없습니다.")

# ─────────────────────────────────────────────────────────────
# TAB 4 · 상관관계
# ─────────────────────────────────────────────────────────────
with tab4:
    if len(num_cols) < 2:
        st.warning("상관관계 분석에는 수치형 변수가 2개 이상 필요합니다.")
    else:
        corr_vars = [c for c in (profile["key_nums"] if profile else num_cols) if c in df.columns]
        if len(corr_vars) < 2:
            corr_vars = num_cols[:15]
        corr_vars = corr_vars[:20]

        corr = corr_matrix(df, corr_vars)

        st.markdown("#### 수치형 변수 상관계수 히트맵")
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[
                [0.0, "#b91c1c"],
                [0.25, "#dc2626"],
                [0.5, "#0f172a"],
                [0.75, "#0369a1"],
                [1.0, "#0284c7"],
            ],
            zmid=0,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=10),
            hoverongaps=False,
        ))
        fig_corr.update_layout(**PLOTLY_BASE)
        fig_corr.update_layout(
            height=max(420, len(corr_vars)*36),
            xaxis=dict(tickangle=-35, tickfont=dict(size=11)),
            yaxis=dict(tickfont=dict(size=11)),
            margin=dict(l=120, r=20, t=30, b=100),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("#### 주목할 강한 상관관계 (|r| ≥ 0.6)")
        pairs = []
        for i in range(len(corr_vars)):
            for j in range(i+1, len(corr_vars)):
                r = corr.iloc[i, j]
                if abs(r) >= 0.6:
                    pairs.append({
                        "변수 A": corr_vars[i],
                        "변수 B": corr_vars[j],
                        "상관계수 r": round(r, 3),
                        "해석": "🔴 강한 양의 상관" if r > 0 else "🔵 강한 음의 상관",
                    })
        if pairs:
            pairs_df = pd.DataFrame(pairs).sort_values("상관계수 r", key=abs, ascending=False)
            st.dataframe(pairs_df, use_container_width=True)
        else:
            st.info("|r| ≥ 0.6인 변수 쌍이 없습니다.")
