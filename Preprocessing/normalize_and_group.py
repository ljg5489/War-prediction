"""
normalize_and_group.py
──────────────────────────────────────────────────────────────────────────────
국가명 통일 & 그룹 레이블 부여 (전 세계)

사용법:
    python normalize_and_group.py <입력CSV> [<입력CSV2> ...]
    python normalize_and_group.py GED.csv --col country
    python normalize_and_group.py qog.csv --col cname
    python normalize_and_group.py GED.csv qog.csv vdem.csv   (여러 파일 한번에)

출력:
    입력파일명_grouped.csv (같은 폴더에 저장)
    추가 컬럼:
      country_std : 통일된 표준 국가명
      group       : A / B / C / OTHER

그룹 의미:
    A     : 고분쟁 10개국        (EDA·시각화 분석 레이블)
    B     : 중위험 10개국        (EDA·시각화 분석 레이블)
    C     : 저위험 대조군 10개국 (step4 전쟁 판별 기준)
    OTHER : 나머지 전 세계 국가  (학습 데이터에 포함)
──────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import argparse
import os

# ── 분석용 그룹 레이블 ────────────────────────────────────────────────────────
CANONICAL = {
    # A그룹 - 고분쟁
    "Syria":                    "A",
    "Yemen":                    "A",
    "Somalia":                  "A",
    "Myanmar":                  "A",
    "Ethiopia":                 "A",
    "South Sudan":              "A",
    "Mali":                     "A",
    "DR Congo":                 "A",
    "Ukraine":                  "A",
    "Iraq":                     "A",
    # B그룹 - 중위험
    "Pakistan":                 "B",
    "Nigeria":                  "B",
    "Venezuela":                "B",
    "Sudan":                    "B",
    "Central African Republic": "B",
    "Taiwan":                   "B",
    "Haiti":                    "B",
    "Lebanon":                  "B",
    "Colombia":                 "B",
    "Ecuador":                  "B",
    # C그룹 - 저위험 대조군
    "Norway":                   "C",
    "Switzerland":              "C",
    "Japan":                    "C",
    "South Korea":              "C",
    "Portugal":                 "C",
    "Uruguay":                  "C",
    "Botswana":                 "C",
    "Mongolia":                 "C",
    "Canada":                   "C",
    "Germany":                  "C",
}

# ── 데이터셋별 국가명 변형 → 표준명 매핑 ────────────────────────────────────
ALIAS_MAP = {
    # Syria
    "syria":                                    "Syria",
    "syrian arab republic":                     "Syria",
    "syr":                                      "Syria",
    # Yemen
    "yemen":                                    "Yemen",
    "yemen, rep.":                              "Yemen",
    "republic of yemen":                        "Yemen",
    "yem":                                      "Yemen",
    # Somalia
    "somalia":                                  "Somalia",
    "som":                                      "Somalia",
    # Myanmar
    "myanmar":                                  "Myanmar",
    "myanmar (burma)":                          "Myanmar",
    "burma":                                    "Myanmar",
    "mmr":                                      "Myanmar",
    "mya":                                      "Myanmar",
    # Ethiopia
    "ethiopia":                                 "Ethiopia",
    "eth":                                      "Ethiopia",
    # South Sudan
    "south sudan":                              "South Sudan",
    "s. sudan":                                 "South Sudan",
    "ssd":                                      "South Sudan",
    # Mali
    "mali":                                     "Mali",
    "mli":                                      "Mali",
    # DR Congo
    "dr congo":                                 "DR Congo",
    "dr congo (zaire)":                         "DR Congo",
    "congo (the democratic republic of the)":   "DR Congo",
    "democratic republic of the congo":         "DR Congo",
    "democratic republic of congo":             "DR Congo",
    "congo, dem. rep.":                         "DR Congo",
    "congo, democratic republic":               "DR Congo",
    "drc":                                      "DR Congo",
    "zaire":                                    "DR Congo",
    "cod":                                      "DR Congo",
    # Ukraine
    "ukraine":                                  "Ukraine",
    "ukr":                                      "Ukraine",
    # Iraq
    "iraq":                                     "Iraq",
    "irq":                                      "Iraq",
    # Pakistan
    "pakistan":                                 "Pakistan",
    "pak":                                      "Pakistan",
    # Nigeria
    "nigeria":                                  "Nigeria",
    "nga":                                      "Nigeria",
    "nig":                                      "Nigeria",
    # Venezuela
    "venezuela":                                "Venezuela",
    "venezuela, rb":                            "Venezuela",
    "bolivarian republic of venezuela":         "Venezuela",
    "ven":                                      "Venezuela",
    # Sudan
    "sudan":                                    "Sudan",
    "sdn":                                      "Sudan",
    "sud":                                      "Sudan",
    # Central African Republic
    "central african republic":                 "Central African Republic",
    "car":                                      "Central African Republic",
    "caf":                                      "Central African Republic",
    # Taiwan
    "taiwan":                                   "Taiwan",
    "taiwan, province of china":                "Taiwan",
    "twn":                                      "Taiwan",
    # Haiti
    "haiti":                                    "Haiti",
    "hti":                                      "Haiti",
    # Lebanon
    "lebanon":                                  "Lebanon",
    "lbn":                                      "Lebanon",
    "leb":                                      "Lebanon",
    # Colombia
    "colombia":                                 "Colombia",
    "col":                                      "Colombia",
    # Ecuador
    "ecuador":                                  "Ecuador",
    "ecu":                                      "Ecuador",
    # Norway
    "norway":                                   "Norway",
    "nor":                                      "Norway",
    # Switzerland
    "switzerland":                              "Switzerland",
    "che":                                      "Switzerland",
    "sui":                                      "Switzerland",
    # Japan
    "japan":                                    "Japan",
    "jpn":                                      "Japan",
    # South Korea
    "south korea":                              "South Korea",
    "korea, south":                             "South Korea",
    "korea, rep.":                              "South Korea",
    "korea (the republic of)":                  "South Korea",
    "republic of korea":                        "South Korea",
    "kor":                                      "South Korea",
    # Portugal
    "portugal":                                 "Portugal",
    "prt":                                      "Portugal",
    # Uruguay
    "uruguay":                                  "Uruguay",
    "ury":                                      "Uruguay",
    # Botswana
    "botswana":                                 "Botswana",
    "bwa":                                      "Botswana",
    # Mongolia
    "mongolia":                                 "Mongolia",
    "mng":                                      "Mongolia",
    # Canada
    "canada":                                   "Canada",
    "can":                                      "Canada",
    # Germany
    "germany":                                  "Germany",
    "deu":                                      "Germany",
    "ger":                                      "Germany",
    # Afghanistan
    "afghanistan":                              "Afghanistan",
    "afg":                                      "Afghanistan",
    # USA
    "united states":                            "United States",
    "united states of america":                 "United States",
    "usa":                                      "United States",
    "us":                                       "United States",
    # Russia
    "russia":                                   "Russia",
    "russian federation":                       "Russia",
    "rus":                                      "Russia",
    # China
    "china":                                    "China",
    "people's republic of china":               "China",
    "chn":                                      "China",
    # India
    "india":                                    "India",
    "ind":                                      "India",
    # Libya
    "libya":                                    "Libya",
    "lby":                                      "Libya",
    # Mozambique
    "mozambique":                               "Mozambique",
    "moz":                                      "Mozambique",
    # Cameroon
    "cameroon":                                 "Cameroon",
    "cmr":                                      "Cameroon",
    # Burkina Faso
    "burkina faso":                             "Burkina Faso",
    "bfa":                                      "Burkina Faso",
    # Niger
    "niger":                                    "Niger",
    "ner":                                      "Niger",
    # Chad
    "chad":                                     "Chad",
    "tcd":                                      "Chad",
    # Philippines
    "philippines":                              "Philippines",
    "phl":                                      "Philippines",
    # Mexico
    "mexico":                                   "Mexico",
    "mex":                                      "Mexico",
    # Brazil
    "brazil":                                   "Brazil",
    "bra":                                      "Brazil",
    # Iran
    "iran":                                     "Iran",
    "iran, islamic rep.":                       "Iran",
    "irn":                                      "Iran",
    # Turkey
    "turkey":                                   "Turkey",
    "turkiye":                                  "Turkey",
    "tur":                                      "Turkey",
    # Egypt
    "egypt":                                    "Egypt",
    "egypt, arab rep.":                         "Egypt",
    "egy":                                      "Egypt",
    # Kenya
    "kenya":                                    "Kenya",
    "ken":                                      "Kenya",
    # United Kingdom
    "united kingdom":                           "United Kingdom",
    "uk":                                       "United Kingdom",
    "gbr":                                      "United Kingdom",
    # France
    "france":                                   "France",
    "fra":                                      "France",
    # North Korea
    "Dem. People's Rep. of Korea":              "North Korea",
    "Democratic People's Republic of Korea":    "North Korea",
    "DPRK":                                     "North Korea",
    "north korea":                               "North Korea",
    
}

CANDIDATE_COLS = [
    "country", "cname", "country_name", "Country", "COUNTRY",
    "country_text_id", "CountryName", "nation", "state",
]

# ── 데이터셋별 슬림 프로파일 ──────────────────────────────────────────────────
# markers  : 이 컬럼들이 모두 있으면 해당 데이터셋으로 판별
# keep     : 슬림 후 남길 컬럼 (country_std, group은 항상 포함)
DATASET_PROFILES = [
    {
        "name":    "UCDP",
        "markers": ["type_of_violence", "best", "date_start"],
        "keep":    ["year", "country_std", "group",
                    "type_of_violence", "best", "date_start"],
    },
    {
        "name":    "QOG",
        "markers": ["al_ethnic2000"],
        "keep":    ["year", "country_std", "group",
                    "al_ethnic2000"],
    },
    {
        "name":    "VDem",
        "markers": ["v2x_libdem", "v2elintim", "v2x_corr"],
        "keep":    ["year", "country_std", "group",
                    "v2x_libdem", "v2elintim", "v2x_corr"],
    },
    {
        "name":    "WorldBank",
        "markers": ["ny_gdp_pcap_kd_zg", "fp_cpi_totl_zg", "ms_mil_xpnd_gd_zs"],
        "keep":    ["year", "country_std", "group",
                    "ny_gdp_pcap_kd_zg", "fp_cpi_totl_zg", "ms_mil_xpnd_gd_zs"],
    },
    {
        "name":    "GDELT",
        "markers": ["avg_tone", "goldstein_avg"],
        "keep":    ["year", "country_std", "group",
                    "avg_tone", "goldstein_avg"],
    },
]


def detect_dataset(df):
    """컬럼 구성으로 데이터셋 종류 자동 판별. 매칭 없으면 None 반환."""
    for profile in DATASET_PROFILES:
        if all(c in df.columns for c in profile["markers"]):
            return profile
    return None


def slim_df(df, profile):
    """프로파일에 정의된 컬럼만 남기고 메모리 절감."""
    before_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    available = [c for c in profile["keep"] if c in df.columns]
    missing   = [c for c in profile["keep"] if c not in df.columns]
    if missing:
        print(f"  ⚠️  없는 컬럼 (스킵): {missing}")
    df_slim   = df[available].copy()
    after_mb  = df_slim.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"  컬럼 {len(df.columns)}개 → {len(df_slim.columns)}개: {list(df_slim.columns)}")
    print(f"  메모리: {before_mb:.1f} MB → {after_mb:.1f} MB")
    return df_slim


# ── 처리 함수 ────────────────────────────────────────────────────────────────

def normalize(name):
    """원본 국가명 → 표준명. 매핑 없으면 원본 그대로 반환."""
    if pd.isna(name):
        return None
    key = str(name).strip().lower()
    if key in ALIAS_MAP:
        return ALIAS_MAP[key]
    return str(name).strip()   # 매핑 실패 → 원본 유지 (OTHER로 분류됨)


def assign_group(std_name):
    """표준명 → 그룹 (A/B/C/OTHER)."""
    if std_name is None:
        return "OTHER"
    return CANONICAL.get(std_name, "OTHER")


def detect_country_col(df):
    """국가명 컬럼 자동 감지."""
    for col in CANDIDATE_COLS:
        if col in df.columns:
            return col
    lower_map = {c.lower(): c for c in df.columns}
    for cand in CANDIDATE_COLS:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def process_file(input_path, country_col=None):
    print(f"\n{'='*60}")
    print(f"  처리 중: {os.path.basename(input_path)}")
    print(f"{'='*60}")

    df = pd.read_csv(input_path, low_memory=False)
    print(f"  행 수: {len(df):,}  /  컬럼 수: {len(df.columns)}")

    # 국가명 컬럼 결정
    if country_col:
        if country_col not in df.columns:
            raise ValueError(
                f"컬럼 '{country_col}'이 없습니다.\n"
                f"실제 컬럼: {list(df.columns[:20])}"
            )
    else:
        country_col = detect_country_col(df)
        if country_col is None:
            raise ValueError(
                "국가명 컬럼을 자동 감지할 수 없습니다.\n"
                "--col 옵션으로 직접 지정해 주세요.\n"
                f"실제 컬럼: {list(df.columns[:20])}"
            )
        print(f"  국가명 컬럼 자동 감지: '{country_col}'")

    # 표준화 & 그룹 부여
    df["country_std"] = df[country_col].apply(normalize)
    df["group"]       = df["country_std"].apply(assign_group)

    # 결과 요약
    total   = len(df)
    n_abc   = (df["group"].isin(["A", "B", "C"])).sum()
    n_other = (df["group"] == "OTHER").sum()

    print(f"\n  📊 그룹 분류 결과")
    print(f"     전체 행         : {total:,}")
    print(f"     A/B/C 분석 그룹 : {n_abc:,}행  ({n_abc/total*100:.1f}%)")
    print(f"     OTHER (나머지)  : {n_other:,}행  ({n_other/total*100:.1f}%) ← 학습 포함")

    print(f"\n  📋 그룹별 국가 분포 (A/B/C)")
    abc = (
        df[df["group"].isin(["A", "B", "C"])]
        .groupby(["group", "country_std"]).size()
        .reset_index(name="행 수")
        .sort_values(["group", "country_std"])
    )
    if len(abc):
        print(abc.to_string(index=False))

    n_other_countries = df[df["group"] == "OTHER"]["country_std"].nunique()
    print(f"\n  📋 OTHER 그룹 국가 수: {n_other_countries}개국")

    # ── 데이터셋 자동 감지 → 필요 컬럼만 유지 ──────────────────────────────
    profile = detect_dataset(df)
    if profile:
        print(f"\n  🔍 {profile['name']} 파일 감지 → 필요 컬럼만 유지")
        df = slim_df(df, profile)
    else:
        print(f"\n  ℹ️  데이터셋 자동 감지 실패 → 전체 컬럼 유지")

    # 저장
    out_path = input_path.replace(".csv", "_grouped.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n  ✅ 저장 완료: {os.path.basename(out_path)}  ({size_mb:.1f} MB)")

    return df


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="국가명 통일 & A/B/C/OTHER 그룹 레이블 부여 (전 세계)"
    )
    parser.add_argument(
        "files", nargs="*",                      # "+" → "*" : 인자 없어도 오류 안 남
        help="처리할 CSV 파일 경로 (여러 개 가능)"
    )
    parser.add_argument(
        "--col", default=None,
        help="국가명 컬럼명 직접 지정 (기본: 자동 감지)"
    )
    args = parser.parse_args()

    print(f"✅ 정의 로딩 완료")
    print(f"   분석 그룹 대상: {len(CANONICAL)}개국  /  alias 항목: {len(ALIAS_MAP)}개")
    print(f"   나머지 국가: OTHER (학습 데이터에 포함)")

    # 인자 없이 실행한 경우 → 대화형으로 경로 입력받기
    file_list = args.files
    if not file_list:
        print("\n파일 경로를 입력하세요 (여러 개면 쉼표로 구분, 엔터로 완료):")
        raw = input("  >> ").strip()
        if not raw:
            print("❌ 파일 경로가 입력되지 않았습니다.")
            return
        # 쉼표 또는 공백으로 구분, 따옴표 제거
        file_list = [p.strip().strip('"').strip("'") for p in raw.replace(",", " ").split()]

    for fpath in file_list:
        if not os.path.exists(fpath):
            print(f"\n❌ 파일 없음: {fpath}")
            continue
        if not fpath.lower().endswith(".csv"):
            print(f"\n❌ CSV 파일만 지원합니다: {fpath}")
            continue
        try:
            process_file(fpath, country_col=args.col)
        except Exception as e:
            print(f"\n❌ 오류: {e}")

    print(f"\n{'='*60}")
    print("  전체 처리 완료")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
