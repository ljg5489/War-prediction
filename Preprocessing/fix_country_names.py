"""
fix_country_names.py
──────────────────────────────────────────────────────────────────────────────
국가명 표기 통일 (전 소스 → 깨끗한 표준명) - 대용량 메모리 절약(Chunk) 버전

문제: UCDP/GED·ACLED·EM-DAT·UNHCR·REIGN 이 서로 다른 국가명 규칙을 써서
      JOIN 시 같은 나라가 안 붙는다. 특히
        - UNHCR 'Syrian Arab Rep.'  ≠  'Syria'        → 시리아 난민/IDP 유실
        - GED   'Yemen (North Yemen)' 가 OTHER 로 분류  → 예멘이 A그룹에서 누락
        - 'Iran (Islamic Republic of)' / 'Türkiye' / 'Russia (Soviet Union)' 등

이 모듈은 두 가지를 제공한다.
  (1) ALIAS_ADDITIONS  : normalize_and_group.py 의 ALIAS_MAP 에 합쳐 영구 수정
  (2) CLI 교정 도구     : 이미 grouped 된 CSV/Parquet 의 country_std·group 를
                          깨끗한 표준명으로 일괄 교정 (파이프라인 재실행 없이 즉시 정합)

깨끗한 표준명을 타깃으로 하므로, 적용 대상을 '모두'에 동일하게 적용해야 한다:
    wri_labeled_fixed.csv, acled_weekly.parquet, emdat/unhcr/reign grouped 파일

사용법:
    python fix_country_names.py wri_labeled_fixed.csv
    python fix_country_names.py acled_weekly.parquet unhcr_grouped.csv reign_grouped.csv
    python fix_country_names.py --print-aliases       # ALIAS_MAP 패치 블록만 출력

출력: 입력파일명_fixed.csv / _fixed.parquet (country_std·group 교정)
──────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import argparse
import os

# ── (1) ALIAS 추가분: 변형 표기(소문자) → 깨끗한 표준명 ───────────────────────
#     normalize_and_group.py 에서:  ALIAS_MAP.update(ALIAS_ADDITIONS)
ALIAS_ADDITIONS = {
    # Iran
    "iran (islamic republic of)": "Iran", "iran (islamic rep. of)": "Iran",
    "iran, islamic rep.": "Iran",
    # Turkey
    "türkiye": "Turkey", "turkiye": "Turkey",
    # Venezuela
    "venezuela (bolivarian republic of)": "Venezuela", "venezuela, rb": "Venezuela",
    "bolivarian republic of venezuela": "Venezuela",
    # Tanzania
    "united republic of tanzania": "Tanzania", "united rep. of tanzania": "Tanzania",
    # Bolivia
    "bolivia (plurinational state of)": "Bolivia",
    # Netherlands
    "netherlands (kingdom of the)": "Netherlands",
    # United Kingdom
    "united kingdom of great britain and northern ireland": "United Kingdom",
    # Syria
    "syrian arab rep.": "Syria", "syrian arab republic": "Syria",
    # Ivory Coast (직선/곡선 아포스트로피 모두; lookup 전 ’→' 정규화)
    "cote d'ivoire": "Ivory Coast", "côte d'ivoire": "Ivory Coast",
    # eSwatini
    "eswatini": "Eswatini", "kingdom of eswatini (swaziland)": "Eswatini",
    "swaziland": "Eswatini",
    # Madagascar  (GED 괄호표기 → clean)
    "madagascar (malagasy)": "Madagascar",
    # Russia
    "russia": "Russia", "russian federation": "Russia",
    "soviet union": "Russia", "russia (soviet union)": "Russia",
    # Yemen  (GED 'Yemen (North Yemen)' 가 OTHER 로 새던 것 교정)
    "yemen": "Yemen", "yemen (north yemen)": "Yemen",
    "republic of yemen": "Yemen", "yemen, rep.": "Yemen",
    # Zimbabwe
    "zimbabwe (rhodesia)": "Zimbabwe",
    # Central African Republic
    "central african rep.": "Central African Republic",
    # South Korea
    "rep. of korea": "South Korea", "republic of korea": "South Korea",
    "korea, rep.": "South Korea", "korea (the republic of)": "South Korea",
    # DR Congo
    "dem. rep. of the congo": "DR Congo", "democratic republic of the congo": "DR Congo",
    "congo, dem. rep.": "DR Congo", "dr congo (zaire)": "DR Congo",
}

# ── A/B/C 그룹 재산정용 (normalize_and_group.CANONICAL 과 동일, 깨끗한 표준명) ──
GROUP_MAP = {
    # A
    "Syria": "A", "Yemen": "A", "Somalia": "A", "Myanmar": "A", "Ethiopia": "A",
    "South Sudan": "A", "Mali": "A", "DR Congo": "A", "Ukraine": "A", "Iraq": "A",
    # B
    "Pakistan": "B", "Nigeria": "B", "Venezuela": "B", "Sudan": "B",
    "Central African Republic": "B", "Taiwan": "B", "Haiti": "B", "Lebanon": "B",
    "Colombia": "B", "Ecuador": "B",
    # C
    "Norway": "C", "Switzerland": "C", "Japan": "C", "South Korea": "C",
    "Portugal": "C", "Uruguay": "C", "Botswana": "C", "Mongolia": "C",
    "Canada": "C", "Germany": "C",
}


def to_canonical(name):
    if pd.isna(name):
        return name
    key = str(name).strip().replace("\u2019", "'").lower()   # ’ → '
    return ALIAS_ADDITIONS.get(key, str(name).strip())


def fix_file(path: str):
    print(f"\n{'='*60}\n  {os.path.basename(path)}\n{'='*60}")
    is_pq = path.lower().endswith(".parquet")
    out = path.replace(".parquet", "_fixed.parquet") if is_pq else path.replace(".csv", "_fixed.csv")

    n_changed = 0
    changed_pairs = set()
    unique_countries = set()

    if is_pq:
        # Parquet 파일 처리 (기존 압축 및 칼럼형 최적화가 되어 있어 단일 로드 유지)
        df = pd.read_parquet(path)
        if "country_std" not in df.columns:
            print("  ❌ country_std 컬럼 없음 — 스킵")
            return

        before = df["country_std"].copy()
        df["country_std"] = df["country_std"].map(to_canonical)
        n_changed = int((before != df["country_std"]).sum())

        mask = before != df["country_std"]
        if mask.any():
            for b, a in zip(before[mask], df["country_std"][mask]):
                changed_pairs.add((b, a))

        if "group" in df.columns:
            new_group = df["country_std"].map(lambda c: GROUP_MAP.get(c))
            df["group"] = new_group.fillna(df["group"]).fillna("OTHER")
        else:
            df["group"] = df["country_std"].map(lambda c: GROUP_MAP.get(c, "OTHER"))

        unique_countries.update(df["country_std"].dropna().unique())
        df.to_parquet(out, index=False)

    else:
        # 🔴 핵심 수정: CSV 파일의 경우 50,000행씩 청크 단위 분할 처리 (OOM 차단)
        chunk_size = 50000
        first_chunk = True
        
        # 컬럼 존재 여부만 가볍게 선행 확인하기 위해 헤더만 읽기
        header_df = pd.read_csv(path, nrows=0)
        if "country_std" not in header_df.columns:
            print("  ❌ country_std 컬럼 없음 — 스킵")
            return

        print(f"  [알림] 대용량 CSV 파일 스트리밍 분석을 시작합니다. ({chunk_size:,}행씩 분할 처리)")
        
        # 분할 순차 분석 루프
        for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
            before_chunk = chunk["country_std"].copy()
            chunk["country_std"] = chunk["country_std"].map(to_canonical)
            n_changed += int((before_chunk != chunk["country_std"]).sum())

            # 변경 이력 수집
            mask = before_chunk != chunk["country_std"]
            if mask.any():
                for b, a in zip(before_chunk[mask], chunk["country_std"][mask]):
                    changed_pairs.add((b, a))

            # 그룹 재산정
            if "group" in chunk.columns:
                new_group = chunk["country_std"].map(lambda c: GROUP_MAP.get(c))
                chunk["group"] = new_group.fillna(chunk["group"]).fillna("OTHER")
            else:
                chunk["group"] = chunk["country_std"].map(lambda c: GROUP_MAP.get(c, "OTHER"))

            unique_countries.update(chunk["country_std"].dropna().unique())

            # 🔴 실시간 파일 이어붙이기 저장
            if first_chunk:
                chunk.to_csv(out, index=False, encoding="utf-8-sig", mode='w')
                first_chunk = False
            else:
                chunk.to_csv(out, index=False, encoding="utf-8-sig", mode='a', header=False)

    # 교정 리포트 출력 (청크 병합 통계)
    print(f"  표준명 교정 행: {n_changed:,}  /  바뀐 고유표기: {len(changed_pairs)}개")
    if len(changed_pairs):
        # 식별 편의성을 위해 정렬 후 상위 20개 매핑 관계 출력
        for b, a in sorted(list(changed_pairs))[:20]:
            grp = GROUP_MAP.get(a, "OTHER")
            print(f"    {str(b)[:34]:<36} → {a:<26} [{grp}]")

    print(f"  ✅ 저장: {os.path.basename(out)}  ({len(unique_countries)}개국)")
    return out


def print_aliases():
    print("# normalize_and_group.py 의 ALIAS_MAP 정의 직후에 추가:")
    print("ALIAS_MAP.update({")
    for k, v in ALIAS_ADDITIONS.items():
        print(f'    {k!r}: {v!r},')
    print("})")


def main():
    p = argparse.ArgumentParser(description="국가명 표기 통일 (전 소스 → 깨끗한 표준명)")
    p.add_argument("files", nargs="*", help="교정할 grouped CSV/Parquet (여러 개 가능)")
    p.add_argument("--print-aliases", action="store_true",
                   help="ALIAS_MAP 패치 블록만 출력 (영구 수정용)")
    args = p.parse_args()

    if args.print_aliases:
        print_aliases(); return

    files = args.files
    if not files:
        print("교정할 파일 경로(쉼표/공백 구분):")
        raw = input("  >> ").strip()
        files = [x.strip().strip('"').strip("'") for x in raw.replace(",", " ").split()]

    for f in files:
        if os.path.exists(f):
            fix_file(f)
        else:
            print(f"❌ 파일 없음: {f}")

    print("\n📌 영구 수정: python fix_country_names.py --print-aliases  → 출력을 "
          "normalize_and_group.py 의 ALIAS_MAP 에 추가")


if __name__ == "__main__":
    main()