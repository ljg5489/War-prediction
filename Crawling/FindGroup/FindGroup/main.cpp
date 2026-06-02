// 소스 코드 상단에 항상 파일 위치와 파일 명을 함께 주석으로 삽입할 것
// 파일 위치: C:\viewer\
// 파일 명: GDELT_Group_Counter.cpp

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <windows.h>

// ── 1. 문자열 앞뒤 공백 및 따옴표 제거 ────────────────────────────────
std::string Trim(const std::string& s) {
    if (s.empty()) return "";
    size_t first = s.find_first_not_of(" \t\r\n\"");
    if (first == std::string::npos) return "";
    size_t last = s.find_last_not_of(" \t\r\n\"");
    return s.substr(first, (last - first + 1));
}

// ── 2. CSV 파서 (따옴표 안의 쉼표 무시) ──────────────────────────────
std::vector<std::string> ParseCSVLine(const std::string& line) {
    std::vector<std::string> result;
    std::string field;
    bool inQuotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"'; ++i;
            }
            else {
                inQuotes = !inQuotes;
            }
        }
        else if (c == ',' && !inQuotes) {
            result.push_back(field);
            field.clear();
        }
        else {
            field += c;
        }
    }
    result.push_back(field);
    return result;
}

int main() {
    // 콘솔 한글 출력 지원
    SetConsoleOutputCP(CP_UTF8);

    // ── 타겟 파일 경로 ──
    std::string inputFilePath = "..\\..\\GDELT\\GDELT_2013_2024_Cleaned_Grouped.csv";

    std::ifstream inFile(inputFilePath);
    if (!inFile.is_open()) {
        std::cerr << "❌ 파일 열기 실패. 경로를 확인해 주세요: " << inputFilePath << "\n";
        return -1;
    }

    std::string headerLine;
    if (!std::getline(inFile, headerLine)) {
        std::cerr << "❌ 파일이 비어있습니다.\n";
        return -1;
    }

    // 헤더에서 'country_std'와 'group' 컬럼 인덱스 찾기
    std::vector<std::string> headers = ParseCSVLine(headerLine);
    int countryIdx = -1;
    int groupIdx = -1;

    for (size_t i = 0; i < headers.size(); ++i) {
        std::string h = Trim(headers[i]);
        if (h == "country_std") countryIdx = static_cast<int>(i);
        else if (h == "group") groupIdx = static_cast<int>(i);
    }

    if (countryIdx == -1 || groupIdx == -1) {
        std::cerr << "❌ 오류: 파일에 'country_std' 또는 'group' 컬럼이 존재하지 않습니다.\n";
        return -1;
    }

    // 통계를 저장할 이중 맵: map<그룹명, map<국가명, 카운트>>
    std::map<std::string, std::map<std::string, int>> group_country_counts;
    int totalRows = 0;

    std::cout << "데이터 분석 중...\n";

    // 데이터 읽기 및 집계
    std::string line;
    while (std::getline(inFile, line)) {
        if (line.empty()) continue;
        totalRows++;

        std::vector<std::string> row = ParseCSVLine(line);

        // ✅ 완벽한 해결: std::max에 괄호를 씌워 매크로 충돌 원천 차단 -> (std::max)
        if (row.size() > static_cast<size_t>((std::max)(countryIdx, groupIdx))) {
            std::string country = Trim(row[countryIdx]);
            std::string group = Trim(row[groupIdx]);

            if (group.empty()) group = "UNKNOWN";
            if (country.empty()) country = "UNKNOWN";

            // 해당 그룹의 해당 국가 카운트 1 증가
            group_country_counts[group][country]++;
        }
    }
    inFile.close();

    // ── 결과 출력 ──
    std::cout << "\n========================================\n";
    std::cout << " 그룹별 인식된 국가 목록 및 데이터 건수\n";
    std::cout << "========================================\n";
    std::cout << " 📌 총 스캔된 데이터: " << totalRows << " 행\n";

    // 출력할 그룹 순서 지정
    std::vector<std::string> group_order = { "A", "B", "C", "OTHER", "UNKNOWN" };

    for (const auto& g : group_order) {
        if (group_country_counts.find(g) != group_country_counts.end()) {
            std::cout << "\n[" << g << " Group]\n";

            // 건수 기준으로 내림차순 정렬하기 위해 vector로 복사
            std::vector<std::pair<std::string, int>> sorted_countries(
                group_country_counts[g].begin(),
                group_country_counts[g].end()
            );

            // C++11 호환성을 위해 auto 대신 명시적 타입(std::pair) 사용
            std::sort(sorted_countries.begin(), sorted_countries.end(),
                [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
                    return a.second > b.second; // 카운트 기준 내림차순
                }
            );

            int groupTotal = 0;
            // 모든 나라 출력
            for (const auto& pair : sorted_countries) {
                std::cout << "    - " << pair.first << " : " << pair.second << " 행\n";
                groupTotal += pair.second;
            }
            std::cout << "    ------------------------------------\n";
            std::cout << "    >> " << g << " 그룹 소계: " << groupTotal << " 행\n";
        }
    }

    std::cout << "\n모든 분석이 완료되었습니다.\n";
    return 0;
}