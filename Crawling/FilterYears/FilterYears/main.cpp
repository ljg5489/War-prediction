#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <windows.h> // 한글 출력용

// ── CSV 파서 (따옴표 안 쉼표 무시) ──────────────────────────────
static std::vector<std::string> ParseCSVLine(const std::string& line) {
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

// ── 데이터 필터링 공통 함수 ─────────────────────────────────────────
void FilterDataset(const std::string& inputPath, const std::string& outputPath, const std::string& datasetName, int yearColIndex, bool isGdelt) {
    std::ifstream inFile(inputPath);
    if (!inFile.is_open()) {
        std::cerr << "❌ 파일 열기 실패: " << inputPath << "\n";
        return;
    }

    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        std::cerr << "❌ 파일 생성 실패: " << outputPath << "\n";
        inFile.close();
        return;
    }

    std::string line;
    // 1. 헤더 복사
    if (std::getline(inFile, line)) {
        outFile << line << "\n";
    }
    else {
        std::cerr << "❌ 파일이 비어있습니다: " << inputPath << "\n";
        return;
    }

    int totalRows = 0;
    int savedRows = 0;
    int errorRows = 0;

    std::cout << "🚀 [" << datasetName << "] 2013~2024년 데이터 필터링 시작...\n";

    // 2. 데이터 순회 및 연도 체크
    while (std::getline(inFile, line)) {
        if (line.empty()) continue;
        totalRows++;

        std::vector<std::string> row = ParseCSVLine(line);
        if (row.size() <= static_cast<size_t>(yearColIndex)) {
            errorRows++;
            continue;
        }

        int year = 0;
        try {
            if (isGdelt) {
                // GDELT는 SQLDATE 형식이 "YYYYMMDD" 이므로 앞 4자리만 추출
                std::string dateStr = row[yearColIndex];
                if (dateStr.length() >= 4) {
                    year = std::stoi(dateStr.substr(0, 4));
                }
            }
            else {
                // V-Dem과 ACLED는 단독 "YYYY" 형식 컬럼 보유
                year = std::stoi(row[yearColIndex]);
            }

            // 🌟 2013년 이상, 2024년 이하인 데이터만 저장
            if (year >= 2013 && year <= 2024) {
                outFile << line << "\n";
                savedRows++;
            }
        }
        catch (...) {
            // 변환 실패(결측치나 쓰레기값) 시 저장하지 않음
            errorRows++;
        }

        // 진행 상황 표시
        if (totalRows % 1000000 == 0) {
            std::cout << "   ... " << totalRows / 1000000 << "백만 행 처리 중\n";
        }
    }

    inFile.close();
    outFile.close();

    std::cout << "✅ [" << datasetName << "] 완료: 총 " << totalRows << "행 중 " << savedRows << "행 추출 (에러/누락: " << errorRows << "행)\n\n";
}

int main() {
    SetConsoleOutputCP(CP_UTF8);

    // =========================================================
    // 📂 파일 경로 설정 (원본 파일과 저장될 새 파일의 경로를 맞춰주세요)
    // =========================================================
    //std::string vdemIn = "..\\..\\V-Dem\\VDemData_All_Countries.csv"; // Grouped 파일 기준
    //std::string vdemOut = "..\\..\\V-Dem\\VDemData_2013_2024.csv";

    std::string acledIn = "..\\..\\ACLED\\Continental\\ACLED_All_Countries.csv";
    std::string acledOut = "..\\..\\ACLED\\Continental\\ACLED_2013_2024.csv";

    //std::string gdeltIn = "..\\..\\GDELT\\GDELT_Daily_All_Countries_Grouped.csv";
    //std::string gdeltOut = "..\\..\\GDELT\\GDELT_2013_2024.csv";

    // =========================================================
    // ⚙️ 필터링 실행
    // 인자: 입력경로, 출력경로, 데이터셋이름, 연도컬럼인덱스, GDELT여부
    // =========================================================

    // 1. V-Dem 필터링 (컬럼 인덱스 2번: 'year')
    //FilterDataset(vdemIn, vdemOut, "V-Dem", 2, false);

    // 2. ACLED 필터링 (컬럼 인덱스 2번: 'year')
    FilterDataset(acledIn, acledOut, "ACLED", 2, false);

    // 3. GDELT 필터링 (컬럼 인덱스 0번: 'SQLDATE', GDELT 특수 파싱 적용)
    //FilterDataset(gdeltIn, gdeltOut, "GDELT", 0, true);

    std::cout << "🎉 모든 데이터셋의 2013~2024 필터링 작업이 완료되었습니다!\n";

    return 0;
}