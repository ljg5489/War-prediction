#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>     // std::log1p 사용
#include <windows.h> // 한글 출력용

// ── CSV 파서 ──────────────────────────────────────────────
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

// ── TotalArticles 로그 변환 및 CSV 저장 함수 ────────────────
void ApplyLogToGDELT(const std::string& inputPath, const std::string& outputPath) {
    std::ifstream inFile(inputPath);
    if (!inFile.is_open()) {
        std::cerr << "파일 열기 실패: " << inputPath << "\n";
        return;
    }

    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        std::cerr << "파일 생성 실패: " << outputPath << "\n";
        inFile.close();
        return;
    }

    std::string line;
    // 1. 헤더 복사
    if (std::getline(inFile, line)) {
        outFile << line << "\n";
    }

    int totalRows = 0;
    int successRows = 0;
    int errorRows = 0;

    std::cout << "GDELT TotalArticles 로그 변환 작업 시작...\n";

    // 2. 데이터 순회
    while (std::getline(inFile, line)) {
        if (line.empty()) continue;
        totalRows++;

        std::vector<std::string> row = ParseCSVLine(line);

        // GDELT 컬럼은 최소 8개(인덱스 0~7)가 있어야 함. [7]이 TotalArticles
        if (row.size() > 7) {
            try {
                if (!row[7].empty()) {
                    // 원본 값을 읽어서 로그(log(1+x)) 변환
                    double originalValue = std::stod(row[7]);
                    double logValue = std::log1p(originalValue);

                    // 변환된 값으로 덮어쓰기
                    row[7] = std::to_string(logValue);
                }

                // 다시 쉼표(,)로 연결하여 파일에 쓰기
                for (size_t i = 0; i < row.size(); ++i) {
                    outFile << row[i];
                    if (i < row.size() - 1) outFile << ",";
                }
                outFile << "\n";
                successRows++;
            }
            catch (...) {
                errorRows++;
            }
        }
        else {
            errorRows++;
        }

        // 진행 상황 표시 (데이터가 많으므로 100만 줄 단위로 출력)
        if (totalRows % 1000000 == 0) {
            std::cout << "   ... " << totalRows / 1000000 << "백만 행 처리 완료\n";
        }
    }

    inFile.close();
    outFile.close();

    std::cout << "변환 완료: 총 " << totalRows << "행 중 " << successRows << "행 저장 (에러/누락: " << errorRows << "행)\n";
    std::cout << "저장된 파일: " << outputPath << "\n\n";
}

int main() {
    SetConsoleOutputCP(CP_UTF8);

    // 원본 파일 경로 (2013~2024 필터링이 끝난 파일)
    std::string gdeltIn = "..\\..\\GDELT\\GDELT_2013_2024_Cleaned_Grouped.csv";

    // 로그 변환이 적용되어 새롭게 저장될 파일 경로
    std::string gdeltOut = "..\\..\\GDELT\\GDELT_2013_2024_Cleaned_Grouped_Log.csv";

    ApplyLogToGDELT(gdeltIn, gdeltOut);

    return 0;
}