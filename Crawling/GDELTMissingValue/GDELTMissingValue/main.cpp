#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <windows.h>

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

// ── CSV 저장용 (필드에 쉼표가 있으면 따옴표로 다시 묶어줌) ─────────────
static std::string JoinCSVLine(const std::vector<std::string>& row) {
    std::string result;
    for (size_t i = 0; i < row.size(); ++i) {
        std::string field = row[i];
        if (field.find(',') != std::string::npos) {
            result += "\"" + field + "\"";
        }
        else {
            result += field;
        }
        if (i < row.size() - 1) result += ",";
    }
    return result;
}

// ── GDELT 중립값 보간 및 새 CSV 저장 함수 ────────────────────────
void CleanAndSaveGDELT(const std::string& inputPath, const std::string& outputPath) {
    std::ifstream inFile(inputPath);
    if (!inFile.is_open()) {
        std::cerr << "file not found: " << inputPath << "\n";
        return;
    }

    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        std::cerr << "file not create " << outputPath << "\n";
        inFile.close();
        return;
    }

    std::string line;
    // 1. 헤더 복사 및 저장
    if (std::getline(inFile, line)) {
        outFile << line << "\n";
    }

    int totalRows = 0;
    int successRows = 0;
    int imputedCount = 0; // 보간(0.0 채우기)이 발생한 횟수 카운트

    std::cout << "Create Start...\n";

    while (std::getline(inFile, line)) {
        if (line.empty()) continue;
        totalRows++;

        std::vector<std::string> row = ParseCSVLine(line);

        // GDELT 헤더는 최소 8개 (인덱스 0~7)
        // [0]sqlDate [1]countryCode [2]eventCode [3]AvgGoldstein [4]AvgTone [5]TotalMentions [6]TotalSources [7]TotalArticles
        if (row.size() > 7) {

            // 🌟 보고서 논리 적용: AvgGoldstein(인덱스 3)이 비어있다면 행을 삭제하지 않고 "0.0" 투입!
            if (row[3].empty()) {
                row[3] = "0.0";
                imputedCount++;
            }

            // 완성된 행을 다시 쉼표로 조립하여 새 파일에 쓰기
            outFile << JoinCSVLine(row) << "\n";
            successRows++;
        }

        // 진행 상황 표시 (100만 줄 단위)
        if (totalRows % 1000000 == 0) {
            std::cout << "   ... " << totalRows / 1000000 << "00Millions processed (missing value interpolated so far: " << imputedCount << "case)\n";
        }
    }

    inFile.close();
    outFile.close();

    std::cout << "Complate!\n";
    std::cout << "- Find ROW: " << totalRows << "행\n";
    std::cout << "- SAVE ROW: " << successRows << "행 (유실된 시그널 없음!)\n";
    std::cout << "- 0.0 change Missing Value : " << imputedCount << "개\n";
    std::cout << "save file: " << outputPath << "\n\n";
}

int main() {
    SetConsoleOutputCP(CP_UTF8);

    // 원본 GDELT 파일 경로
    std::string gdeltIn = "..\\..\\GDELT\\GDELT_2013_2024.csv";

    // 결측치가 0.0으로 완벽하게 채워져서 생성될 새 CSV 파일 경로
    std::string gdeltOut = "..\\..\\GDELT\\GDELT_2013_2024_Cleaned.csv";

    CleanAndSaveGDELT(gdeltIn, gdeltOut);

    return 0;
}