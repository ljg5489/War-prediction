#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
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

// ── V-Dem 전처리 및 새 CSV 저장 함수 ────────────────────────────
// ── V-Dem 전처리 및 새 CSV 저장 함수 ────────────────────────────
void CleanAndSaveVDem(const std::string& inputPath, const std::string& outputPath) {
    std::ifstream inFile(inputPath);
    if (!inFile.is_open()) {
        std::cerr << "file not found: " << inputPath << "\n";
        return;
    }

    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        std::cerr << "file not found: " << outputPath << "\n";
        inFile.close();
        return;
    }

    std::string line;

    // 1. 헤더 처리 (e_pt_coup, e_civil_war 컬럼 완전 삭제)
    if (std::getline(inFile, line)) {
        std::vector<std::string> headers = ParseCSVLine(line);
        std::vector<std::string> newHeaders;

        for (size_t i = 0; i < headers.size(); ++i) {
            // 인덱스 6(e_pt_coup)과 7(e_civil_war)은 버림
            if (i == 6 || i == 7) continue;
            newHeaders.push_back(headers[i]);
        }
        outFile << JoinCSVLine(newHeaders) << "\n";
    }

    // 🌟 Forward Fill(전진 채우기)을 위한 국가별 이전 값 저장소
    std::map<std::string, std::string> last_v2elpeace;
    std::map<std::string, std::string> last_v2elintim;

    int totalRows = 0;
    int successRows = 0;

    std::cout << "START Creating...\n";

    while (std::getline(inFile, line)) {
        if (line.empty()) continue;
        totalRows++;

        std::vector<std::string> row = ParseCSVLine(line);

        // V-Dem 헤더는 최소 14개 (인덱스 0~13)
        if (row.size() > 13) {
            std::string c_name = row[0]; // 국가명

            // 1) v2elpeace [인덱스 3]: 선거 평화성 (Forward Fill, 없으면 최악치 1.0)
            if (!row[3].empty()) {
                last_v2elpeace[c_name] = row[3];
            }
            else {
                if (last_v2elpeace.find(c_name) != last_v2elpeace.end()) {
                    row[3] = last_v2elpeace[c_name];
                }
                else {
                    row[3] = "1.0"; // 독재국가 초기화
                }
            }

            // 2) v2elintim [인덱스 13]: 선거 위협 (Forward Fill, 없으면 최악치 1.0)
            if (!row[13].empty()) {
                last_v2elintim[c_name] = row[13];
            }
            else {
                if (last_v2elintim.find(c_name) != last_v2elintim.end()) {
                    row[13] = last_v2elintim[c_name];
                }
                else {
                    row[13] = "1.0"; // 독재국가 초기화
                }
            }

            // 🌟 완성된 행 조립 (인덱스 6과 7을 제외하고 새 파일에 쓰기)
            std::vector<std::string> newRow;
            for (size_t i = 0; i < row.size(); ++i) {
                // 인덱스 6(e_pt_coup)과 7(e_civil_war) 데이터는 버림
                if (i == 6 || i == 7) continue;
                newRow.push_back(row[i]);
            }

            outFile << JoinCSVLine(newRow) << "\n";
            successRows++;
        }
    }

    inFile.close();
    outFile.close();

    std::cout << "Complete!\n";
    std::cout << "save file: " << outputPath << "\n\n";
}

int main() {
    SetConsoleOutputCP(CP_UTF8);

    // 원본 V-Dem 파일 경로
    std::string vdemIn = "..\\..\\V-Dem\\VDemData_2013_2024.csv";

    // 결측치가 완벽하게 채워져서 생성될 새 CSV 파일 경로
    std::string vdemOut = "..\\..\\V-Dem\\VDemData_2013_2024_Cleaned.csv";

    CleanAndSaveVDem(vdemIn, vdemOut);

    return 0;
}