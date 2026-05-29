#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <windows.h>
#include <algorithm>
#include <cmath>

#include "GdeltData.h"
#include "AcledData.h"
#include "VDemData.h"
#include "HistogramViewer.h"
#include "BoxPlotViewer.h"
#include "ScatterViewer.h"
#include "LineChartViewer.h"
#include "InteractiveGroupViewer.h"
#include "EscalationViewer.h"
#include "ToneDropViewer.h"
#include "DeathCrossViewer.h"

#include <vtkRenderWindowInteractor.h>
#include <vtkAutoInit.h>
#include <vtkPen.h>
#include <vtkAxis.h>

VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingContextOpenGL2);

// ============================================================
// 중앙값(Median) 계산 함수 (이상치 탐지용)
// ============================================================
static double CalculateMedian(std::vector<double> data) {
    if (data.empty()) return 0.0;
    std::sort(data.begin(), data.end());
    size_t n = data.size();
    if (n % 2 == 0) {
        return (data[n / 2 - 1] + data[n / 2]) / 2.0;
    }
    else {
        return data[n / 2];
    }
}

// ============================================================
// MAD (Median Absolute Deviation) 기반 이상치 탐지 및 출력 함수
// ============================================================
static void DetectOutliersMAD(std::vector<double>& data, const std::string& variableName, double threshold = 3.0) {
    if (data.empty()) {
        std::cout << "[" << variableName << "] 데이터가 없습니다.\n";
        return;
    }

    double median = CalculateMedian(data);

    std::vector<double> absoluteDeviations;
    absoluteDeviations.reserve(data.size());
    for (double val : data) {
        absoluteDeviations.push_back(std::abs(val - median));
    }

    double mad = CalculateMedian(absoluteDeviations);
    double scaledMad = mad * 1.4826;

    std::cout << "\n========================================\n";
    std::cout << " [" << variableName << "] 통계적 이상치 분석 (MAD) \n";
    std::cout << "========================================\n";
    std::cout << "- 데이터 개수: " << data.size() << "\n";
    std::cout << "- 중앙값(Median): " << median << "\n";
    std::cout << "- MAD(스케일 적용): " << scaledMad << "\n";

    if (scaledMad == 0.0) {
        std::cout << " -> 데이터 변동성이 너무 적어(MAD=0) 이상치 탐지가 어렵습니다.\n";
        return;
    }

    int outlierCount = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        double modifiedZScore = std::abs(data[i] - median) / scaledMad;

        if (modifiedZScore > threshold) {
            outlierCount++;
            if (outlierCount <= 5) {
                std::cout << " -> ⚠️ 이상치 발견: 값 = " << data[i]
                    << " (Modified Z-Score: " << modifiedZScore << ")\n";
            }
        }
    }

    if (outlierCount > 5) {
        std::cout << " -> ... 외 " << (outlierCount - 5) << "개의 이상치가 더 존재합니다.\n";
    }
    std::cout << " -> 📌 총 이상치 개수: " << outlierCount << "개 ("
        << (double)outlierCount / data.size() * 100.0 << "%)\n";
}

// ============================================================
// 버그1 수정: 큰따옴표 필드 안의 쉼표를 올바르게 처리하는 CSV 파서
// ============================================================
static std::vector<std::string> ParseCSVLine(const std::string& line) {
    std::vector<std::string> result;
    std::string field;
    bool inQuotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"';
                ++i;
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

// ==========================================================
//  사전(Dictionary) 세팅
// ==========================================================
std::unordered_map<std::string, std::string> CANONICAL = {
    {"Syria", "A"}, {"Yemen", "A"}, {"Somalia", "A"}, {"Myanmar", "A"},
    {"Ethiopia", "A"}, {"South Sudan", "A"}, {"Mali", "A"}, {"DR Congo", "A"},
    {"Ukraine", "A"}, {"Iraq", "A"},
    {"Pakistan", "B"}, {"Nigeria", "B"}, {"Venezuela", "B"}, {"Sudan", "B"},
    {"Central African Republic", "B"}, {"Taiwan", "B"}, {"Haiti", "B"},
    {"Lebanon", "B"}, {"Colombia", "B"}, {"Ecuador", "B"},
    {"Norway", "C"}, {"Switzerland", "C"}, {"Japan", "C"}, {"South Korea", "C"},
    {"Portugal", "C"}, {"Uruguay", "C"}, {"Botswana", "C"}, {"Mongolia", "C"},
    {"Canada", "C"}, {"Germany", "C"},
    {"Afghanistan", "TARGET"}
};

std::unordered_map<std::string, std::string> ALIAS_MAP = {
    {"syria", "Syria"}, {"syrian arab republic", "Syria"}, {"syr", "Syria"}, {"sy", "Syria"},
    {"yemen", "Yemen"}, {"yemen, rep.", "Yemen"}, {"republic of yemen", "Yemen"}, {"yem", "Yemen"}, {"ym", "Yemen"},
    {"somalia", "Somalia"}, {"som", "Somalia"}, {"so", "Somalia"},
    {"myanmar", "Myanmar"}, {"myanmar (burma)", "Myanmar"}, {"burma", "Myanmar"}, {"mmr", "Myanmar"}, {"mya", "Myanmar"}, {"bm", "Myanmar"},
    {"ethiopia", "Ethiopia"}, {"eth", "Ethiopia"}, {"et", "Ethiopia"},
    {"south sudan", "South Sudan"}, {"s. sudan", "South Sudan"}, {"ssd", "South Sudan"}, {"od", "South Sudan"},
    {"mali", "Mali"}, {"mli", "Mali"}, {"ml", "Mali"},
    {"dr congo", "DR Congo"}, {"dr congo (zaire)", "DR Congo"}, {"congo (the democratic republic of the)", "DR Congo"},
    {"democratic republic of the congo", "DR Congo"}, {"democratic republic of congo", "DR Congo"}, {"congo, dem. rep.", "DR Congo"},
    {"congo, democratic republic", "DR Congo"}, {"drc", "DR Congo"}, {"zaire", "DR Congo"}, {"cod", "DR Congo"}, {"cg", "DR Congo"},
    {"ukraine", "Ukraine"}, {"ukr", "Ukraine"}, {"up", "Ukraine"},
    {"iraq", "Iraq"}, {"irq", "Iraq"}, {"iz", "Iraq"},
    {"pakistan", "Pakistan"}, {"pak", "Pakistan"}, {"pk", "Pakistan"},
    {"nigeria", "Nigeria"}, {"nga", "Nigeria"}, {"nig", "Nigeria"}, {"ni", "Nigeria"},
    {"venezuela", "Venezuela"}, {"venezuela, rb", "Venezuela"}, {"bolivarian republic of venezuela", "Venezuela"}, {"ven", "Venezuela"}, {"ve", "Venezuela"},
    {"sudan", "Sudan"}, {"sdn", "Sudan"}, {"sud", "Sudan"}, {"su", "Sudan"},
    {"central african republic", "Central African Republic"}, {"car", "Central African Republic"}, {"caf", "Central African Republic"}, {"ct", "Central African Republic"},
    {"taiwan", "Taiwan"}, {"taiwan, province of china", "Taiwan"}, {"twn", "Taiwan"}, {"tw", "Taiwan"},
    {"haiti", "Haiti"}, {"hti", "Haiti"}, {"ha", "Haiti"},
    {"lebanon", "Lebanon"}, {"lbn", "Lebanon"}, {"leb", "Lebanon"}, {"le", "Lebanon"},
    {"colombia", "Colombia"}, {"col", "Colombia"}, {"co", "Colombia"},
    {"ecuador", "Ecuador"}, {"ecu", "Ecuador"}, {"ec", "Ecuador"},
    {"norway", "Norway"}, {"nor", "Norway"}, {"no", "Norway"},
    {"switzerland", "Switzerland"}, {"che", "Switzerland"}, {"sui", "Switzerland"}, {"sz", "Switzerland"},
    {"japan", "Japan"}, {"jpn", "Japan"}, {"ja", "Japan"},
    {"south korea", "South Korea"}, {"korea, south", "South Korea"}, {"korea, rep.", "South Korea"},
    {"korea (the republic of)", "South Korea"}, {"republic of korea", "South Korea"}, {"kor", "South Korea"}, {"ks", "South Korea"},
    {"portugal", "Portugal"}, {"prt", "Portugal"}, {"po", "Portugal"},
    {"uruguay", "Uruguay"}, {"ury", "Uruguay"}, {"uy", "Uruguay"},
    {"botswana", "Botswana"}, {"bwa", "Botswana"}, {"bc", "Botswana"},
    {"mongolia", "Mongolia"}, {"mng", "Mongolia"}, {"mg", "Mongolia"},
    {"canada", "Canada"}, {"can", "Canada"}, {"ca", "Canada"},
    {"germany", "Germany"}, {"deu", "Germany"}, {"ger", "Germany"}, {"gm", "Germany"},
    {"afghanistan", "Afghanistan"}, {"afg", "Afghanistan"}, {"af", "Afghanistan"},
    {"united states", "USA"}, {"united states of america", "USA"}, {"usa", "USA"}, {"us", "USA"},
    {"mexico", "Mexico"}, {"mex", "Mexico"}, {"mx", "Mexico"},
    {"brazil", "Brazil"}, {"bra", "Brazil"}, {"br", "Brazil"},
    {"argentina", "Argentina"}, {"arg", "Argentina"}, {"ar", "Argentina"},
    {"chile", "Chile"}, {"chl", "Chile"}, {"ci", "Chile"},
    {"peru", "Peru"}, {"per", "Peru"}, {"pe", "Peru"},
    {"cuba", "Cuba"}, {"cub", "Cuba"}, {"cu", "Cuba"},
    {"united kingdom", "United Kingdom"}, {"uk", "United Kingdom"}, {"gbr", "United Kingdom"}, {"gb", "United Kingdom"},
    {"france", "France"}, {"fra", "France"}, {"fr", "France"},
    {"russia", "Russia"}, {"russian federation", "Russia"}, {"rus", "Russia"}, {"rs", "Russia"}, {"ru", "Russia"},
    {"italy", "Italy"}, {"ita", "Italy"}, {"it", "Italy"},
    {"spain", "Spain"}, {"esp", "Spain"}, {"sp", "Spain"},
    {"poland", "Poland"}, {"pol", "Poland"}, {"pl", "Poland"},
    {"netherlands", "Netherlands"}, {"nld", "Netherlands"}, {"nl", "Netherlands"},
    {"sweden", "Sweden"}, {"swe", "Sweden"}, {"sw", "Sweden"},
    {"greece", "Greece"}, {"grc", "Greece"}, {"gr", "Greece"},
    {"china", "China"}, {"chn", "China"}, {"ch", "China"}, {"cn", "China"},
    {"india", "India"}, {"ind", "India"}, {"in", "India"},
    {"indonesia", "Indonesia"}, {"idn", "Indonesia"}, {"id", "Indonesia"},
    {"philippines", "Philippines"}, {"phl", "Philippines"}, {"rp", "Philippines"}, {"ph", "Philippines"},
    {"australia", "Australia"}, {"aus", "Australia"}, {"as", "Australia"}, {"au", "Australia"},
    {"new zealand", "New Zealand"}, {"nzl", "New Zealand"}, {"nz", "New Zealand"},
    {"north korea", "North Korea"}, {"prk", "North Korea"}, {"kn", "North Korea"}, {"korea, north", "North Korea"},
    {"vietnam", "Vietnam"}, {"vnm", "Vietnam"}, {"vm", "Vietnam"}, {"vn", "Vietnam"},
    {"thailand", "Thailand"}, {"tha", "Thailand"}, {"th", "Thailand"},
    {"malaysia", "Malaysia"}, {"mys", "Malaysia"}, {"my", "Malaysia"},
    {"saudi arabia", "Saudi Arabia"}, {"sau", "Saudi Arabia"}, {"sa", "Saudi Arabia"},
    {"egypt", "Egypt"}, {"egy", "Egypt"}, {"eg", "Egypt"},
    {"turkey", "Turkey"}, {"tur", "Turkey"}, {"tu", "Turkey"}, {"tr", "Turkey"}, {"turkiye", "Turkey"},
    {"iran", "Iran"}, {"irn", "Iran"}, {"ir", "Iran"}, {"islamic republic of iran", "Iran"},
    {"israel", "Israel"}, {"isr", "Israel"}, {"is", "Israel"}, {"il", "Israel"},
    {"united arab emirates", "UAE"}, {"are", "UAE"}, {"ae", "UAE"}, {"uae", "UAE"},
    {"algeria", "Algeria"}, {"dza", "Algeria"}, {"ag", "Algeria"}, {"dz", "Algeria"},
    {"morocco", "Morocco"}, {"mar", "Morocco"}, {"mo", "Morocco"},
    {"south africa", "South Africa"}, {"zaf", "South Africa"}, {"sf", "South Africa"}, {"za", "South Africa"},
    {"kenya", "Kenya"}, {"ken", "Kenya"}, {"ke", "Kenya"},
    {"uganda", "Uganda"}, {"uga", "Uganda"}, {"ug", "Uganda"},
    {"angola", "Angola"}, {"ago", "Angola"}, {"ao", "Angola"}
};

std::string Trim(const std::string& s) {
    if (s.empty()) return "";
    size_t first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, (last - first + 1));
}

std::string ToLowerCase(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

int main() {
    SetConsoleOutputCP(CP_UTF8);

    // 파일 경로 지정
    std::string csvPath = "..\\..\\GDELT\\GDELT_2013_2024_Cleaned.csv";
    std::string csvPath_ACLED = "..\\..\\ACLED\\Continental\\ACLED_2013_2024.csv";
    std::string csvPath_Vdem = "..\\..\\V-Dem\\VDemData_2013_2024_Cleaned.csv";

    // ── ACLED 파일 열기 ──────────────────────────────────────
    std::ifstream file_Acled(csvPath_ACLED);
    if (!file_Acled.is_open()) {
        std::cerr << "File Open Failed: 경로를 다시 확인해 주세요!\n"
            << csvPath_ACLED << std::endl;
        return -1;
    }

    // ── GDELT 파일 열기 ──────────────────────────────────────
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "File Open Failed!" << std::endl;
        return -1;
    }

    // VDem 파일 열기
    std::ifstream file_Vdem(csvPath_Vdem);
    if (!file_Vdem.is_open()) {
        std::cerr << "File Open Failed!" << std::endl;
        return -1;
    }

    // ==========================================================
    //  미리보기(Preview) 부분
    // ==========================================================

    // ── ACLED Preview ─────────────────────────────────────────
    // 실제 컬럼 순서:
    //  [0]event_id_cnty [1]event_date [2]year [3]event_type [4]sub_event_type
    //  [5]interaction   [6]fatalities [7]latitude [8]longitude
    //  [9]country       [10]country_std [11]group
    std::string line_Acled;

    if (std::getline(file_Acled, line_Acled)) {
        std::cout << "========================================\n";
        std::cout << " Header Preview - ACLED \n";
        std::cout << "========================================\n";

        std::vector<std::string> headers = ParseCSVLine(line_Acled);
        for (int i = 0; i < (int)headers.size(); ++i) {
            std::cout << "[" << i << "] " << headers[i] << "  |  ";
        }
        std::cout << "\n\n";
    }

    std::cout << "========================================\n";
    std::cout << " Data Preview - ACLED \n";
    std::cout << "========================================\n";

    for (int i = 0; i < 3; ++i) {
        if (std::getline(file_Acled, line_Acled)) {
            std::cout << "Row " << i + 1 << ":\n";
            std::vector<std::string> row = ParseCSVLine(line_Acled);
            for (int j = 0; j < (int)row.size(); ++j) {
                if (row[j].empty())
                    std::cout << "[" << j << "] (Empty)  |  ";
                else
                    std::cout << "[" << j << "] " << row[j] << "  |  ";
            }
            std::cout << "\n----------------------------------------\n";
        }
    }

    // ── GDELT Preview ─────────────────────────────────────────
    std::string line;

    if (std::getline(file, line)) {
        std::cout << "========================================\n";
        std::cout << " Header Preview - GDELT \n";
        std::cout << "========================================\n";

        std::stringstream ss(line);
        std::string cell;
        int colIdx = 0;
        while (std::getline(ss, cell, ',')) {
            std::cout << "[" << colIdx << "] " << cell << "  |  ";
            colIdx++;
        }
        std::cout << "\n\n";
    }

    std::cout << "========================================\n";
    std::cout << " Data Preview - GDELT\n";
    std::cout << "========================================\n";

    for (int i = 0; i < 3; ++i) {
        if (std::getline(file, line)) {
            std::cout << "Row " << i + 1 << ":\n";
            std::stringstream ss(line);
            std::string cell;
            int colIdx = 0;
            while (std::getline(ss, cell, ',')) {
                if (cell.empty())
                    std::cout << "[" << colIdx << "] (Empty)  |  ";
                else
                    std::cout << "[" << colIdx << "] " << cell << "  |  ";
                colIdx++;
            }
            std::cout << "\n----------------------------------------\n";
        }
    }

    // VDem Preview
    std::string line_Vdem;

    if (std::getline(file_Vdem, line_Vdem)) {
        std::cout << "========================================\n";
        std::cout << " Header Preview - V-Dem \n";
        std::cout << "========================================\n";

        std::vector<std::string> headers = ParseCSVLine(line_Vdem);
        for (int i = 0; i < (int)headers.size(); ++i) {
            std::cout << "[" << i << "] " << headers[i] << "  |  ";
        }
        std::cout << "\n\n";
    }

    std::cout << "========================================\n";
    std::cout << " Data Preview - V-Dem \n";
    std::cout << "========================================\n";

    for (int i = 0; i < 3; ++i) {
        if (std::getline(file_Vdem, line_Vdem)) {
            std::cout << "Row " << i + 1 << ":\n";
            std::vector<std::string> row = ParseCSVLine(line_Vdem);
            for (int j = 0; j < (int)row.size(); ++j) {
                if (row[j].empty())
                    std::cout << "[" << j << "] (Empty)  |  ";
                else
                    std::cout << "[" << j << "] " << row[j] << "  |  ";
            }
            std::cout << "\n----------------------------------------\n";
        }
    }

    // ==========================================================
    //  구조체 포인터 선언
    // ==========================================================
    std::vector<GdeltPoint> rawData;
    std::vector<AcledPoint> rawData_Acled;
    std::vector<VDemPoint> rawData_Vdem;

    std::map<std::string, std::map<std::string, int>> acledMissingMap;
    std::map<std::string, std::map<std::string, int>> vdemMissingMap;
    std::map<std::string, std::map<std::string, int>> gdeltMissingMap;

    // ==========================================================
    //  Loading 부분
    // ==========================================================

    // ── GDELT Loading ─────────────────────────────────────────
    std::getline(file, line); // Header skip

    std::cout << "Loading GDELT..." << std::endl;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;

        while (std::getline(ss, cell, ',')) row.push_back(cell);

        if (row.size() > 7) {
            try {
                GdeltPoint pt;

                pt.sqlDate = row[0];
                pt.countryCode = row[1];
                pt.eventCode = row[2];

                std::string c_name = pt.countryCode.empty() ? "UNKNOWN" : pt.countryCode;

                // 1. AvgGoldstein
                try {
                    if (!row[3].empty()) pt.goldstein = std::stod(row[3]);
                    else gdeltMissingMap[c_name]["AvgGoldstein"]++;
                }
                catch (...) { gdeltMissingMap[c_name]["AvgGoldstein"]++; }

                // 2. AvgTone
                try {
                    if (!row[4].empty()) pt.avgTone = std::stod(row[4]);
                    else gdeltMissingMap[c_name]["AvgTone"]++;
                }
                catch (...) { gdeltMissingMap[c_name]["AvgTone"]++; }

                // 3. TotalMentions
                try {
                    if (!row[5].empty()) pt.totalMentions = std::stoi(row[5]);
                    else gdeltMissingMap[c_name]["TotalMentions"]++;
                }
                catch (...) { gdeltMissingMap[c_name]["TotalMentions"]++; }

                // 4. TotalSources
                try {
                    if (!row[6].empty()) pt.totalSources = std::stoi(row[6]);
                    else gdeltMissingMap[c_name]["TotalSources"]++;
                }
                catch (...) { gdeltMissingMap[c_name]["TotalSources"]++; }

                // 5. TotalArticles
                try {
                    if (!row[7].empty()) pt.totalArticles = std::stoi(row[7]);
                    else gdeltMissingMap[c_name]["TotalArticles"]++;
                }
                catch (...) { gdeltMissingMap[c_name]["TotalArticles"]++; }

                rawData.push_back(pt);
            }
            catch (...) {
                continue;
            }
        }
    }

    file.close();
    std::cout << "Load GDELT Complete! (" << rawData.size() << " points)" << std::endl;

    // ── ACLED Loading ─────────────────────────────────────────
    file_Acled.close();
    file_Acled.open(csvPath_ACLED);
    if (!file_Acled.is_open()) {
        std::cerr << "ACLED 재오픈 실패!" << std::endl;
        return -1;
    }
    std::getline(file_Acled, line_Acled); // 헤더 정확히 1번만 스킵

    std::cout << "Loading ACLED..." << std::endl;

    while (std::getline(file_Acled, line_Acled)) {
        std::vector<std::string> row = ParseCSVLine(line_Acled);

        // ✅ 수정: 새로운 헤더는 최소 13개 컬럼(인덱스 0~12)을 가집니다.
        if (row.size() > 12) {
            AcledPoint pt;

            // ── 문자열 필드 (새로운 인덱스 적용) ─────────────────────────
            pt.week = row[0];          // [0] WEEK
            pt.country = row[2];       // [2] COUNTRY
            pt.eventType = row[4];     // [4] EVENT_TYPE

            std::string c_name = pt.country.empty() ? "UNKNOWN" : pt.country;

            // ── 숫자 필드: 개별 try-catch (새로운 인덱스 적용) ───────────

            // 1. FATALITIES [7]
            try {
                if (!row[7].empty()) pt.fatalities = std::stoi(row[7]);
                else acledMissingMap[c_name]["FATALITIES"]++;
            }
            catch (...) { acledMissingMap[c_name]["FATALITIES"]++; }

            // 2. LATITUDE (CENTROID_LATITUDE) [11]
            try {
                if (!row[11].empty()) pt.latitude = std::stod(row[11]);
                else acledMissingMap[c_name]["LATITUDE"]++;
            }
            catch (...) { acledMissingMap[c_name]["LATITUDE"]++; }

            // 3. LONGITUDE (CENTROID_LONGITUDE) [12]
            try {
                if (!row[12].empty()) pt.longitude = std::stod(row[12]);
                else acledMissingMap[c_name]["LONGITUDE"]++;
            }
            catch (...) { acledMissingMap[c_name]["LONGITUDE"]++; }

            // (선택) AcledPoint 구조체에 events 필드가 있다면 [6]번 인덱스로 파싱
            // try { if (!row[6].empty()) pt.events = std::stoi(row[6]); } catch (...) {}
            // try { if (!row[8].empty()) pt.populationExposure = std::stod(row[8]); } catch (...) {}

            rawData_Acled.push_back(pt);
        }
    }

    file_Acled.close();
    std::cout << "Load ACLED Complete! (" << rawData_Acled.size() << " points)" << std::endl;

    // ── V-Dem Loading ─────────────────────────────────────────
    file_Vdem.close();
    file_Vdem.open(csvPath_Vdem);
    if (!file_Vdem.is_open()) {
        std::cerr << "V-Dem 재오픈 실패!" << std::endl;
        return -1;
    }
    std::getline(file_Vdem, line_Vdem); // 헤더 정확히 1번만 스킵

    std::cout << "Loading V-Dem..." << std::endl;

    while (std::getline(file_Vdem, line_Vdem)) {
        std::vector<std::string> row = ParseCSVLine(line_Vdem);

        // ✅ 수정 포인트 1: 2개의 열이 삭제되었으므로, 최소 12개(인덱스 0~11)로 변경
        if (row.size() > 11) {
            VDemPoint pt;

            pt.country_name = row[0];
            pt.country_text_id = row[1];

            std::string c_name = pt.country_name.empty() ? "UNKNOWN" : pt.country_name;

            try {
                if (!row[2].empty()) pt.year = std::stoi(row[2]);
                else vdemMissingMap[c_name]["year"]++;
            }
            catch (...) { vdemMissingMap[c_name]["year"]++; }

            try {
                if (!row[3].empty()) pt.v2elpeace = std::stod(row[3]);
                else vdemMissingMap[c_name]["v2elpeace"]++;
            }
            catch (...) { vdemMissingMap[c_name]["v2elpeace"]++; }

            try {
                if (!row[4].empty()) pt.v2x_rule = std::stod(row[4]);
                else vdemMissingMap[c_name]["v2x_rule"]++;
            }
            catch (...) { vdemMissingMap[c_name]["v2x_rule"]++; }

            try {
                if (!row[5].empty()) pt.v2x_clphy = std::stod(row[5]);
                else vdemMissingMap[c_name]["v2x_clphy"]++;
            }
            catch (...) { vdemMissingMap[c_name]["v2x_clphy"]++; }

            // ✅ 수정 포인트 2: 삭제된 e_pt_coup, e_civil_war는 껍데기만 0으로 채움
            pt.e_pt_coup = 0.0;
            pt.e_civil_war = 0.0;

            // ✅ 수정 포인트 3: 인덱스가 2씩 앞으로 당겨짐 (8->6, 9->7 ... 13->11)
            try {
                if (!row[6].empty()) pt.v2x_libdem = std::stod(row[6]);
                else vdemMissingMap[c_name]["v2x_libdem"]++;
            }
            catch (...) { vdemMissingMap[c_name]["v2x_libdem"]++; }

            try {
                if (!row[7].empty()) pt.v2x_corr = std::stod(row[7]);
                else vdemMissingMap[c_name]["v2x_corr"]++;
            }
            catch (...) { vdemMissingMap[c_name]["v2x_corr"]++; }

            try {
                if (!row[8].empty()) pt.v2x_veracc = std::stod(row[8]);
                else vdemMissingMap[c_name]["v2x_veracc"]++;
            }
            catch (...) { vdemMissingMap[c_name]["v2x_veracc"]++; }

            try {
                if (!row[9].empty()) pt.v2xcs_ccsi = std::stod(row[9]);
                else vdemMissingMap[c_name]["v2xcs_ccsi"]++;
            }
            catch (...) { vdemMissingMap[c_name]["v2xcs_ccsi"]++; }

            try {
                if (!row[10].empty()) pt.v2x_polyarchy = std::stod(row[10]);
                else vdemMissingMap[c_name]["v2x_polyarchy"]++;
            }
            catch (...) { vdemMissingMap[c_name]["v2x_polyarchy"]++; }

            try {
                if (!row[11].empty()) pt.v2elintim = std::stod(row[11]);
                else vdemMissingMap[c_name]["v2elintim"]++;
            }
            catch (...) { vdemMissingMap[c_name]["v2elintim"]++; }

            rawData_Vdem.push_back(pt);
        }
    }

    file_Vdem.close();
    std::cout << "Load V-Dem Complete! (" << rawData_Vdem.size() << " points)" << std::endl;

    // ==========================================================
    //  결측치(Missing Values) 국가별 / 변수별 최종 출력 보고서
    // ==========================================================
    std::cout << "\n========================================\n";
    std::cout << " ACLED 결측치 통계 \n";
    std::cout << "========================================\n";
    if (acledMissingMap.empty()) {
        std::cout << " -> 발견된 결측치 없음.\n";
    }
    else {
        for (const auto& countryPair : acledMissingMap) {
            std::cout << "[" << countryPair.first << "] ";
            bool first = true;
            for (const auto& varPair : countryPair.second) {
                if (!first) std::cout << ", ";
                std::cout << varPair.first << ": " << varPair.second;
                first = false;
            }
            std::cout << "\n";
        }
    }

    std::cout << "\n========================================\n";
    std::cout << " V-Dem 결측치 통계 \n";
    std::cout << "========================================\n";
    if (vdemMissingMap.empty()) {
        std::cout << " -> 발견된 결측치 없음.\n";
    }
    else {
        for (const auto& countryPair : vdemMissingMap) {
            std::cout << "[" << countryPair.first << "] ";
            bool first = true;
            for (const auto& varPair : countryPair.second) {
                if (!first) std::cout << ", ";
                std::cout << varPair.first << ": " << varPair.second;
                first = false;
            }
            std::cout << "\n";
        }
    }

    std::cout << "\n========================================\n";
    std::cout << " GDELT 결측치 통계 \n";
    std::cout << "========================================\n";
    if (gdeltMissingMap.empty()) {
        std::cout << " -> 발견된 결측치 없음.\n";
    }
    else {
        for (const auto& countryPair : gdeltMissingMap) {
            std::string code = countryPair.first;
            std::string lowerCode = ToLowerCase(code);
            if (ALIAS_MAP.find(lowerCode) != ALIAS_MAP.end()) {
                std::cout << "[" << code << ":" << ALIAS_MAP[lowerCode] << "] ";
            }
            else {
                std::cout << "[" << code << "] ";
            }

            bool first = true;
            for (const auto& varPair : countryPair.second) {
                if (!first) std::cout << ", ";
                std::cout << varPair.first << ": " << varPair.second;
                first = false;
            }
            std::cout << "\n";
        }
    }

    std::cout << "========================================\n\n";

    std::cout << "Load ACLED Complete! (" << rawData_Acled.size() << " points)" << std::endl;
    std::cout << "Load V-Dem Complete! (" << rawData_Vdem.size() << " points)" << std::endl;

    // ==========================================================
    // 주요 수치형 변수에 대한 MAD 기반 통계적 이상치 탐지
    // ==========================================================
    std::cout << "\n[데이터 이상치(Outlier) 탐지 시작]\n";

    // 1. GDELT: AvgGoldstein, TotalArticles
    std::vector<double> gdelt_goldstein, gdelt_articles;
    for (const auto& pt : rawData) {
        gdelt_goldstein.push_back(pt.goldstein);
        gdelt_articles.push_back(pt.totalArticles);
    }
    DetectOutliersMAD(gdelt_goldstein, "GDELT: AvgGoldstein", 3.0);
    DetectOutliersMAD(gdelt_articles, "GDELT: TotalArticles", 3.0);

    auto view_hist_goldstein = ShowHistogram(rawData, 20, -10.0, 10.0);
    auto view_outlier_gdelt = ShowOutlierScatterPlot(gdelt_articles, "GDELT: TotalArticles");

    // 2. ACLED: fatalities 검사
    vtkSmartPointer<vtkContextView> view_outlier_acled = nullptr;
    vtkSmartPointer<vtkContextView> view_hist_acled = nullptr;
    vtkSmartPointer<vtkContextView> view_box_acled = nullptr;
    if (!rawData_Acled.empty()) {
        std::vector<double> acled_fatalities;
        for (const auto& pt : rawData_Acled) {
            acled_fatalities.push_back(pt.fatalities);
        }
        DetectOutliersMAD(acled_fatalities, "ACLED: Fatalities", 3.0);

        view_outlier_acled = ShowOutlierScatterPlot(acled_fatalities, "ACLED: Fatalities");
        view_hist_acled = ShowHistogram(acled_fatalities, "ACLED: Fatalities Distribution", 50);
        view_box_acled = ShowBoxPlot_Fatalities(rawData_Acled, "ACLED: Fatalities");
    }

    // ==========================================================
    //  파이프라인 가동
    // ==========================================================
    std::vector<std::string> codesA = { "SYR", "YEM", "SOM", "MMR", "ETH", "SSD", "MLI", "COD", "UKR", "IRQ" };
    std::vector<std::string> namesA = { "Syria", "Yemen", "Somalia", "Myanmar", "Ethiopia", "South Sudan", "Mali", "DR Congo", "Ukraine", "Iraq" };

    std::vector<std::string> codesB = { "PAK", "NGA", "VEN", "SDN", "CAF", "TWN", "HTI", "LBN", "COL", "ECU" };
    std::vector<std::string> namesB = { "Pakistan", "Nigeria", "Venezuela", "Sudan", "Central African Rep", "Taiwan", "Haiti", "Lebanon", "Colombia", "Ecuador" };

    std::vector<std::string> codesC = { "NOR", "CHE", "JPN", "KOR", "PRT", "URY", "BWA", "MNG", "CAN", "DEU" };
    std::vector<std::string> namesC = { "Norway", "Switzerland", "Japan", "South Korea", "Portugal", "Uruguay", "Botswana", "Mongolia", "Canada", "Germany" };
    /*
    auto viewA = ShowInteractiveGroupChart(rawData_Vdem, "[Group A]", codesA, namesA);
    auto viewB = ShowInteractiveGroupChart(rawData_Vdem, "[Group B]", codesB, namesB);
    auto viewC = ShowInteractiveGroupChart(rawData_Vdem, "[Group C]", codesC, namesC);

    auto view_tone_sy = ShowToneDropChart(rawData, "SY", 14);
    auto view_tone_bm = ShowToneDropChart(rawData, "BM", 14);
    auto view_tone_su = ShowToneDropChart(rawData, "SU", 14);
    auto view_tone_et = ShowToneDropChart(rawData, "ET", 14);

    auto view_deathcross_sy = ShowDeathCrossChart(rawData, "SY");
    auto view_deathcross_bm = ShowDeathCrossChart(rawData, "BM");
    auto view_deathcross_su = ShowDeathCrossChart(rawData, "SU");
    auto view_deathcross_et = ShowDeathCrossChart(rawData, "ET");

    auto view_esc = ShowEscalationChart(rawData_Acled, "Syria");
    auto view_esc_mmr = ShowEscalationChart(rawData_Acled, "Myanmar");
    auto view_esc_sdn = ShowEscalationChart(rawData_Acled, "Sudan");
    auto view_esc_eth = ShowEscalationChart(rawData_Acled, "Ethiopia");

    auto view_scatterview = ShowScatterPlot(rawData_Acled);

    std::vector<std::string> targetXVars = {
        "EVENTS",
        "EVENT_TYPE",
        "SUB_EVENT_TYPE",
        "DISORDER_TYPE"
    };

    auto view_corr = ShowInteractiveScatterPlot(rawData_Acled, targetXVars);
    */
    if (view_outlier_gdelt != nullptr) {
        std::cout << "[1/4] GDELT: TotalArticles 이상치(Outlier) 차트를 띄웁니다.\n";
        view_outlier_gdelt->GetInteractor()->Start();
    }

    if (view_hist_goldstein != nullptr) {
        std::cout << "[2/4] GDELT: AvgGoldstein 히스토그램을 띄웁니다.\n";
        view_hist_goldstein->GetInteractor()->Start();
    }

    if (view_hist_acled != nullptr) {
        std::cout << "[3/4] ACLED: Fatalities 히스토그램을 띄웁니다.\n";
        view_hist_acled->GetInteractor()->Start();
    }

    if (view_box_acled != nullptr) {
        std::cout << "[4/4] ACLED: Fatalities 박스플롯을 띄웁니다.\n";
        view_box_acled->GetInteractor()->Start();
    }

    return 0;
}