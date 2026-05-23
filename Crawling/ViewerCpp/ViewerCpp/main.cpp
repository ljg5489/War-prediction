#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <windows.h>

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

VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingContextOpenGL2);

// ============================================================
// ✅ 버그1 수정: 큰따옴표 필드 안의 쉼표를 올바르게 처리하는 CSV 파서
//    기존 std::getline(ss, cell, ',') 방식은 "Somalia, South" 같은
//    필드 내부의 쉼표를 컬럼 구분자로 오인하여 인덱스를 밀어버림.
// ============================================================
static std::vector<std::string> ParseCSVLine(const std::string& line) {
    std::vector<std::string> result;
    std::string field;
    bool inQuotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            // "" 형태의 이스케이프된 따옴표 처리
            if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"';
                ++i;
            }
            else {
                inQuotes = !inQuotes; // 따옴표 구역 진입/탈출
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
    result.push_back(field); // 마지막 필드 추가
    return result;
}

int main() {
    // 파일 경로 지정
    std::string csvPath = "..\\..\\GDELT\\GDELT_Daily_All_Countries.csv";
    std::string csvPath_ACLED = "..\\..\\ACLED\\Continental\\ACLED_All_Countries.csv";
    std::string csvPath_Vdem = "..\\..\\V-Dem\\VDemData_All_Countries.csv";

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
    std::string line_Acled;

    if (std::getline(file_Acled, line_Acled)) {
        std::cout << "========================================\n";
        std::cout << " Header Preview - ACLED \n";
        std::cout << "========================================\n";

        // ✅ ParseCSVLine 사용: 따옴표 안 쉼표에도 안전
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

        // ✅ ParseCSVLine 사용: 따옴표 안 쉼표에도 안전
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

    // 결측치 카운트 변수
    int GdeltmissingValue = 0;
    int AcledmissingValue = 0;
    int VdemmissingValue = 0;

    // ==========================================================
    //  Loading 부분
    // ==========================================================

    // ── GDELT Loading ─────────────────────────────────────────
    /*std::getline(file, line); // Header skip

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

                if (!row[3].empty()) pt.goldstein = std::stod(row[3]);
                if (!row[4].empty()) pt.avgTone = std::stod(row[4]);
                if (!row[5].empty()) pt.totalMentions = std::stoi(row[5]);
                if (!row[6].empty()) pt.totalSources = std::stoi(row[6]);
                if (!row[7].empty()) pt.totalArticles = std::stoi(row[7]);

                rawData.push_back(pt);
            }
            catch (...) {
                continue;
            }
        }
    }

    file.close();
    std::cout << "Load GDELT Complete! (" << rawData.size() << " points)" << std::endl;
    */

    // ── ACLED Loading ─────────────────────────────────────────
    // ✅ 버그2 수정: Preview 단계에서 이미 헤더 + 데이터 3줄을 읽어버렸으므로
    //    파일을 닫고 다시 열어 헤더를 정확히 1번만 스킵합니다.
    file_Acled.close();
    file_Acled.open(csvPath_ACLED);
    if (!file_Acled.is_open()) {
        std::cerr << "ACLED 재오픈 실패!" << std::endl;
        return -1;
    }
    std::getline(file_Acled, line_Acled); // 헤더 정확히 1번만 스킵

    std::cout << "Loading ACLED..." << std::endl;

    while (std::getline(file_Acled, line_Acled)) {
        // ✅ 버그1 수정: 따옴표 인식 CSV 파서 사용
        std::vector<std::string> row = ParseCSVLine(line_Acled);

        if (row.size() > 12) {
            AcledPoint pt;

            // 문자열 필드
            pt.week = row[0];
            pt.country = row[2];
            pt.eventType = row[4];

            // 숫자 필드: 개별 try-catch로 한 줄 전체를 버리지 않음
            // 1. 발생 건수 (EVENTS)
            try {
                if (!row[6].empty()) pt.events = std::stoi(row[6]);
                else AcledmissingValue++;
            }
            catch (...) { AcledmissingValue++; }

            // 2. 사망자 수 (FATALITIES)
            try {
                if (!row[7].empty()) pt.fatalities = std::stoi(row[7]);
                else AcledmissingValue++;
            }
            catch (...) { AcledmissingValue++; }

            // 3. 노출 인구 (POPULATION_EXPOSURE) - 누락되었던 부분 추가!
            try {
                if (!row[8].empty()) pt.populationExposure = std::stoi(row[8]);
                else AcledmissingValue++;
            }
            catch (...) { AcledmissingValue++; }

            // 4. 위도 (LATITUDE)
            try {
                if (!row[11].empty()) pt.latitude = std::stod(row[11]);
                else AcledmissingValue++;
            }
            catch (...) { AcledmissingValue++; }

            // 5. 경도 (LONGITUDE)
            try {
                if (!row[12].empty()) pt.longitude = std::stod(row[12]);
                else AcledmissingValue++;
            }
            catch (...) { AcledmissingValue++; }

            // 위도·경도가 정상 범위일 때만 저장
            if (pt.latitude != 0.0 && pt.longitude != 0.0) {
                rawData_Acled.push_back(pt);
            }
        }
    }

    file_Acled.close();

    // ✅ 로드된 점 개수 확인용 디버그 출력 (0이면 CSV 컬럼 인덱스 재확인 필요)
    std::cout << "Load ACLED Complete! (" << rawData_Acled.size() << " points)" << std::endl;
    std::cout << "Missing Value ACLED : " << AcledmissingValue << std::endl;



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
        // ✅ 버그1 수정: 따옴표 인식 CSV 파서 사용
        std::vector<std::string> row = ParseCSVLine(line_Vdem);

        if (row.size() > 12) {
            VDemPoint pt;

            // 1. 문자열 (그냥 넣음)
            pt.country_name = row[0];
            pt.country_text_id = row[1];

            // 2. 숫자는 모조리 try-catch로 개별 방어
            try { if (!row[2].empty()) pt.year = std::stoi(row[2]); }
            catch (...) {}
            try { if (!row[3].empty()) pt.v2elpeace = std::stod(row[3]); }
            catch (...) {}
            try { if (!row[4].empty()) pt.v2x_rule = std::stod(row[4]); }
            catch (...) {}
            try { if (!row[5].empty()) pt.v2x_clphy = std::stod(row[5]); }
            catch (...) {}
            try { if (!row[6].empty()) pt.e_pt_coup = std::stod(row[6]); }
            catch (...) {}
            try { if (!row[7].empty()) pt.e_civil_war = std::stod(row[7]); }
            catch (...) {} // 비어있어도 무사 통과
            try { if (!row[8].empty()) pt.v2x_libdem = std::stod(row[8]); }
            catch (...) {}
            try { if (!row[9].empty()) pt.v2x_corr = std::stod(row[9]); }
            catch (...) {}
            try { if (!row[10].empty()) pt.v2x_veracc = std::stod(row[10]); }
            catch (...) {}
            try { if (!row[11].empty()) pt.v2xcs_ccsi = std::stod(row[11]); }
            catch (...) {}
            try { if (!row[12].empty()) pt.v2x_polyarchy = std::stod(row[12]); }
            catch (...) {}
            try { if (!row[13].empty()) pt.v2elintim = std::stod(row[13]); }
            catch (...) {}

            rawData_Vdem.push_back(pt);
        }
    }

    file_Vdem.close();

    // ✅ 로드된 점 개수 확인용 디버그 출력 (0이면 CSV 컬럼 인덱스 재확인 필요)
    std::cout << "Load V-Dem Complete! (" << rawData_Vdem.size() << " points)" << std::endl;


    // ==========================================================
    //  파이프라인 가동
    // ==========================================================
    // GEDELT
    /*
    auto view1 = ShowHistogram(rawData, 50, -10.0, 10.0);
    auto view2 = ShowBoxPlot(rawData, "Goldstein Scale");
    auto view3 = ShowScatterPlot(rawData, "Goldstein Scale");
    auto view4 = ShowLineChart(rawData, "Average Tone");
    auto view_map = ShowScatterPlot(rawData_Acled);
    */
    
    // V-Dem에 대한 시각화(그룹별로 창을 나눠 방향키로 탭바꾸기)
    //  Group A: 심각한 분쟁 및 위기 국가 (High Conflict)
    std::vector<std::string> codesA = { "SYR", "YEM", "SOM", "MMR", "ETH", "SSD", "MLI", "COD", "UKR", "IRQ" };
    std::vector<std::string> namesA = { "Syria", "Yemen", "Somalia", "Myanmar", "Ethiopia", "South Sudan", "Mali", "DR Congo", "Ukraine", "Iraq" };

    //  Group B: 중간 단계 및 잠재적 불안정 국가 (At-Risk)
    std::vector<std::string> codesB = { "PAK", "NGA", "VEN", "SDN", "CAF", "TWN", "HTI", "LBN", "COL", "ECU" };
    std::vector<std::string> namesB = { "Pakistan", "Nigeria", "Venezuela", "Sudan", "Central African Rep", "Taiwan", "Haiti", "Lebanon", "Colombia", "Ecuador" };

    //  Group C: 안정적인 민주주의 국가 (Stable)
    std::vector<std::string> codesC = { "NOR", "CHE", "JPN", "KOR", "PRT", "URY", "BWA", "MNG", "CAN", "DEU" };
    std::vector<std::string> namesC = { "Norway", "Switzerland", "Japan", "South Korea", "Portugal", "Uruguay", "Botswana", "Mongolia", "Canada", "Germany" };

    // 3개의 독립적인 상호작용 창 생성 - V-DEM
    auto viewA = ShowInteractiveGroupChart(rawData_Vdem, "[Group A]", codesA, namesA);
    auto viewB = ShowInteractiveGroupChart(rawData_Vdem, "[Group B]", codesB, namesB);
    auto viewC = ShowInteractiveGroupChart(rawData_Vdem, "[Group C]", codesC, namesC);

    // 미디어 톤 급락 - GDELT
    auto view_tone_sy = ShowToneDropChart(rawData, "SY", 14); // 시리아
    auto view_tone_bm = ShowToneDropChart(rawData, "BM", 14); // 미얀마
    auto view_tone_su = ShowToneDropChart(rawData, "SU", 14); // 수단 추가
    auto view_tone_et = ShowToneDropChart(rawData, "ET", 14); // 에티오피아 추가

    // 데스 크로스 확인 - GDELT
    auto view_deathcross_sy = ShowDeathCrossChart(rawData, "SY");
    auto view_deathcross_bm = ShowDeathCrossChart(rawData, "BM");
    auto view_deathcross_su = ShowDeathCrossChart(rawData, "SU");
    auto view_deathcross_et = ShowDeathCrossChart(rawData, "ET");

    // 분쟁 데이터가 가장 풍부한 시리아(Syria)를 예시로 에스컬레이션 속도 측정 - ACLED
    auto view_esc = ShowEscalationChart(rawData_Acled, "Syria");
    auto view_esc_mmr = ShowEscalationChart(rawData_Acled, "Myanmar");   // 쿠데타발 급가속
    auto view_esc_sdn = ShowEscalationChart(rawData_Acled, "Sudan");     // 군벌 간 전면전 폭발
    auto view_esc_eth = ShowEscalationChart(rawData_Acled, "Ethiopia");  // 지역 갈등의 전쟁화

    auto view_scatterview = ShowScatterPlot(rawData_Acled); // 위도, 경도 지도 그리기 

    // 🌟 보고 싶은 X축 변수들의 이름을 리스트로 묶습니다.
    std::vector<std::string> targetXVars = {
        "EVENTS",                // 발생 건수 (규모)
        "POPULATION_EXPOSURE",   // 노출 인구 (밀집도 리스크)
        "EVENT_TYPE",            // 상위 사건 분류
        "SUB_EVENT_TYPE",        // 상세 사건 분류 (강력 추천)
        "DISORDER_TYPE"          // 정치적 폭력 성격
    };

    // 🌟 데이터와 리스트를 함께 던져서 인터랙티브 뷰어를 호출합니다.
    auto view_corr = ShowInteractiveScatterPlot(rawData_Acled, targetXVars);

    if (view_esc != nullptr) {
        view_esc->GetInteractor()->Start();
    }

    return 0;
}