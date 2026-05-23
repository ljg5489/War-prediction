#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <cctype>

// ── 1. 문자열 유틸리티 함수 ──────────────────────────────────────────
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

// ── 3. 사전(Dictionary) 세팅 ─────────────────────────────────────────
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

std::vector<std::string> CANDIDATE_COLS = {
    "ActionGeo_CountryCode", "SQLDATE", "EventCode", "AvgGoldstein", "AvgTone",
    "TotalMentions", "TotalSources", "TotalArticles",
    "country", "cname", "country_name", "Country", "COUNTRY",
    "country_text_id", "CountryName", "nation", "state",
    "Actor1CountryCode", "Actor2CountryCode"
};

// ── 4. 처리 로직 ─────────────────────────────────────────────────────
std::string Normalize(const std::string& name) {
    if (name.empty()) return "";
    std::string lower = ToLowerCase(Trim(name));
    if (ALIAS_MAP.find(lower) != ALIAS_MAP.end()) return ALIAS_MAP[lower];
    if (lower.length() > 2) {
        std::string auto_name = lower;
        auto_name[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(auto_name[0])));
        return auto_name;
    }
    return "";
}

std::string AssignGroup(const std::string& std_name) {
    if (std_name.empty()) return "UNKNOWN";
    if (CANONICAL.find(std_name) != CANONICAL.end()) return CANONICAL[std_name];
    return "OTHER";
}

int main() {
    std::string inputFilePath = "..\\..\\GDELT\\GDELT_Daily_All_Countries.csv";
    std::string outputFilePath = "..\\..\\GDELT\\GDELT_Daily_All_Countries_Grouped.csv";

    std::ifstream inFile(inputFilePath);
    if (!inFile.is_open()) {
        std::cerr << "파일 열기 실패: " << inputFilePath << "\n";
        return -1;
    }

    std::ofstream outFile(outputFilePath);
    if (!outFile.is_open()) {
        std::cerr << "출력 파일 생성 실패: " << outputFilePath << "\n";
        return -1;
    }

    std::string headerLine;
    if (!std::getline(inFile, headerLine)) {
        std::cerr << "파일이 비어있습니다.\n";
        return -1;
    }

    std::vector<std::string> headers = ParseCSVLine(headerLine);
    int countryColIndex = -1;
    std::string foundColName = "";

    for (size_t i = 0; i < headers.size(); ++i) {
        std::string h = ToLowerCase(Trim(headers[i]));
        for (const auto& cand : CANDIDATE_COLS) {
            if (h == ToLowerCase(cand)) {
                countryColIndex = static_cast<int>(i);
                foundColName = headers[i];
                break;
            }
        }
        if (countryColIndex != -1) break;
    }

    if (countryColIndex == -1) {
        std::cerr << "국가명 컬럼을 자동 감지할 수 없습니다.\n";
        return -1;
    }

    std::cout << "국가명 컬럼 자동 감지: '" << foundColName << "' (인덱스: " << countryColIndex << ")\n";

    outFile << headerLine << ",country_std,group\n";

    int total = 0;
    int matched = 0;
    std::map<std::string, int> unknown_counts;

    std::cout << " GDELT 초고속 데이터 처리 시작...\n";

    std::string line;
    while (std::getline(inFile, line)) {
        if (line.empty()) continue;
        total++;

        std::vector<std::string> row = ParseCSVLine(line);
        std::string country_val = "";

        if (countryColIndex >= 0 && static_cast<size_t>(countryColIndex) < row.size()) {
            country_val = row[countryColIndex];
        }

        std::string std_name = Normalize(country_val);
        std::string group = AssignGroup(std_name);

        if (group != "UNKNOWN") matched++;
        else {
            std::string badName = Trim(country_val);
            if (!badName.empty()) unknown_counts[badName]++;
        }

        outFile << line << "," << std_name << "," << group << "\n";
    }

    inFile.close();
    outFile.close();

    int unknown = total - matched;

    std::cout << "\n========================================\n";
    std::cout << " 매핑 결과 요약\n";
    std::cout << "========================================\n";
    std::cout << " - 전체 행 수   : " << total << " 행\n";
    std::cout << " - 성공(Matched): " << matched << " 행 (" << (total > 0 ? (double)matched / total * 100 : 0) << "%)\n";
    std::cout << " - 실패(Unknown): " << unknown << " 행 (" << (total > 0 ? (double)unknown / total * 100 : 0) << "%)\n";

    if (unknown > 0) {
        std::cout << "\n 매핑 실패 데이터 상위 노출 (최대 15개)\n";
        std::vector<std::pair<std::string, int>> vec(unknown_counts.begin(), unknown_counts.end());
        std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
            });

        int printCount = 0;
        for (const auto& pair : vec) {
            std::cout << "    '" << pair.first << "' -> " << pair.second << "행\n";
            printCount++;
            if (printCount >= 15) break;
        }
    }
    std::cout << "\n모든 처리가 완료되었습니다.\n";
    return 0;
}