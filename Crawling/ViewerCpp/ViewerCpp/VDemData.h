#pragma once
#include <string>

struct VDemPoint {
    std::string country_name;   // [0]
    std::string country_text_id;// [1] 예: AFG
    int year = 0;               // [2] 연도 (X축)

    // Y축에 그릴 지표들 (결측치가 많을 수 있으므로 0.0 초기화)
    double v2elpeace = 0.0;     // [3]
    double v2x_rule = 0.0;      // [4] 법치 지수
    double v2x_clphy = 0.0;     // [5] 신체적 폭력으로부터의 자유
    double e_pt_coup = 0.0;     // [6]
    double e_civil_war = 0.0;   // [7] (미리보기에서 Empty 확인됨)
    double v2x_libdem = 0.0;    // [8] 자유 민주주의 지수
    double v2x_corr = 0.0;      // [9] 부패 지수
    double v2x_veracc = 0.0;    // [10]
    double v2xcs_ccsi = 0.0;    // [11]
    double v2x_polyarchy = 0.0; // [12] 선거 민주주의 지수
};