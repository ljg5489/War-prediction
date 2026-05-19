#pragma once
#include <string>

struct GdeltPoint {
    // [0], [1], [2] : 날짜, 국가코드, 이벤트코드는 문자로 저장
    std::string sqlDate = "";
    std::string countryCode = "";
    std::string eventCode = "";

    // [3], [4] : 척도와 톤은 소수점이 있는 실수(double)로 저장
    double goldstein = 0.0;
    double avgTone = 0.0;

    // [5], [6], [7] : 횟수, 출처 수, 기사 수는 정수(int)로 저장
    int totalMentions = 0;
    int totalSources = 0;
    int totalArticles = 0;
};