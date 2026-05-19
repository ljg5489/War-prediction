#pragma once
#include <string>

struct AcledPoint {
    // [0] ~ [5] : 주차, 지역, 국가, 행정구역, 이벤트 종류 (문자열)
    std::string week = "";
    std::string region = "";
    std::string country = "";
    std::string admin1 = "";
    std::string eventType = "";
    std::string subEventType = "";

    // [6] ~ [8] : 발생 건수, 사상자 수, 노출 인구 (정수)
    int events = 0;
    int fatalities = 0;
    int populationExposure = 0;

    // [9], [10] : 무력 충돌 유형, 고유 ID (문자열)
    std::string disorderType = "";
    std::string id = "";

    // [11], [12] : 중심점 위도, 경도 (실수)
    double latitude = 0.0;
    double longitude = 0.0;
};