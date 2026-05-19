#pragma warning(disable: 4996) 
#include "EscalationViewer.h"
#include <iostream>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm> // std::max 사용을 위해 추가

#include <vtkContextScene.h>
#include <vtkChartXY.h>
#include <vtkPlotLine.h>
#include <vtkTable.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkAxis.h>
#include <vtkPen.h>
#include <vtkTextProperty.h>

// 🌟 이동 평균 및 표준편차를 계산하는 헬퍼 함수
void CalculateRollingZScore(const std::vector<double>& values, int windowSize, std::vector<double>& zScores) {
    zScores.assign(values.size(), 0.0);

    for (size_t i = 0; i < values.size(); ++i) {
        if (i < windowSize) {
            zScores[i] = 0.0; // 데이터가 충분히 쌓이기 전에는 0 처리
            continue;
        }

        double sum = 0.0;
        for (size_t j = i - windowSize; j < i; ++j) sum += values[j];
        double mean = sum / windowSize;

        double sq_sum = 0.0;
        for (size_t j = i - windowSize; j < i; ++j) sq_sum += (values[j] - mean) * (values[j] - mean);

        double stddev = std::sqrt(sq_sum / windowSize);

        // 🚨 핵심 수정 1: 사망자가 0명이다가 1명 생겼을 때 Z-score가 수백만으로 폭발하는 것을 방지
        // 표준편차가 최소 1.0은 되도록 하한선을 걸어줍니다.
        stddev = std::max(stddev, 1.0);

        // 현재 주의 Z-score 계산
        zScores[i] = (values[i] - mean) / stddev;
    }
}

vtkSmartPointer<vtkContextView> ShowEscalationChart(const std::vector<AcledPoint>& data, const std::string& targetCountry) {
    if (data.empty()) return nullptr;

    // 1. 타겟 국가의 주차(Week)별 사상자(Fatalities) 합산
    std::map<std::string, double> weeklyFatalities;
    for (const auto& pt : data) {
        if (pt.country == targetCountry) {
            // 빈 문자열 방어
            if (!pt.week.empty()) {
                weeklyFatalities[pt.week] += pt.fatalities;
            }
        }
    }

    if (weeklyFatalities.empty()) {
        std::cerr << "No data found for country: " << targetCountry << std::endl;
        return nullptr;
    }

    // 2. 시간 순 벡터로 옮기기
    std::vector<double> fatalities;
    for (const auto& pair : weeklyFatalities) {
        fatalities.push_back(pair.second);
    }

    // 3. Z-score 계산 (장기 12주)
    std::vector<double> zLong;
    CalculateRollingZScore(fatalities, 12, zLong);

    std::vector<double> escValues(fatalities.size(), 0.0);

    // 🚨 핵심 수정 2: 분모 0 방지용 엡실론을 1e-6에서 1.0으로 넉넉하게 변경
    // 이래야 자잘한 국지전 노이즈에 그래프가 요동치지 않고, 진짜 '전쟁급 스케일'에서만 반응합니다.
    double epsilon = 1.0;

    for (size_t i = 0; i < fatalities.size(); ++i) {
        if (i >= 12) {
            double z_short = zLong[i];
            double z_long_prev = zLong[i - 1];
            escValues[i] = (z_short - z_long_prev) / (std::abs(z_long_prev) + epsilon);
        }
    }

    // 4. VTK 테이블에 데이터 삽입
    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();
    vtkSmartPointer<vtkIntArray> arrX = vtkSmartPointer<vtkIntArray>::New(); arrX->SetName("Week Index");
    vtkSmartPointer<vtkFloatArray> arrY = vtkSmartPointer<vtkFloatArray>::New(); arrY->SetName("ESC Velocity");
    vtkSmartPointer<vtkFloatArray> arrZero = vtkSmartPointer<vtkFloatArray>::New(); arrZero->SetName("Baseline");

    for (size_t i = 0; i < escValues.size(); ++i) {
        arrX->InsertNextValue(static_cast<int>(i));
        arrY->InsertNextValue(static_cast<float>(escValues[i]));
        arrZero->InsertNextValue(0.0f); // Y=0 기준선
    }

    table->AddColumn(arrX);
    table->AddColumn(arrY);
    table->AddColumn(arrZero);

    // 5. 뷰어 및 창 렌더링 세팅
    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(0.1, 0.12, 0.16);
    view->GetRenderWindow()->SetSize(900, 500);

    std::string windowTitle = "Escalation Velocity - " + targetCountry;
    view->GetRenderWindow()->SetWindowName(windowTitle.c_str());

    vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    chart->SetShowLegend(true);
    chart->SetTitle(("Conflict Escalation (ESC) : " + targetCountry).c_str());
    chart->GetTitleProperties()->SetColor(0.9, 0.9, 0.9);
    chart->GetTitleProperties()->SetFontSize(18);

    // Y축 범위 자동 조절을 켜서 스파이크를 잘 잡도록 세팅
    chart->GetAxis(vtkAxis::LEFT)->SetBehavior(vtkAxis::AUTO);
    chart->GetAxis(vtkAxis::BOTTOM)->SetTitle("Weeks (Time)");
    chart->GetAxis(vtkAxis::LEFT)->SetTitle("Acceleration Index");

    // 🔴 메인 ESC 선 그리기 (빨간색)
    vtkPlotLine* lineESC = vtkPlotLine::SafeDownCast(chart->AddPlot(vtkChart::LINE));
    lineESC->SetInputData(table, 0, 1);
    lineESC->SetColor(255, 60, 60, 255);
    lineESC->SetWidth(2.5);

    // ⚪ Y=0 기준선
    vtkPlotLine* lineZero = vtkPlotLine::SafeDownCast(chart->AddPlot(vtkChart::LINE));
    lineZero->SetInputData(table, 0, 2);
    lineZero->SetColor(150, 150, 150, 200);
    lineZero->SetWidth(1.0);

    view->GetRenderWindow()->Render();
    return view;
}