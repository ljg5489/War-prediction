#pragma warning(disable: 4996) 
#include "DeathCrossViewer.h"
#include <iostream>
#include <map>
#include <vector>
#include <string>

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
#include <vtkTextProperty.h> // 🌟 필수 포함!
#include <vtkPen.h>

vtkSmartPointer<vtkContextView> ShowDeathCrossChart(
    const std::vector<GdeltPoint>& data,
    const std::string& targetCountryCode)
{
    if (data.empty()) return nullptr;

    // 1. 날짜별(YYYYMM)로 이벤트 성격 카운팅
    // 구조: map<YYYYMM, pair<TotalEvents, pair<ConflictEvents, CoopEvents>>>
    std::map<std::string, std::pair<int, std::pair<int, int>>> monthlyStats;

    for (const auto& pt : data) {
        if (pt.countryCode == targetCountryCode && pt.sqlDate.length() >= 6) {
            std::string month = pt.sqlDate.substr(0, 6); // YYYYMM 단위(30일)로 묶음

            int rootCode = 0;
            try {
                // CAMEO 코드가 3자리(예: 141)면 앞 2자리(14)를, 2자리 이하면 그대로 변환
                if (pt.eventCode.length() >= 2) {
                    rootCode = std::stoi(pt.eventCode.substr(0, 2));
                }
                else if (!pt.eventCode.empty()) {
                    rootCode = std::stoi(pt.eventCode);
                }
            }
            catch (...) { continue; } // 변환 오류 무시

            // 통계 집계
            monthlyStats[month].first++; // 총 이벤트 수 증가
            if (rootCode >= 14) monthlyStats[month].second.first++;   // Conflict (시위, 폭동, 무력충돌)
            if (rootCode <= 6)  monthlyStats[month].second.second++;  // Cooperation (협력, 지원, 성명)
        }
    }

    if (monthlyStats.empty()) return nullptr;

    // 2. 비율 계산 (0.0 ~ 1.0)
    std::vector<double> conflictRatios;
    std::vector<double> isolationIndices;

    for (const auto& pair : monthlyStats) {
        int total = pair.second.first;
        int conflict = pair.second.second.first;
        int coop = pair.second.second.second;

        if (total > 0) {
            conflictRatios.push_back(static_cast<double>(conflict) / total);
            // 고립 지수 = 1 - (협력비율)
            isolationIndices.push_back(1.0 - (static_cast<double>(coop) / total));
        }
        else {
            conflictRatios.push_back(0.0);
            isolationIndices.push_back(0.0);
        }
    }

    // 3. VTK 테이블 데이터 삽입
    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();
    vtkSmartPointer<vtkIntArray> arrX = vtkSmartPointer<vtkIntArray>::New(); arrX->SetName("Months");
    vtkSmartPointer<vtkFloatArray> arrConflict = vtkSmartPointer<vtkFloatArray>::New(); arrConflict->SetName("Conflict Ratio");
    vtkSmartPointer<vtkFloatArray> arrIsolation = vtkSmartPointer<vtkFloatArray>::New(); arrIsolation->SetName("Isolation Index");

    for (size_t i = 0; i < conflictRatios.size(); ++i) {
        arrX->InsertNextValue(static_cast<int>(i));
        arrConflict->InsertNextValue(static_cast<float>(conflictRatios[i]));
        arrIsolation->InsertNextValue(static_cast<float>(isolationIndices[i]));
    }

    table->AddColumn(arrX);
    table->AddColumn(arrConflict);
    table->AddColumn(arrIsolation);

    // 4. 뷰어 및 렌더링 세팅
    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0); 
    view->GetRenderWindow()->SetSize(900, 500);
    view->GetRenderWindow()->SetWindowName(("Death Cross - " + targetCountryCode).c_str());

    vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    chart->SetShowLegend(true);

    chart->SetTitle(("Death Cross: Conflict vs Isolation (" + targetCountryCode + ")").c_str());
    chart->GetTitleProperties()->SetColor(0.9, 0.9, 0.9);
    chart->GetTitleProperties()->SetFontSize(20);

    // X축 세팅
    chart->GetAxis(vtkAxis::BOTTOM)->SetTitle("Time (Months)");

    // 🌟 핵심: 비율(Ratio)이므로 Y축을 0.0 ~ 1.0으로 고정
    vtkAxis* yAxis = chart->GetAxis(vtkAxis::LEFT);
    yAxis->SetTitle("Index Ratio (0.0 - 1.0)");
    yAxis->SetBehavior(vtkAxis::FIXED);
    yAxis->SetRange(0.0, 1.0);

    // 🔴 1. 내부 폭발(Conflict Ratio) 선 - 강렬한 붉은색
    vtkPlotLine* lineConflict = vtkPlotLine::SafeDownCast(chart->AddPlot(vtkChart::LINE));
    lineConflict->SetInputData(table, 0, 1);
    lineConflict->SetColor(255, 60, 60, 255);
    lineConflict->SetWidth(3.0);

    // 🟡 2. 국제적 고립(Isolation Index) 선 - 경고의 노란색
    vtkPlotLine* lineIsolation = vtkPlotLine::SafeDownCast(chart->AddPlot(vtkChart::LINE));
    lineIsolation->SetInputData(table, 0, 2);
    lineIsolation->SetColor(255, 200, 0, 255);
    lineIsolation->SetWidth(3.0);

    view->GetRenderWindow()->Render();
    return view;
}