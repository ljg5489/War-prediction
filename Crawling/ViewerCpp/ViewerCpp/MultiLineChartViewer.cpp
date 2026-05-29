#pragma warning(disable: 4996) 
#include "MultiLineChartViewer.h"
#include <iostream>

#include <vtkContextScene.h>
#include <vtkChartXY.h>
#include <vtkPlotLine.h>
#include <vtkTable.h>
#include <vtkIntArray.h>
#include <vtkFloatArray.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkAxis.h>
#include <vtkPen.h>

vtkSmartPointer<vtkContextView> ShowVDemMultiLine(const std::vector<VDemPoint>& data, const std::string& targetCountry){
    if (data.empty()) return nullptr;

    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();

    // 🌟 X축: 연도 (Year)
    vtkSmartPointer<vtkIntArray> arrYear = vtkSmartPointer<vtkIntArray>::New();
    arrYear->SetName("Year");

    // 🌟 Y축: 비교할 4개의 핵심 지표 배열 생성
    vtkSmartPointer<vtkFloatArray> arrRule = vtkSmartPointer<vtkFloatArray>::New();
    arrRule->SetName("Rule of Law");
    vtkSmartPointer<vtkFloatArray> arrDem = vtkSmartPointer<vtkFloatArray>::New();
    arrDem->SetName("Liberal Democracy");
    vtkSmartPointer<vtkFloatArray> arrPhys = vtkSmartPointer<vtkFloatArray>::New();
    arrPhys->SetName("Physical Freedom");
    vtkSmartPointer<vtkFloatArray> arrCorr = vtkSmartPointer<vtkFloatArray>::New();
    arrCorr->SetName("Corruption Index");

    // 타겟 국가의 데이터만 추출하여 테이블에 삽입
    for (const auto& pt : data) {
        if (pt.country_text_id == targetCountry) {
            arrYear->InsertNextValue(pt.year);
            arrRule->InsertNextValue(static_cast<float>(pt.v2x_rule));
            arrDem->InsertNextValue(static_cast<float>(pt.v2x_libdem));
            arrPhys->InsertNextValue(static_cast<float>(pt.v2x_clphy));
            arrCorr->InsertNextValue(static_cast<float>(pt.v2x_corr));
        }
    }

    table->AddColumn(arrYear);
    table->AddColumn(arrRule);
    table->AddColumn(arrDem);
    table->AddColumn(arrPhys);
    table->AddColumn(arrCorr);

    // ── 뷰 & 창 설정 ──
    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0); // 다크 모드
    view->GetRenderWindow()->SetSize(1000, 600);
    view->GetRenderWindow()->SetWindowName((targetCountry + " - V-Dem Multi-Line Chart").c_str());
    
    vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    chart->SetShowLegend(true); // 🌟 선이 여러 개이므로 범례(Legend) 켜기

    // X축 세팅 (연도)
    vtkAxis* xAxis = chart->GetAxis(vtkAxis::BOTTOM);
    xAxis->SetTitle("Year");

    // ── 1. 법치 지수 (초록색) ──
    vtkPlotLine* lineRule = vtkPlotLine::SafeDownCast(chart->AddPlot(vtkChart::LINE));
    lineRule->SetInputData(table, 0, 1);
    lineRule->SetColor(0, 255, 100, 255); // Green
    lineRule->SetWidth(2.5);

    // ── 2. 자유 민주주의 (파란색) ──
    vtkPlotLine* lineDem = vtkPlotLine::SafeDownCast(chart->AddPlot(vtkChart::LINE));
    lineDem->SetInputData(table, 0, 2);
    lineDem->SetColor(50, 150, 255, 255); // Blue
    lineDem->SetWidth(2.5);

    // ── 3. 신체적 폭력 자유 (주황색) ──
    vtkPlotLine* linePhys = vtkPlotLine::SafeDownCast(chart->AddPlot(vtkChart::LINE));
    linePhys->SetInputData(table, 0, 3);
    linePhys->SetColor(255, 150, 0, 255); // Orange
    linePhys->SetWidth(2.5);

    // ── 4. 부패 지수 (빨간색) ──
    vtkPlotLine* lineCorr = vtkPlotLine::SafeDownCast(chart->AddPlot(vtkChart::LINE));
    lineCorr->SetInputData(table, 0, 4);
    lineCorr->SetColor(255, 50, 50, 255); // Red
    lineCorr->SetWidth(2.5);

    view->GetRenderWindow()->Render();
    return view;
}