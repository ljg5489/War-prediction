#pragma warning(disable: 4996)

#include "ScatterViewer.h"
#include <iostream>

#include <vtkContextScene.h>
#include <vtkChartXY.h>
#include <vtkPlotPoints.h>
#include <vtkTable.h>
#include <vtkFloatArray.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkAxis.h>       // ✅ 추가: 축 범위 고정에 필요
#include <vtkTextProperty.h> // ✅ 추가: 축 글씨 스타일

// ============================================================
//  GDELT Scatter Plot
// ============================================================
vtkSmartPointer<vtkContextView> ShowScatterPlot(
    const std::vector<GdeltPoint>& data,
    const std::string& columnName)
{
    if (data.empty()) return nullptr;

    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();

    vtkSmartPointer<vtkFloatArray> arrX = vtkSmartPointer<vtkFloatArray>::New();
    arrX->SetName("Data_Index");
    vtkSmartPointer<vtkFloatArray> arrY = vtkSmartPointer<vtkFloatArray>::New();
    arrY->SetName(columnName.c_str());

    int index = 0;
    for (const auto& pt : data) {
        arrX->InsertNextValue(index++);
        arrY->InsertNextValue(pt.goldstein);
    }

    table->AddColumn(arrX);
    table->AddColumn(arrY);

    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);
    view->GetRenderWindow()->SetSize(800, 400);
    view->GetRenderWindow()->SetWindowName("3. Scatter Plot Viewer");

    vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    chart->SetShowLegend(false);

    vtkPlotPoints* points = vtkPlotPoints::SafeDownCast(
        chart->AddPlot(vtkChart::POINTS));
    points->SetInputData(table, 0, 1);
    points->SetColorF(1.0, 0.27, 0.0, 0.2);
    points->SetMarkerSize(1.5);

    view->GetRenderWindow()->Render();
    return view;
}

// ============================================================
//  ACLED World Conflict Map — Scatter Plot
// ============================================================
vtkSmartPointer<vtkContextView> ShowScatterPlot(
    const std::vector<AcledPoint>& data)
{
    if (data.empty()) return nullptr;

    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();

    // X축: 경도(Longitude),  Y축: 위도(Latitude)
    vtkSmartPointer<vtkFloatArray> arrX = vtkSmartPointer<vtkFloatArray>::New();
    arrX->SetName("Longitude");
    vtkSmartPointer<vtkFloatArray> arrY = vtkSmartPointer<vtkFloatArray>::New();
    arrY->SetName("Latitude");

    int validCount = 0;
    for (const auto& pt : data) {
        if (pt.latitude != 0.0 && pt.longitude != 0.0) {
            arrX->InsertNextValue(static_cast<float>(pt.longitude));
            arrY->InsertNextValue(static_cast<float>(pt.latitude));
            ++validCount;
        }
    }

    std::cout << "[ScatterViewer] Valid ACLED points: " << validCount << std::endl;

    table->AddColumn(arrX);
    table->AddColumn(arrY);

    // ── 뷰 & 창 설정 ─────────────────────────────────────────
    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(0.08, 0.08, 0.12);
    view->GetRenderWindow()->SetSize(1400, 700); // 와이드 스크린으로 확장
    view->GetRenderWindow()->SetWindowName("ACLED World Conflict Map");

    // ── 차트 생성 ─────────────────────────────────────────────
    vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    chart->SetShowLegend(false);

    // ── ✅ 핵심 수정: 축 범위를 세계 지도 전체로 고정 ──────────
    // X축: 경도 -180 ~ +180
    vtkAxis* xAxis = chart->GetAxis(vtkAxis::BOTTOM);
    xAxis->SetRange(-180.0, 180.0);
    xAxis->SetBehavior(vtkAxis::FIXED);   // 자동 스케일 OFF
    xAxis->SetTitle("Longitude");
    xAxis->GetTitleProperties()->SetColor(0.8, 0.8, 0.8);
    xAxis->GetLabelProperties()->SetColor(0.7, 0.7, 0.7);
    xAxis->GetGridPen()->SetColorF(0.25, 0.25, 0.30, 1.0);

    // Y축: 위도 -90 ~ +90
    vtkAxis* yAxis = chart->GetAxis(vtkAxis::LEFT);
    yAxis->SetRange(-90.0, 90.0);
    yAxis->SetBehavior(vtkAxis::FIXED);   // 자동 스케일 OFF
    yAxis->SetTitle("Latitude");
    yAxis->GetTitleProperties()->SetColor(0.8, 0.8, 0.8);
    yAxis->GetLabelProperties()->SetColor(0.7, 0.7, 0.7);
    yAxis->GetGridPen()->SetColorF(0.25, 0.25, 0.30, 1.0);

    // ── 플롯 설정 ─────────────────────────────────────────────
    vtkPlotPoints* points = vtkPlotPoints::SafeDownCast(
        chart->AddPlot(vtkChart::POINTS));
    points->SetInputData(table, 0, 1);

    // 노란 계열 반투명: 분쟁 밀집 지역이 더 밝게 보임
    points->SetColorF(1.0, 0.85, 0.1, 0.55);

    // ✅ 점 크기 증가: 1.6 → 3.5 (지도 위에서 충분히 식별 가능)
    points->SetMarkerSize(3.5);

    // ✅ 마커 스타일: 원형
    points->SetMarkerStyle(vtkPlotPoints::CIRCLE);

    view->GetRenderWindow()->Render();
    return view;
}