// 🌟 최상단에 추가: VTK의 구버전 문법 잔소리를 강제로 차단합니다.
#pragma warning(disable: 4996) 
#include "HistogramViewer.h"
#include <iostream>
#include <algorithm> // std::min_element, std::max_element 사용을 위해 필요
#include <vtkContextScene.h>
#include <vtkChartXY.h>
#include <vtkPlot.h>
#include <vtkPlotBar.h>
#include <vtkTable.h>
#include <vtkFloatArray.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkAxis.h>
#include <vtkBrush.h>

vtkSmartPointer<vtkContextView> ShowHistogram(const std::vector<GdeltPoint>& data, int numBins, double minVal, double maxVal) {
    if (data.empty()) return nullptr;

    std::vector<int> bins(numBins, 0);
    double binSize = (maxVal - minVal) / numBins;

    for (const auto& pt : data) {
        if (pt.goldstein >= minVal && pt.goldstein <= maxVal) {
            int binIndex = static_cast<int>((pt.goldstein - minVal) / binSize);
            if (binIndex >= numBins) binIndex = numBins - 1;
            bins[binIndex]++;
        }
    }

    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();
    vtkSmartPointer<vtkFloatArray> arrX = vtkSmartPointer<vtkFloatArray>::New();
    arrX->SetName("Goldstein_Index"); // 내부 데이터용 X축 이름

    vtkSmartPointer<vtkFloatArray> arrY = vtkSmartPointer<vtkFloatArray>::New();
    // 🌟 수정 1: 범례(Legend)에 표시될 이름을 직관적으로 변경
    arrY->SetName("AvgGoldstein (Event Count)");

    for (int i = 0; i < numBins; ++i) {
        arrX->InsertNextValue(minVal + (i * binSize) + (binSize / 2.0));
        arrY->InsertNextValue(bins[i]);
    }

    table->AddColumn(arrX);
    table->AddColumn(arrY);

    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);
    view->GetRenderWindow()->SetSize(800, 600);
    view->GetRenderWindow()->SetWindowName("1. Histogram Viewer (GDELT)");

    vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    chart->SetShowLegend(true);

    vtkPlot* line = chart->AddPlot(vtkChart::BAR);
    line->SetInputData(table, 0, 1);

    // 🌟 절대 SetColor를 쓰지 않고 SetColorF로만 작성
    line->SetColorF(0.0, 0.4, 0.8, 1.0);
    line->GetBrush()->SetColorF(0.0, 0.4, 0.8, 0.6);

    // 🌟 수정 2: X축과 Y축 타이틀도 논문 퀄리티에 맞게 명확히 변경
    chart->GetAxis(vtkAxis::BOTTOM)->SetTitle("AvgGoldstein Score (-10.0 to +10.0)");
    chart->GetAxis(vtkAxis::LEFT)->SetTitle("Frequency (Number of Events)");

    view->GetRenderWindow()->Render();
    return view;
}

// 🌟 추가할 오버로딩 함수: ACLED 사상자 수처럼 순수 double 벡터를 받기 위한 히스토그램
vtkSmartPointer<vtkContextView> ShowHistogram(const std::vector<double>& data, const std::string& title, int numBins) {
    if (data.empty()) return nullptr;

    // 데이터의 최솟값과 최댓값을 자동으로 찾음
    double minVal = *std::min_element(data.begin(), data.end());
    double maxVal = *std::max_element(data.begin(), data.end());

    // 모든 값이 동일할 경우(예: 모두 0) 0으로 나누는 에러 방지
    if (minVal == maxVal) {
        maxVal = minVal + 1.0;
    }

    std::vector<int> bins(numBins, 0);
    double binSize = (maxVal - minVal) / numBins;

    // 빈도수 계산
    for (double val : data) {
        int binIndex = static_cast<int>((val - minVal) / binSize);
        if (binIndex >= numBins) binIndex = numBins - 1;
        bins[binIndex]++;
    }

    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();
    vtkSmartPointer<vtkFloatArray> arrX = vtkSmartPointer<vtkFloatArray>::New();
    arrX->SetName("Value");

    vtkSmartPointer<vtkFloatArray> arrY = vtkSmartPointer<vtkFloatArray>::New();
    arrY->SetName("Frequency");

    for (int i = 0; i < numBins; ++i) {
        arrX->InsertNextValue(minVal + (i * binSize) + (binSize / 2.0));
        arrY->InsertNextValue(bins[i]);
    }

    table->AddColumn(arrX);
    table->AddColumn(arrY);

    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);
    view->GetRenderWindow()->SetSize(800, 600);
    view->GetRenderWindow()->SetWindowName(title.c_str());

    vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    chart->SetShowLegend(true);
    chart->SetTitle(title.c_str());

    vtkPlot* line = chart->AddPlot(vtkChart::BAR);
    line->SetInputData(table, 0, 1);

    // 사상자 수(Fatalities)에 어울리는 붉은색 톤으로 설정 (SetColorF 유지)
    line->SetColorF(0.8, 0.2, 0.2, 1.0);
    line->GetBrush()->SetColorF(0.8, 0.2, 0.2, 0.6);

    chart->GetAxis(vtkAxis::BOTTOM)->SetTitle("Value (Fatalities)");
    chart->GetAxis(vtkAxis::LEFT)->SetTitle("Frequency (Event Count)");

    view->GetRenderWindow()->Render();
    return view;
}