// 🌟 최상단에 추가: VTK의 구버전 문법 잔소리를 강제로 차단합니다.
#pragma warning(disable: 4996) 

#include "HistogramViewer.h"
#include <iostream>

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
    arrX->SetName("Goldstein_Index");
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
    view->GetRenderWindow()->SetWindowName("1. Histogram Viewer");

    vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    chart->SetShowLegend(true);

    vtkPlot* line = chart->AddPlot(vtkChart::BAR);
    line->SetInputData(table, 0, 1);

    // 🌟 절대 SetColor를 쓰지 않고 SetColorF로만 작성했습니다.
    line->SetColorF(0.0, 0.4, 0.8, 1.0);
    line->GetBrush()->SetColorF(0.0, 0.4, 0.8, 0.6);

    chart->GetAxis(vtkAxis::BOTTOM)->SetTitle("Scale (-10 to 10)");
    chart->GetAxis(vtkAxis::LEFT)->SetTitle("Frequency");

    view->GetRenderWindow()->Render();
    return view;
}