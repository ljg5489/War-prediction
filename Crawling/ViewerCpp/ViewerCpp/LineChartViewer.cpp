#pragma warning(disable: 4996) 
#include "LineChartViewer.h"
#include <iostream>

#include <vtkContextScene.h>
#include <vtkChartXY.h>
#include <vtkPlot.h>
#include <vtkTable.h>
#include <vtkFloatArray.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

vtkSmartPointer<vtkContextView> ShowLineChart(const std::vector<GdeltPoint>& data, const std::string& columnName) {
    if (data.empty()) return nullptr;

    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();
    vtkSmartPointer<vtkFloatArray> arrX = vtkSmartPointer<vtkFloatArray>::New();
    arrX->SetName("Data_Index");
    vtkSmartPointer<vtkFloatArray> arrY = vtkSmartPointer<vtkFloatArray>::New();
    arrY->SetName(columnName.c_str());

    // 🌟 샘플링 전면 해제: 4,700만 개 데이터 원본 100% 렌더링
    for (size_t i = 0; i < data.size(); i++) {
        arrX->InsertNextValue(static_cast<float>(i));
        arrY->InsertNextValue(static_cast<float>(data[i].avgTone));
    }

    table->AddColumn(arrX);
    table->AddColumn(arrY);

    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);

    // 🌟 데이터가 너무 많으니 창을 와이드(1200x600)로 키웁니다
    view->GetRenderWindow()->SetSize(1200, 600);
    view->GetRenderWindow()->SetWindowName("4. Line Chart Viewer (ALL 47M DATA)");

    vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    chart->SetShowLegend(false);

    vtkPlot* line = chart->AddPlot(vtkChart::LINE);
    line->SetInputData(table, 0, 1);

    // 🌟 선 두께를 최소치(1.0)로 설정해서 어떻게든 흐름이 보이게 발악합니다
    line->SetColor(0, 114, 178, 255);
    line->SetWidth(1.0);

    view->GetRenderWindow()->Render();
    return view;
}