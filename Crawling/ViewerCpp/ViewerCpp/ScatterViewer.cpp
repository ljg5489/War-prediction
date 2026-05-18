// 🌟 최상단에 추가: 구버전 문법 차단
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

vtkSmartPointer<vtkContextView> ShowScatterPlot(const std::vector<GdeltPoint>& data, const std::string& columnName) {
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

    vtkPlotPoints* points = vtkPlotPoints::SafeDownCast(chart->AddPlot(vtkChart::POINTS));
    points->SetInputData(table, 0, 1);

    // 🌟 SetColorF로 완벽 교체 및 2D 마커 설정
    points->SetColorF(1.0, 0.27, 0.0, 0.2);
    points->SetMarkerSize(1.5);

    view->GetRenderWindow()->Render();
    return view;
}