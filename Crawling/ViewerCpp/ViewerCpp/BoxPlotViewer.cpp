// 🌟 최상단에 추가: VTK 내부의 꼬인 경고 메시지를 원천 차단
#pragma warning(disable: 4996) 

#include "BoxPlotViewer.h"
#include <iostream>

#include <vtkContextScene.h>
#include <vtkChartBox.h>
#include <vtkPlotBox.h>
#include <vtkTable.h>
#include <vtkDoubleArray.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkComputeQuartiles.h>

vtkSmartPointer<vtkContextView> ShowBoxPlot(const std::vector<GdeltPoint>& data, const std::string& columnName) {
    if (data.empty()) return nullptr;

    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();
    vtkSmartPointer<vtkDoubleArray> arrX = vtkSmartPointer<vtkDoubleArray>::New();
    arrX->SetName(columnName.c_str());

    for (const auto& pt : data) {
        arrX->InsertNextValue(pt.goldstein);
    }
    table->AddColumn(arrX);

    vtkSmartPointer<vtkComputeQuartiles> quartiles = vtkSmartPointer<vtkComputeQuartiles>::New();
    quartiles->SetInputData(table);
    quartiles->Update();

    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);
    view->GetRenderWindow()->SetSize(500, 800);
    view->GetRenderWindow()->SetWindowName("2. Box Plot Viewer");

    vtkSmartPointer<vtkChartBox> chart = vtkSmartPointer<vtkChartBox>::New();
    view->GetScene()->AddItem(chart);

    chart->GetPlot(0)->SetInputData(quartiles->GetOutput());
    chart->SetColumnVisibility(columnName.c_str(), true);

    view->GetRenderWindow()->Render();
    return view;
}