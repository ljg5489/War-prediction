// 🌟 최상단에 추가: VTK 내부의 꼬인 경고 메시지를 원천 차단
#pragma warning(disable: 4996) 

#include "BoxPlotViewer.h"
#include <iostream>

#include <vtkAxis.h>
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

vtkSmartPointer<vtkContextView> ShowBoxPlot_Acled(const std::vector<AcledPoint>& data, const std::string& columnName) {
    if (data.empty()) return nullptr;

    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();
    vtkSmartPointer<vtkDoubleArray> arrX = vtkSmartPointer<vtkDoubleArray>::New();
    arrX->SetName(columnName.c_str());

    for (const auto& pt : data) {
        arrX->InsertNextValue(pt.longitude);
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

vtkSmartPointer<vtkContextView> ShowBoxPlot_Fatalities(const std::vector<AcledPoint>& data, const std::string& columnName) {
    if (data.empty()) return nullptr;

    // ── 테이블 구성 ──────────────────────────────────────────
    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();
    vtkSmartPointer<vtkDoubleArray> arrX = vtkSmartPointer<vtkDoubleArray>::New();
    arrX->SetName(columnName.c_str());

    for (const auto& pt : data) {
        arrX->InsertNextValue(static_cast<double>(pt.fatalities));
    }
    table->AddColumn(arrX);

    // ── 사분위 계산 ──────────────────────────────────────────
    vtkSmartPointer<vtkComputeQuartiles> quartiles = vtkSmartPointer<vtkComputeQuartiles>::New();
    quartiles->SetInputData(table);
    quartiles->Update();

    // ✅ null 체크 추가
    if (!quartiles->GetOutput() || quartiles->GetOutput()->GetNumberOfColumns() == 0) {
        std::cerr << "[ShowBoxPlot_Fatalities] quartiles 출력이 비어 있습니다.\n";
        return nullptr;
    }

    // ── 뷰 생성 ──────────────────────────────────────────────
    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);
    view->GetRenderWindow()->SetSize(500, 800);
    view->GetRenderWindow()->SetWindowName("ACLED Fatalities Box Plot");

    // ── 차트 생성 ─────────────────────────────────────────────
    vtkSmartPointer<vtkChartBox> chart = vtkSmartPointer<vtkChartBox>::New();
    view->GetScene()->AddItem(chart);

    // ✅ 핵심 수정: GetPlot(0) 대신 SafeDownCast 사용
    chart->SetColumnVisibility(columnName.c_str(), true);
    chart->GetPlot(0)->SetInputData(quartiles->GetOutput());  // ← 이 순서가 중요

    vtkPlotBox* plot = vtkPlotBox::SafeDownCast(chart->GetPlot(0));
    if (!plot) {
        std::cerr << "[ShowBoxPlot_Fatalities] vtkPlotBox 캐스팅 실패\n";
        return nullptr;
    }
    plot->SetInputData(quartiles->GetOutput());

    // ✅ 축 접근도 안전하게
    vtkAxis* axis = chart->GetAxis(vtkAxis::LEFT);
    if (axis) axis->SetTitle("Fatalities Count");

    view->GetRenderWindow()->Render();
    return view;
}