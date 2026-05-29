#pragma warning(disable: 4996) 
#include "InteractiveGroupViewer.h"
#include <iostream>

#include <vtkCommand.h>
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
#include <vtkTextProperty.h>

// 🌟 키보드 입력을 감지하는 옵저버 (감시자) 클래스
class VDemKeyObserver : public vtkCommand {
public:
    static VDemKeyObserver* New() { return new VDemKeyObserver; }

    std::vector<VDemPoint> allData;
    std::vector<std::string> countryCodes;
    std::vector<std::string> countryNames;
    std::string groupName;
    int currentIndex = 0;

    // 차트를 다시 그리기 위해 포인터를 보관합니다.
    vtkSmartPointer<vtkTable> table;
    vtkSmartPointer<vtkChartXY> chart;
    vtkSmartPointer<vtkRenderWindow> renderWindow;

    // 키보드가 눌렸을 때 실행되는 함수
    virtual void Execute(vtkObject* caller, unsigned long eventId, void* callData) override {
        vtkRenderWindowInteractor* iren = vtkRenderWindowInteractor::SafeDownCast(caller);
        if (!iren) return;

        std::string key = iren->GetKeySym(); // 어떤 키가 눌렸는지 확인

        if (key == "Right") {
            currentIndex = (currentIndex + 1) % countryCodes.size();
            UpdateChart();
        }
        else if (key == "Left") {
            currentIndex = (currentIndex - 1 + countryCodes.size()) % countryCodes.size();
            UpdateChart();
        }
    }

    // 데이터를 교체하고 화면을 새로고침하는 함수
    void UpdateChart() {
        std::string targetCode = countryCodes[currentIndex];

        // 1. 기존 테이블의 배열들을 가져와서 싹 비웁니다.
        vtkIntArray* arrYear = vtkIntArray::SafeDownCast(table->GetColumnByName("Year"));
        vtkFloatArray* arrRule = vtkFloatArray::SafeDownCast(table->GetColumnByName("Rule"));
        vtkFloatArray* arrDem = vtkFloatArray::SafeDownCast(table->GetColumnByName("Democracy"));
        vtkFloatArray* arrPhys = vtkFloatArray::SafeDownCast(table->GetColumnByName("Physical"));
        vtkFloatArray* arrCorr = vtkFloatArray::SafeDownCast(table->GetColumnByName("Corruption"));

        arrYear->SetNumberOfValues(0);
        arrRule->SetNumberOfValues(0);
        arrDem->SetNumberOfValues(0);
        arrPhys->SetNumberOfValues(0);
        arrCorr->SetNumberOfValues(0);

        // 2. 새로운 국가의 데이터만 채워 넣습니다.
        for (const auto& pt : allData) {
            if (pt.country_text_id == targetCode) {
                arrYear->InsertNextValue(pt.year);
                arrRule->InsertNextValue(static_cast<float>(pt.v2x_rule));
                arrDem->InsertNextValue(static_cast<float>(pt.v2x_libdem));
                arrPhys->InsertNextValue(static_cast<float>(pt.v2x_clphy));
                arrCorr->InsertNextValue(static_cast<float>(pt.v2x_corr));
            }
        }

        // 3. 차트 제목 업데이트 및 렌더링
        table->Modified(); // 데이터가 바뀌었다고 VTK에 알림

        // (주의: VTK 창 제목에서 한글이 깨질 수 있어 영어 이름 활용을 권장합니다)
        std::string newTitle = groupName + " - " + countryNames[currentIndex] + " (" + targetCode + ")";
        chart->SetTitle(newTitle);
        chart->GetTitleProperties()->SetColor(0.9, 0.9, 0.9);
        chart->GetTitleProperties()->SetFontSize(24);

        renderWindow->Render(); // 화면 새로고침!
    }
};

// 🌟 메인 뷰어 생성 함수
vtkSmartPointer<vtkContextView> ShowInteractiveGroupChart(
    const std::vector<VDemPoint>& data,
    const std::string& groupName,
    const std::vector<std::string>& countryCodes,
    const std::vector<std::string>& countryNames)
{
    if (data.empty() || countryCodes.empty()) return nullptr;

    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();

    vtkSmartPointer<vtkIntArray> arrYear = vtkSmartPointer<vtkIntArray>::New(); arrYear->SetName("Year");
    vtkSmartPointer<vtkFloatArray> arrRule = vtkSmartPointer<vtkFloatArray>::New(); arrRule->SetName("Rule");
    vtkSmartPointer<vtkFloatArray> arrDem = vtkSmartPointer<vtkFloatArray>::New(); arrDem->SetName("Democracy");
    vtkSmartPointer<vtkFloatArray> arrPhys = vtkSmartPointer<vtkFloatArray>::New(); arrPhys->SetName("Physical");
    vtkSmartPointer<vtkFloatArray> arrCorr = vtkSmartPointer<vtkFloatArray>::New(); arrCorr->SetName("Corruption");

    table->AddColumn(arrYear);
    table->AddColumn(arrRule);
    table->AddColumn(arrDem);
    table->AddColumn(arrPhys);
    table->AddColumn(arrCorr);

    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0); // 다크 모드
    view->GetRenderWindow()->SetSize(800, 500);
    view->GetRenderWindow()->SetWindowName(groupName.c_str());

    vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    chart->SetShowLegend(true);

    vtkAxis* xAxis = chart->GetAxis(vtkAxis::BOTTOM);
    xAxis->SetTitle("Year");

    // Y축을 0.0 ~ 1.0으로 고정하여 국가 간 비교를 직관적으로 만듭니다.
    vtkAxis* yAxis = chart->GetAxis(vtkAxis::LEFT);
    yAxis->SetBehavior(vtkAxis::FIXED);
    yAxis->SetRange(0.0, 1.0);

    // 4개의 선 색상 세팅
    vtkPlotLine* lineRule = vtkPlotLine::SafeDownCast(chart->AddPlot(vtkChart::LINE));
    lineRule->SetInputData(table, 0, 1); lineRule->SetColor(0, 255, 100, 255); lineRule->SetWidth(2.5);

    vtkPlotLine* lineDem = vtkPlotLine::SafeDownCast(chart->AddPlot(vtkChart::LINE));
    lineDem->SetInputData(table, 0, 2); lineDem->SetColor(50, 150, 255, 255); lineDem->SetWidth(2.5);

    vtkPlotLine* linePhys = vtkPlotLine::SafeDownCast(chart->AddPlot(vtkChart::LINE));
    linePhys->SetInputData(table, 0, 3); linePhys->SetColor(255, 150, 0, 255); linePhys->SetWidth(2.5);

    vtkPlotLine* lineCorr = vtkPlotLine::SafeDownCast(chart->AddPlot(vtkChart::LINE));
    lineCorr->SetInputData(table, 0, 4); lineCorr->SetColor(255, 50, 50, 255); lineCorr->SetWidth(2.5);

    // 🌟 옵저버 연결 및 초기 데이터 세팅
    vtkSmartPointer<VDemKeyObserver> observer = vtkSmartPointer<VDemKeyObserver>::New();
    observer->allData = data;
    observer->countryCodes = countryCodes;
    observer->countryNames = countryNames;
    observer->groupName = groupName;
    observer->table = table;
    observer->chart = chart;
    observer->renderWindow = view->GetRenderWindow();

    observer->UpdateChart(); // 첫 번째 국가의 데이터를 그려놓습니다.

    // 키보드를 누를 때마다 옵저버가 작동하도록 연결합니다.
    view->GetInteractor()->AddObserver(vtkCommand::KeyPressEvent, observer);

    view->GetRenderWindow()->Render();
    return view;
}