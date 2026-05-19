#pragma warning(disable: 4996) 
#include "ToneDropViewer.h"
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
#include <vtkTextProperty.h> // 저번의 에러 방지용!
#include <vtkPen.h>

vtkSmartPointer<vtkContextView> ShowToneDropChart(
    const std::vector<GdeltPoint>& data,
    const std::string& targetCountryCode,
    int windowSize)
{
    if (data.empty()) return nullptr;

    // 1. 날짜별로 톤(AvgTone) 합산 및 사건 개수 카운트 (일일 평균을 내기 위함)
    // std::map을 쓰면 날짜(YYYYMMDD 문자열)가 자동으로 오름차순(과거->현재) 정렬됩니다.
    std::map<std::string, std::pair<double, int>> dailyToneMap;

    for (const auto& pt : data) {
        if (pt.countryCode == targetCountryCode) {
            if (!pt.sqlDate.empty()) {
                dailyToneMap[pt.sqlDate].first += pt.avgTone;  // 톤 합계
                dailyToneMap[pt.sqlDate].second += 1;          // 기사/사건 개수
            }
        }
    }

    if (dailyToneMap.empty()) {
        std::cerr << "GDELT Data not found for country code: " << targetCountryCode << std::endl;
        return nullptr;
    }

    // 2. 날짜별 순수 일일 평균 톤 산출
    std::vector<double> dailyAvgTones;
    for (const auto& pair : dailyToneMap) {
        double dailyTone = pair.second.first / pair.second.second;
        dailyAvgTones.push_back(dailyTone);
    }

    // 3. 🌟 핵심: 14일 이동평균(Moving Average) 계산 🌟
    std::vector<double> movingAverages(dailyAvgTones.size(), 0.0);
    for (size_t i = 0; i < dailyAvgTones.size(); ++i) {
        // 초반부(14일이 안 쌓인 기간)는 쌓인 만큼만 평균을 냅니다 (0으로 떨어지는 것 방지)
        if (i < windowSize - 1) {
            double sum = 0;
            for (size_t j = 0; j <= i; ++j) sum += dailyAvgTones[j];
            movingAverages[i] = sum / (i + 1);
        }
        // 14일이 꽉 찼을 때는 정확히 직전 14일 치 평균을 냅니다.
        else {
            double sum = 0;
            for (size_t j = i - windowSize + 1; j <= i; ++j) sum += dailyAvgTones[j];
            movingAverages[i] = sum / windowSize;
        }
    }

    // 4. VTK 테이블에 데이터 삽입
    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();
    vtkSmartPointer<vtkIntArray> arrX = vtkSmartPointer<vtkIntArray>::New(); arrX->SetName("Days");
    vtkSmartPointer<vtkFloatArray> arrY = vtkSmartPointer<vtkFloatArray>::New(); arrY->SetName("MA_Tone");
    vtkSmartPointer<vtkFloatArray> arrZero = vtkSmartPointer<vtkFloatArray>::New(); arrZero->SetName("ZeroLine"); // 기준선

    for (size_t i = 0; i < movingAverages.size(); ++i) {
        arrX->InsertNextValue(static_cast<int>(i));
        arrY->InsertNextValue(static_cast<float>(movingAverages[i]));
        arrZero->InsertNextValue(0.0f); // 위험 임계선 (어조가 0 이하로 떨어지면 위험)
    }

    table->AddColumn(arrX);
    table->AddColumn(arrY);
    table->AddColumn(arrZero);

    // 5. 뷰어 및 렌더링 세팅
    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(0.08, 0.1, 0.14); // 심해 다크 모드
    view->GetRenderWindow()->SetSize(1000, 500); // 가로로 긴 시계열 최적화 비율
    view->GetRenderWindow()->SetWindowName(("Media Tone Drop - " + targetCountryCode).c_str());

    vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    chart->SetShowLegend(true);

    chart->SetTitle(("14-Day Moving Average Media Tone : " + targetCountryCode).c_str());
    chart->GetTitleProperties()->SetColor(0.9, 0.9, 0.9);
    chart->GetTitleProperties()->SetFontSize(20);

    chart->GetAxis(vtkAxis::BOTTOM)->SetTitle("Time (Days)");
    chart->GetAxis(vtkAxis::LEFT)->SetTitle("Average Tone (-10 to 10)");

    // 🔵 14일 이동평균 메인 선 (차갑고 날카로운 사이언 색상)
    vtkPlotLine* lineTone = vtkPlotLine::SafeDownCast(chart->AddPlot(vtkChart::LINE));
    lineTone->SetInputData(table, 0, 1);
    lineTone->SetColor(0, 200, 255, 255);
    lineTone->SetWidth(2.5);

    // 🔴 임계 기준선 (Y = 0) (이 선을 밑으로 뚫으면 위험)
    vtkPlotLine* lineZero = vtkPlotLine::SafeDownCast(chart->AddPlot(vtkChart::LINE));
    lineZero->SetInputData(table, 0, 2);
    lineZero->SetColor(255, 50, 50, 200); // 붉은색 경고선
    lineZero->SetWidth(1.5);

    view->GetRenderWindow()->Render();
    return view;
}