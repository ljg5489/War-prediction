#pragma warning(disable: 4996)

#include "ScatterViewer.h"
#include <iostream>

#include <algorithm>               
#include <cmath>                   
#include <vtkUnsignedCharArray.h>  

#include <vtkContextScene.h>
#include <vtkChartXY.h>
#include <vtkPlotPoints.h>
#include <vtkTable.h>
#include <vtkFloatArray.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkAxis.h>       
#include <vtkTextProperty.h> 
#include <vtkCommand.h>
#include <map>
#include <vtkDoubleArray.h>
#include <vtkStringArray.h>
#include <vtkPen.h>

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


// ============================================================
//  3. ACLED Correlation Scatter Plot (단일 변수 고정)
// ============================================================
vtkSmartPointer<vtkContextView> ShowScatterPlot(
    const std::vector<AcledPoint>& data,
    const std::string& xVarName)
{
    if (data.empty()) return nullptr;

    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();
    vtkSmartPointer<vtkFloatArray> arrX = vtkSmartPointer<vtkFloatArray>::New();
    arrX->SetName(xVarName.c_str());
    vtkSmartPointer<vtkFloatArray> arrY = vtkSmartPointer<vtkFloatArray>::New();
    arrY->SetName("FATALITIES");

    int validCount = 0;
    for (const auto& pt : data) {
        float xVal = 0.0f;
        if (xVarName == "EVENTS") {
            xVal = static_cast<float>(pt.events);
        }
        else if (xVarName == "POPULATION_EXPOSURE") {
            xVal = static_cast<float>(pt.populationExposure);
        }
        else {
            std::cerr << "지원하지 않는 X축 변수입니다: " << xVarName << std::endl;
            return nullptr;
        }

        float yVal = static_cast<float>(pt.fatalities);
        arrX->InsertNextValue(xVal);
        arrY->InsertNextValue(yVal);
        ++validCount;
    }

    table->AddColumn(arrX);
    table->AddColumn(arrY);

    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    // ✅ 배경을 완전한 하얀색으로 변경
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);
    view->GetRenderWindow()->SetSize(900, 600);

    std::string windowTitle = "Correlation: " + xVarName + " vs FATALITIES";
    view->GetRenderWindow()->SetWindowName(windowTitle.c_str());

    vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    chart->SetShowLegend(false);

    chart->SetTitle(windowTitle.c_str());
    // ✅ 흰 배경에 맞춰 제목을 검은색으로 변경
    chart->GetTitleProperties()->SetColor(0.0, 0.0, 0.0);
    chart->GetTitleProperties()->SetFontSize(18);

    vtkAxis* xAxis = chart->GetAxis(vtkAxis::BOTTOM);
    xAxis->SetTitle(xVarName.c_str());
    xAxis->SetBehavior(vtkAxis::AUTO);
    // ✅ X축 글씨와 선을 모두 어둡게 변경
    xAxis->GetTitleProperties()->SetColor(0.0, 0.0, 0.0);
    xAxis->GetLabelProperties()->SetColor(0.0, 0.0, 0.0);
    xAxis->GetGridPen()->SetColor(220, 220, 220, 255); // 밝은 회색 그리드

    vtkAxis* yAxis = chart->GetAxis(vtkAxis::LEFT);
    yAxis->SetTitle("Fatalities (Y)");
    yAxis->SetBehavior(vtkAxis::AUTO);
    // ✅ Y축 글씨와 선을 모두 어둡게 변경
    yAxis->GetTitleProperties()->SetColor(0.0, 0.0, 0.0);
    yAxis->GetLabelProperties()->SetColor(0.0, 0.0, 0.0);
    yAxis->GetGridPen()->SetColor(220, 220, 220, 255);

    vtkPlotPoints* points = vtkPlotPoints::SafeDownCast(chart->AddPlot(vtkChart::POINTS));
    points->SetInputData(table, 0, 1);

    // ✅ 점을 통계학 표준 파란색(Blue) 계열로 변경 (반투명 적용)
    points->SetColor(50, 130, 220, 150);
    points->SetMarkerSize(3.0);
    points->SetMarkerStyle(vtkPlotPoints::CIRCLE);

    view->GetRenderWindow()->Render();
    return view;
}

// ============================================================
// 🌟 4. 방향키 입력을 감지하는 감시자(Observer) 클래스
// ============================================================
class AcledKeyObserver : public vtkCommand {
public:
    static AcledKeyObserver* New() { return new AcledKeyObserver; }

    std::vector<AcledPoint> allData;
    std::vector<std::string> xVarNames;
    int currentIndex = 0;

    vtkSmartPointer<vtkTable> table;
    vtkSmartPointer<vtkChartXY> chart;
    vtkSmartPointer<vtkRenderWindow> renderWindow;

    virtual void Execute(vtkObject* caller, unsigned long eventId, void* callData) override {
        vtkRenderWindowInteractor* iren = vtkRenderWindowInteractor::SafeDownCast(caller);
        if (!iren) return;

        std::string key = iren->GetKeySym();

        if (key == "Right") {
            currentIndex = (currentIndex + 1) % xVarNames.size();
            UpdateChart();
        }
        else if (key == "Left") {
            currentIndex = (currentIndex - 1 + xVarNames.size()) % xVarNames.size();
            UpdateChart();
        }
    }

    void UpdateChart() {
        std::string currentXVar = xVarNames[currentIndex];

        vtkFloatArray* arrX = vtkFloatArray::SafeDownCast(table->GetColumn(0));
        vtkFloatArray* arrY = vtkFloatArray::SafeDownCast(table->GetColumn(1));

        arrX->SetNumberOfValues(0);
        arrY->SetNumberOfValues(0);

        bool isCategorical = false;
        std::map<std::string, float> categoryMap;
        float nextCategoryId = 1.0f;

        std::vector<float> tempX;
        std::vector<float> tempY;
        std::vector<std::string> tempCat;

        // 1️⃣ 데이터를 임시 수집합니다.
        for (const auto& pt : allData) {
            tempY.push_back(static_cast<float>(pt.fatalities));

            if (currentXVar == "EVENTS") {
                tempX.push_back(static_cast<float>(pt.events));
            }
            else if (currentXVar == "POPULATION_EXPOSURE") {
                tempX.push_back(static_cast<float>(pt.populationExposure));
            }
            else {
                // 🌟 여기서 문자열(글자) 변수들을 감지합니다!
                isCategorical = true;
                std::string catVal = "";
                if (currentXVar == "REGION") catVal = pt.region;
                else if (currentXVar == "ADMIN1") catVal = pt.admin1;
                else if (currentXVar == "EVENT_TYPE") catVal = pt.eventType;
                else if (currentXVar == "SUB_EVENT_TYPE") catVal = pt.subEventType;
                else if (currentXVar == "DISORDER_TYPE") catVal = pt.disorderType;

                if (catVal.empty()) catVal = "Unknown";
                tempCat.push_back(catVal);
            }
        }

        // 2️⃣ 수치형 데이터일 경우 사분면(Quadrant)을 만들기 위해 평균(Mean)을 구합니다.
        float meanX = 0.0f, meanY = 0.0f;
        if (!isCategorical && !tempX.empty()) {
            for (float x : tempX) meanX += x;
            meanX /= tempX.size();
        }
        if (!tempY.empty()) {
            for (float y : tempY) meanY += y;
            meanY /= tempY.size();
        }

        // 3️⃣ 테이블에 변환된 최종 데이터를 밀어 넣습니다.
        for (size_t i = 0; i < tempY.size(); ++i) {
            float finalX = 0.0f;
            float finalY = 0.0f;

            if (isCategorical) {
                // [문자열 모드]: 글자를 좌표(1, 2, 3...)로 변환
                std::string cat = tempCat[i];
                if (categoryMap.find(cat) == categoryMap.end()) {
                    categoryMap[cat] = nextCategoryId++;
                }
                finalX = categoryMap[cat];
                finalY = tempY[i]; // 사망자 절대값
            }
            else {
                // [수치형 모드]: 평균 이동(Mean-Centering) 적용하여 사분면 십자가 중앙 정렬!
                finalX = tempX[i] - meanX;
                finalY = tempY[i] - meanY;
            }

            arrX->InsertNextValue(finalX);
            arrY->InsertNextValue(finalY);
        }

        table->Modified();

        // 4️⃣ 축(Axis) 이름과 눈금 디자인 변경
        vtkAxis* xAxis = chart->GetAxis(vtkAxis::BOTTOM);
        vtkAxis* yAxis = chart->GetAxis(vtkAxis::LEFT);

        if (isCategorical) {
            xAxis->SetTitle(currentXVar.c_str());
            yAxis->SetTitle("Fatalities (Raw Count)");

            // X축 밑에 숫자가 아닌 글자(Battles 등)를 직접 달아줍니다.
            vtkSmartPointer<vtkDoubleArray> customTicks = vtkSmartPointer<vtkDoubleArray>::New();
            vtkSmartPointer<vtkStringArray> customLabels = vtkSmartPointer<vtkStringArray>::New();
            for (const auto& pair : categoryMap) {
                customTicks->InsertNextValue(pair.second);
                customLabels->InsertNextValue(pair.first.c_str());
            }
            xAxis->SetCustomTickPositions(customTicks, customLabels);
            xAxis->GetLabelProperties()->SetOrientation(45); // 글자 겹침 방지 기울임
        }
        else {
            // 평균 이동이 적용되었음을 축에 명시합니다.
            std::string xLabel = currentXVar + " (Centered: X - Mean)";
            xAxis->SetTitle(xLabel.c_str());
            yAxis->SetTitle("Fatalities (Centered: Y - Mean)");

            // 기본 숫자형 축으로 원상복구
            xAxis->SetCustomTickPositions(nullptr, nullptr);
            xAxis->GetLabelProperties()->SetOrientation(0);
        }

        std::string newTitle = "Correlation: [" + currentXVar + "] vs [FATALITIES]";
        chart->SetTitle(newTitle.c_str());

        chart->RecalculateBounds();
        renderWindow->Render();
    }
};
// ============================================================
// 🌟 5. ACLED Interactive Correlation Scatter Plot 
// ============================================================
vtkSmartPointer<vtkContextView> ShowInteractiveScatterPlot(
    const std::vector<AcledPoint>& data,
    const std::vector<std::string>& xVarNames)
{
    if (data.empty() || xVarNames.empty()) return nullptr;

    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();
    vtkSmartPointer<vtkFloatArray> arrX = vtkSmartPointer<vtkFloatArray>::New();
    vtkSmartPointer<vtkFloatArray> arrY = vtkSmartPointer<vtkFloatArray>::New();

    arrX->SetName("X_Data");
    arrY->SetName("Y_Data");

    table->AddColumn(arrX);
    table->AddColumn(arrY);

    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0); // 교과서 스타일 흰 배경
    view->GetRenderWindow()->SetSize(900, 600);
    view->GetRenderWindow()->SetWindowName("Interactive Correlation Dashboard");

    vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    chart->SetShowLegend(false);

    chart->GetTitleProperties()->SetColor(0.0, 0.0, 0.0);
    chart->GetTitleProperties()->SetFontSize(18);

    vtkAxis* xAxis = chart->GetAxis(vtkAxis::BOTTOM);
    xAxis->SetBehavior(vtkAxis::AUTO);
    xAxis->GetTitleProperties()->SetColor(0.0, 0.0, 0.0);
    xAxis->GetLabelProperties()->SetColor(0.0, 0.0, 0.0);
    xAxis->GetGridPen()->SetColor(220, 220, 220, 255);

    vtkAxis* yAxis = chart->GetAxis(vtkAxis::LEFT);
    yAxis->SetBehavior(vtkAxis::AUTO);
    yAxis->GetTitleProperties()->SetColor(0.0, 0.0, 0.0);
    yAxis->GetLabelProperties()->SetColor(0.0, 0.0, 0.0);
    yAxis->GetGridPen()->SetColor(220, 220, 220, 255);

    vtkPlotPoints* points = vtkPlotPoints::SafeDownCast(chart->AddPlot(vtkChart::POINTS));
    if (points) {
        points->SetInputData(table, 0, 1);
        points->SetColor(50, 130, 220, 150); // 정통 통계학 파란색
        points->SetMarkerSize(3.5);
        points->SetMarkerStyle(vtkPlotPoints::CIRCLE);
    }

    vtkSmartPointer<AcledKeyObserver> observer = vtkSmartPointer<AcledKeyObserver>::New();
    observer->allData = data;
    observer->xVarNames = xVarNames;
    observer->table = table;
    observer->chart = chart;
    observer->renderWindow = view->GetRenderWindow();

    observer->UpdateChart();

    view->GetInteractor()->AddObserver(vtkCommand::KeyPressEvent, observer);

    view->GetRenderWindow()->Render();
    return view;
}

vtkSmartPointer<vtkContextView> ShowOutlierScatterPlot(const std::vector<double>& data, const std::string& variableName, double threshold) {
    if (data.empty()) return nullptr;

    // 1. 데이터 테이블 분리 (정상 데이터용, 이상치 데이터용)
    vtkSmartPointer<vtkTable> normalTable = vtkSmartPointer<vtkTable>::New();
    vtkSmartPointer<vtkFloatArray> nX = vtkSmartPointer<vtkFloatArray>::New(); nX->SetName("Index");
    vtkSmartPointer<vtkFloatArray> nY = vtkSmartPointer<vtkFloatArray>::New(); nY->SetName(variableName.c_str());
    normalTable->AddColumn(nX); normalTable->AddColumn(nY);

    vtkSmartPointer<vtkTable> outlierTable = vtkSmartPointer<vtkTable>::New();
    vtkSmartPointer<vtkFloatArray> oX = vtkSmartPointer<vtkFloatArray>::New(); oX->SetName("Index");
    vtkSmartPointer<vtkFloatArray> oY = vtkSmartPointer<vtkFloatArray>::New(); oY->SetName(variableName.c_str());
    outlierTable->AddColumn(oX); outlierTable->AddColumn(oY);

    // 2. MAD 및 경계값 계산
    std::vector<double> sortedData = data;
    std::sort(sortedData.begin(), sortedData.end());
    double median = (sortedData.size() % 2 == 0) ? (sortedData[sortedData.size() / 2 - 1] + sortedData[sortedData.size() / 2]) / 2.0 : sortedData[sortedData.size() / 2];

    std::vector<double> absDev;
    for (double val : data) absDev.push_back(std::abs(val - median));
    std::sort(absDev.begin(), absDev.end());
    double mad = (absDev.size() % 2 == 0) ? (absDev[absDev.size() / 2 - 1] + absDev[absDev.size() / 2]) / 2.0 : absDev[absDev.size() / 2];
    double scaledMad = mad * 1.4826;
    double range = threshold * scaledMad;
    double upperLimit = median + range;
    double lowerLimit = std::max(0.0, median - range);

    // 3. 데이터 샘플링 및 분리 입력
    // 4300만 개를 다 그리면 렌더링이 멈추므로 10만 개 단위로 샘플링합니다.
    int step = (data.size() > 100000) ? (int)(data.size() / 100000) : 1;
    float lastX = 0.0f;
    for (size_t i = 0; i < data.size(); i += step) {
        bool isOutlier = (scaledMad > 0.0 && std::abs(data[i] - median) > range);
        if (isOutlier) {
            oX->InsertNextValue(static_cast<float>(i));
            oY->InsertNextValue(static_cast<float>(data[i]));
        }
        else {
            nX->InsertNextValue(static_cast<float>(i));
            nY->InsertNextValue(static_cast<float>(data[i]));
        }
        lastX = static_cast<float>(i);
    }

    // 4. 차트 설정
    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);
    view->GetRenderWindow()->SetSize(900, 600);
    view->GetRenderWindow()->SetWindowName(("Outlier: " + variableName).c_str());

    vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    chart->SetShowLegend(false);

    // ✅ 중앙값을 화면 중앙에 배치
    chart->GetAxis(vtkAxis::LEFT)->SetRange(-20.0, 400);
    chart->GetAxis(vtkAxis::LEFT)->SetBehavior(vtkAxis::FIXED);
    chart->GetAxis(vtkAxis::BOTTOM)->SetRange(0, lastX);
    chart->GetAxis(vtkAxis::BOTTOM)->SetBehavior(vtkAxis::FIXED);

    // 5. 플롯 생성 (정상/이상치 각각 별도 객체로 생성)
    vtkPlotPoints* normalPoints = vtkPlotPoints::SafeDownCast(chart->AddPlot(vtkChart::POINTS));
    normalPoints->SetInputData(normalTable, 0, 1);
    normalPoints->SetColor(50, 180, 220, 150); // 파랑 (정상)
    normalPoints->SetMarkerSize(2.0);

    vtkPlotPoints* outlierPoints = vtkPlotPoints::SafeDownCast(chart->AddPlot(vtkChart::POINTS));
    outlierPoints->SetInputData(outlierTable, 0, 1);
    outlierPoints->SetColor(255, 60, 60, 200); // 빨강 (이상치)
    outlierPoints->SetMarkerSize(3.0);

    // 6. 기준선 세트 (중앙값, 상한선, 하한선)
    vtkSmartPointer<vtkTable> lineTable = vtkSmartPointer<vtkTable>::New();
    vtkSmartPointer<vtkFloatArray> lx = vtkSmartPointer<vtkFloatArray>::New(); lx->SetName("X");
    vtkSmartPointer<vtkFloatArray> lm = vtkSmartPointer<vtkFloatArray>::New(); lm->SetName("Median");
    vtkSmartPointer<vtkFloatArray> lu = vtkSmartPointer<vtkFloatArray>::New(); lu->SetName("Upper");
    vtkSmartPointer<vtkFloatArray> ll = vtkSmartPointer<vtkFloatArray>::New(); ll->SetName("Lower");

    lx->InsertNextValue(0); lx->InsertNextValue(lastX);
    lm->InsertNextValue(median); lm->InsertNextValue(median);
    lu->InsertNextValue(upperLimit); lu->InsertNextValue(upperLimit);
    ll->InsertNextValue(lowerLimit); ll->InsertNextValue(lowerLimit);

    lineTable->AddColumn(lx); lineTable->AddColumn(lm); lineTable->AddColumn(lu); lineTable->AddColumn(ll);

    auto addLine = [&](int colIdx, float r, float g, float b, bool dashed) {
        vtkPlot* p = chart->AddPlot(vtkChart::LINE);
        p->SetInputData(lineTable, 0, colIdx);
        p->SetColor(r, g, b, 255);
        p->SetWidth(2.0);
        if (dashed) p->GetPen()->SetLineType(vtkPen::DASH_LINE);
        };

    addLine(1, 30, 100, 200, false); // 중앙값 (파란 실선)
    addLine(2, 255, 60, 60, true);   // 상한선 (빨간 점선)
    addLine(3, 255, 60, 60, true);   // 하한선 (빨간 점선)

    chart->GetAxis(vtkAxis::BOTTOM)->SetTitle("Data Index (Sampled)");
    chart->GetAxis(vtkAxis::LEFT)->SetTitle(variableName.c_str());

    return view;
}