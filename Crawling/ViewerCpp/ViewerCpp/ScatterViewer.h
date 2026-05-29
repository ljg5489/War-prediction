#pragma once
#include <vector>
#include <string>
#include "GdeltData.h"
#include "AcledData.h"
#include "VDemData.h"
#include <vtkSmartPointer.h>
#include <vtkContextView.h>

vtkSmartPointer<vtkContextView> ShowScatterPlot(const std::vector<GdeltPoint>& data, const std::string& columnName);
vtkSmartPointer<vtkContextView> ShowScatterPlot(const std::vector<AcledPoint>& data);
// 3. ACLED 단일 변수 상관성 산점도 (Y: Fatalities 고정)
vtkSmartPointer<vtkContextView> ShowScatterPlot(const std::vector<AcledPoint>& data, const std::string& xVarName);
vtkSmartPointer<vtkContextView> ShowInteractiveScatterPlot(const std::vector<AcledPoint>& data, const std::vector<std::string>& xVarNames);
vtkSmartPointer<vtkContextView> ShowOutlierScatterPlot(const std::vector<double>& data, const std::string& variableName, double threshold = 3.0);