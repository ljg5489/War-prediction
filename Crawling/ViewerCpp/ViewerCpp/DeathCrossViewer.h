#pragma once
#include "GdeltData.h"
#include <vector>
#include <string>
#include <vtkSmartPointer.h>
#include <vtkContextView.h>

// 🌟 고립 및 위협 이벤트 비중 비교(데스크로스) 뷰어 선언
vtkSmartPointer<vtkContextView> ShowDeathCrossChart(
    const std::vector<GdeltPoint>& data,
    const std::string& targetCountryCode
);