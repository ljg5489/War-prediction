#pragma once
#include "GdeltData.h"
#include <vector>
#include <string>
#include <vtkSmartPointer.h>
#include <vtkContextView.h>

// 🌟 미디어 톤 급락 추적 뷰어 선언
// 기본적으로 14일(2주) 이동평균을 사용하도록 windowSize의 기본값을 14로 줍니다.
vtkSmartPointer<vtkContextView> ShowToneDropChart(
    const std::vector<GdeltPoint>& data,
    const std::string& targetCountryCode,
    int windowSize = 14
);