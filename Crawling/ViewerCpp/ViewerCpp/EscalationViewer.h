#pragma once
#include "AcledData.h"
#include <vector>
#include <string>
#include <vtkSmartPointer.h>
#include <vtkContextView.h>

// 에스컬레이션 가속도 추이 뷰어 선언
vtkSmartPointer<vtkContextView> ShowEscalationChart(
    const std::vector<AcledPoint>& data,
    const std::string& targetCountry
);