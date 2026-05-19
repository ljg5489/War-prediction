#pragma once
#include "VDemData.h"
#include <vector>
#include <string>
#include <vtkSmartPointer.h>
#include <vtkContextView.h>

// 방향키로 상호작용하는 그룹 전용 뷰어
vtkSmartPointer<vtkContextView> ShowInteractiveGroupChart(
    const std::vector<VDemPoint>& data,
    const std::string& groupName,
    const std::vector<std::string>& countryCodes,
    const std::vector<std::string>& countryNames
);