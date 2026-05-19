#pragma once

#include "VDemData.h" // VDemPoint 구조체 인식
#include <vector>
#include <string>
#include <vtkSmartPointer.h>
#include <vtkContextView.h>

// V-Dem 다중 꺾은선 차트 뷰어 선언
vtkSmartPointer<vtkContextView> ShowVDemMultiLine(const std::vector<VDemPoint>& data, const std::string& targetCountry);