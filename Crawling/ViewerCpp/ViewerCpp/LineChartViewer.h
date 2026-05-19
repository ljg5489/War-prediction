#pragma once
#include "GdeltData.h"
#include <vector>
#include <string>
#include <vtkSmartPointer.h>
#include <vtkContextView.h>

vtkSmartPointer<vtkContextView> ShowLineChart(const std::vector<GdeltPoint>& data, const std::string& columnName);