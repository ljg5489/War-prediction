#pragma once
#include <vector>
#include <string>
#include "GdeltData.h"
#include <vtkSmartPointer.h>
#include <vtkContextView.h>

vtkSmartPointer<vtkContextView> ShowBoxPlot(const std::vector<GdeltPoint>& data, const std::string& columnName);