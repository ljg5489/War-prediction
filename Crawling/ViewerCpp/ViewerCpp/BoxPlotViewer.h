#pragma once
#include <vector>
#include <string>
#include "GdeltData.h"
#include "AcledData.h"
#include <vtkSmartPointer.h>
#include <vtkContextView.h>

vtkSmartPointer<vtkContextView> ShowBoxPlot(const std::vector<GdeltPoint>& data, const std::string& columnName);
vtkSmartPointer<vtkContextView> ShowBoxPlot_Acled(const std::vector<AcledPoint>& data, const std::string& columnName);