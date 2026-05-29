#pragma once
#include <vector>
#include "GdeltData.h"
#include <vtkSmartPointer.h>
#include <vtkContextView.h>

vtkSmartPointer<vtkContextView> ShowHistogram(const std::vector<GdeltPoint>& data, int numBins, double minVal, double maxVal);
vtkSmartPointer<vtkContextView> ShowHistogram(const std::vector<double>& data, const std::string& title, int numBins);