#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <windows.h> 

#include "GdeltData.h"
#include "HistogramViewer.h"
#include "BoxPlotViewer.h"
#include "ScatterViewer.h" 

#include <vtkRenderWindowInteractor.h>
#include <vtkAutoInit.h>

VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingContextOpenGL2);

int main() {
    SetConsoleOutputCP(CP_UTF8);

    std::string csvPath = "..\\..\\GDELT\\GDELT_Daily_All_Countries.csv";

    std::vector<GdeltPoint> rawData;
    std::ifstream file(csvPath);

    if (!file.is_open()) {
        std::cerr << "File Open Failed!" << std::endl;
        return -1;
    }

    std::string line;
    std::getline(file, line); // Header skip

    std::cout << "Loading..." << std::endl;

    // 예전 속도 그대로 나오는 순정 로딩부
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;

        while (std::getline(ss, cell, ',')) row.push_back(cell);

        if (row.size() > 3 && !row[3].empty()) {
            try {
                GdeltPoint pt;
                pt.goldstein = std::stod(row[3]);
                rawData.push_back(pt);
            }
            catch (...) { continue; }
        }
    }
    file.close();
    std::cout << "Load Complete!" << std::endl;

    // 파이프라인 가동
    auto view1 = ShowHistogram(rawData, 50, -10.0, 10.0);
    auto view2 = ShowBoxPlot(rawData, "Goldstein Scale");
    auto view3 = ShowScatterPlot(rawData, "Goldstein Scale");

    if (view1 != nullptr) {
        view1->GetInteractor()->Start();
    }

    return 0;
}