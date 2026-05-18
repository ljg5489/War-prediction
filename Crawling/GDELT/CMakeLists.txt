cmake_minimum_required(VERSION 3.12)
project(GdeltVtkMacroViewer)

set(CMAKE_CXX_STANDARD 17)

# VTK 구성요소 자동 탐색
find_package(VTK REQUIRED COMPONENTS
  CommonCore
  CommonDataModel
  FiltersGeneral
  InteractionStyle
  RenderingContextOpenGL2
  RenderingCore
  RenderingFreeType
  RenderingOpenGL2
  InteractionWidgets
)

add_executable(GdeltVtkViewer main.cpp)

# VTK 라이브러리 링크 및 인클루드 패스 연결
vtk_module_autoinit(
  TARGETS GdeltVtkViewer
  MODULES VTK::RenderingOpenGL2 VTK::InteractionStyle
)

target_link_libraries(GdeltVtkViewer PRIVATE ${VTK_LIBRARIES})