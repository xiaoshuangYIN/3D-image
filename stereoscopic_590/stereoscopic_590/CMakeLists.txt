cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(stereo-depth)

find_package(OpenCV REQUIRED)


include_directories(${GEO_ROOT}/libs/install/include)
link_directories(${GEO_ROOT}/libs/install/lib)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

include_directories(${OpenCv_INCLUDE_DIRS})

set(BuildFiles src/dataset.cpp)
LIST(APPEND BuildFiles src/error-metrics.cpp)
LIST(APPEND BuildFiles src/ncc.cpp)
LIST(APPEND BuildFiles src/graph-cut.cpp)

add_executable(stereo-depth src/main.cpp ${BuildFiles})
target_link_libraries(stereo-depth ${OpenCV_LIBS})

set(CMAKE_CXX_FLAGS "-std=c++11 -g -O3 -Wall" )
