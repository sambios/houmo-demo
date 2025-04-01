cmake_minimum_required(VERSION 3.15.3)
project(yolov5)

set(WS /home/yuan)
#set(WS /workspace)

set(CMAKE_BUILD_TYPE Debug)

set(TCIM_RUNTIME_PATH ${WS}/houmo/work/houmo-tcim-runtime)

find_library(TCIM_LIBRARY NAMES tcim_runtime_lite PATHS ${TCIM_RUNTIME_PATH}/lib NO_DEFAULT_PATH)
find_package(OpenCV REQUIRED)
set(onnxruntime_DIR /home/yuan/work/onnxruntime)

set(houmo_inc_dirs
		${TCIM_RUNTIME_PATH}/include
		${OpenCV_INCLUDE_DIRS}
		)

include_directories(${houmo_inc_dirs} ${onnxruntime_DIR}/include)
link_directories(/home/yuan/work/onnxruntime/lib)

message("houmo_inc_dirs=${houmo_inc_dirs}")

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else ()
	add_definitions(-w)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif ()

# yolov8
add_executable(yolov8m yolov8m.cc)
target_link_libraries(yolov8m ${OpenCV_LIBS} tcim_runtime_lite)