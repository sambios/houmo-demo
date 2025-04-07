cmake_minimum_required(VERSION 3.15.3)
project(houmo-demo)

# set global variables
set(WS $ENV{HOME})
set(CMAKE_BUILD_TYPE Debug)

set(TCIM_RUNTIME_PATH $ENV{TCIM_RUNTIME_PATH})

set(TOP ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(OpenCV REQUIRED)

set(houmo_inc_dirs
        ${TCIM_RUNTIME_PATH}/include
        ${OpenCV_INCLUDE_DIRS}
        ${TOP}/common
        )

include_directories(${houmo_inc_dirs})
link_directories(${TCIM_RUNTIME_PATH}/lib)

message("houmo_inc_dirs=${houmo_inc_dirs}")

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else ()
    add_definitions(-w)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif ()

add_subdirectory(yolov5 yolov5.out)
add_subdirectory(yolov8 yolov8.out)
add_subdirectory(onnx onnx.out)