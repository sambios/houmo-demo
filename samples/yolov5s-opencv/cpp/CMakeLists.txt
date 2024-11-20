cmake_minimum_required(VERSION 3.16)
project(yolov5s)

set(CMAKE_CXX_STANDARD 17)
#add_definitions(-w)

find_package(OpenCV REQUIRED)

# houmo includes and libs
set(HOUMO_INCLUDE_DIRS
        $ENV{HOUMO_PATH}/include
        $ENV{TCIM_PATH}/include
        ${OpenCV_INCLUDE_DIRS}
)

set(HOUMO_LINK_DIRS
        $ENV{HOUMO_PATH}/lib
        $ENV{TCIM_PATH}/lib
)

message(STATUS "houmo_inc_dirs=${HOUMO_INCLUDE_DIRS}")
message(STATUS "houmo_link_dirs=${HOUMO_LINK_DIRS}")

include_directories(${HOUMO_INCLUDE_DIRS})
link_directories(${HOUMO_LINK_DIRS})

add_executable(yolov5s main.cpp
        YoloV5.cpp
        YoloV5.h
        base_model.h
        process.h
        process.cpp
        )
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS} tvm_runtime idnnl hdplrt pthread rt gomp)

add_executable(test_hdpl test_hdpl.cpp)
target_link_libraries(test_hdpl ${OpenCV_LIBS} tvm_runtime idnnl hdplrt pthread rt gomp)

add_executable(hm_model hmmodel.cpp)
target_link_libraries(hm_model ${OpenCV_LIBS} tvm_runtime idnnl hdplrt pthread rt gomp)