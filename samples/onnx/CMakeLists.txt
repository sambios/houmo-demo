cmake_minimum_required(VERSION 3.15.3)
project(onnxtool)

# for onnxruntime
if (DEFINED ENV{ONNX_RUNTIME_PATH})
    set(ONNX_RUNTIME_PATH $ENV{ONNX_RUNTIME_PATH})
else (NOT DEFINED ENV{ONNX_RUNTIME_PATH})
    set(ONNX_RUNTIME_PATH /home/yuan/work/onnxruntime)
endif ()


set(onnxruntime_inc_dirs ${ONNX_RUNTIME_PATH}/include)
set(onnxruntime_link_dirs ${ONNX_RUNTIME_PATH}/lib)
set(onnxruntime_libs onnxruntime)


add_executable(onnxtool onnx_tool.cpp)
target_link_libraries(onnxtool ${OpenCV_LIBS} ${onnxruntime_libs})
target_include_directories(onnxtool PRIVATE ${onnxruntime_inc_dirs})
target_link_directories(onnxtool PRIVATE ${onnxruntime_link_dirs})