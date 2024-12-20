//
// Created by huaishan.yuan on 24-11-9.
//

#ifndef YOLOV5S_BASE_MODEL_H
#define YOLOV5S_BASE_MODEL_H

#include <iostream>
#include <map>
#include <cassert>
#include <sstream>
#include <random>
#include "tcim/tcim_runtime.h"
#include "hdpl/hdpl_runtime_api.h"


#define TCIM_FUNC_CHECK(exp) do { \
                                  auto ret = (exp); \
                                  if (tcim::OK != ret) { \
                                    std::cout << #exp << " ret=" << ret << std::endl; \
                                  }\
                            }while(0)

#define TCIM_FUNC_ASSERT(exp) do { \
                                  auto ret = (exp); \
                                  if (tcim::OK != ret) { \
                                    std::cout << #exp << " ret=" << ret << std::endl; \
                                    return ret;\
                                  }\
                            }while(0)

namespace hm {
    struct Utils {
        static cv::Mat letterBox(const cv::Mat& image, int width, int height)
        {
            float scale_x = width / (float)image.cols;
            float scale_y = height / (float)image.rows;
            float scale = std::min(scale_x, scale_y);
            float i2d[6], d2i[6];

            i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + width + scale  - 1) * 0.5;
            i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + height + scale - 1) * 0.5;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

            cv::Mat dst_image(height, width, CV_8UC3);
            cv::warpAffine(image, dst_image, m2x3_i2d, dst_image.size(), cv::INTER_LINEAR,
                           cv::BORDER_CONSTANT, cv::Scalar::all(96));
            return dst_image;
        }

        static void convertI420toNV12(uint8_t *i420bytes, uint8_t *nv12bytes, int width, int height) {
            int lenY = width*height;
            int lenU = lenY >> 2;
            memcpy(nv12bytes, i420bytes, lenY);
            for(int i = 0; i < lenU; ++i) {
                nv12bytes[lenY + 2*i] = i420bytes[lenY + i];
                nv12bytes[lenY + 2*i + 1] = i420bytes[lenY + lenU + i];
            }
        }

    }; //~ struct utils

    class Profile {
        int64_t start_us_;
        std::string tag_;
        int threshold_{50};
    public:
        Profile() {}
        Profile(const std::string &name, int threshold = 0) {
            begin(name, threshold);
        }
        ~Profile() {}

        void begin(const std::string &name, int threshold = 0) {
            tag_ = name;
            auto n = std::chrono::steady_clock::now();
            start_us_ = n.time_since_epoch().count();
            threshold_ = threshold;
        }

        void end() {
            auto n = std::chrono::steady_clock::now().time_since_epoch().count();
            auto delta = (n - start_us_) / 1000;
            if (delta < threshold_ * 1000) {
                //printf("%s used:%d us\n", tag_.c_str(), delta);
            } else {
                printf("WARN:%s used:%d ms > %d ms\n", tag_.c_str(), (int)delta/1000, (int)threshold_);
            }
        }
    };

    class HMTensor {
    public:
        //Disable assign and copy
        HMTensor(const HMTensor &rhs) = delete;

        HMTensor &operator=(const HMTensor &other) = delete;

        // constructor
        HMTensor(const std::string &name, const tcim::TensorInfo &tensorInfo) : m_tensorInfo(tensorInfo),
                                                                                m_devMemPtr(nullptr),
                                                                                m_hostMemPtr(nullptr), m_name(name) {
            //std::cout << "HMTensor ctor (name=" << name << "), tensorInfo=" << tensorInfo << std::endl;
        }

        virtual ~HMTensor() {
            //std::cout << "HMTensor " << m_name << " Dtor .. " << std::endl;
            if (m_hostMemPtr != nullptr) {
                free(m_hostMemPtr);
            }

            if (m_devMemPtr != nullptr) {
                hdplFree(m_devMemPtr);
            }
        }

        tcim::Tensor &Tensor() {
            if (nullptr == m_tensorPtr) {
                // construct a new tensor
                if (m_tensorInfo.Device() == tcim::HDPL) {
                    if (bestDevMemPtr() == nullptr) {
                        AllocDeviceMemory();
                    }
                    m_tensorPtr = std::make_shared<tcim::Tensor>(m_tensorInfo, bestDevMemPtr(), m_tensorInfo.MemSize());
                } else {
                    if (nullptr == bestHostMemPtr()) {
                        AllocHostMemory();
                    }
                    m_tensorPtr = std::make_shared<tcim::Tensor>(m_tensorInfo, bestHostMemPtr(), m_tensorInfo.MemSize());
                }
            }
            return *m_tensorPtr;
        }

        int AllocDeviceMemory() {
            if (m_customDevMemPtr != nullptr) {
                std::cout << "Tensor's custom device memory is set, please check!" << std::endl;
                return -1;
            }

            if (m_devMemPtr != nullptr) return 0;
            auto ret = hdplMalloc(&m_devMemPtr, m_tensorInfo.MemSize());
            if (hdplSuccess != ret) {
                std::cout << "Alloc device memory failed, size=" << m_tensorInfo.MemSize() << std::endl;
                return -1;
            }
            return ret;
        }

        void *DeviceMemoryPtr() {
            return bestDevMemPtr();
        }

        int AllocHostMemory() {
            if (m_customHostMemPtr != nullptr) {
                std::cout << "Tensor's custom host memory is set, please check!" << std::endl;
                return -1;
            }
            if (m_hostMemPtr != nullptr) return 0;
            m_hostMemPtr = malloc(m_tensorInfo.MemSize());
            if (nullptr != m_hostMemPtr) return -1;
            return 0;
        }

        int SetHostMemory(void *userDefinePtr) {
            m_customHostMemPtr = userDefinePtr;
            return 0;
        }

        int SetDeviceMemory(void *userDefinePtr) {
            m_customDevMemPtr = userDefinePtr;
            return 0;
        }

        void *HostMemoryPtr() {
            return bestHostMemPtr();
        }

        int HostToDeviceSync() {
            return hdplMemcpy(bestDevMemPtr(), bestHostMemPtr(), m_tensorInfo.MemSize(), hdplMemcpyHostToDevice);
        }

        int DeviceToHostSync() {
            return hdplMemcpyDtoH(bestHostMemPtr(), bestDevMemPtr(), m_tensorInfo.MemSize());
        }

        int BatchSize() {
            return (int) m_tensorInfo.Shape()[0];
        }

        int Width() {
            return (int) m_tensorInfo.Shape()[2];
        }

        int Height() {
            return (int) m_tensorInfo.Shape()[1];
        }

        const std::vector<int64_t> &Shape() {
            return m_tensorInfo.Shape();
        }

        // for debug
        void GenData(bool random, int8_t preset = 0) {
            int8_t* p = (int8_t*)bestHostMemPtr();
            if (random) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<int> dist(-128, 127);
                for (int i = 0; i < m_tensorInfo.MemSize(); ++i) {
                    p[i] = dist(gen);
                }
            } else {
                memset(p, preset, m_tensorInfo.MemSize());
            }
        }

    protected:
        inline void *bestHostMemPtr() {
            return m_customHostMemPtr == nullptr ? m_hostMemPtr : m_customHostMemPtr;
        }

        inline void *bestDevMemPtr() {
            return m_customDevMemPtr == nullptr ? m_devMemPtr :  m_customDevMemPtr;
        }

        void *m_devMemPtr = nullptr;
        void *m_customDevMemPtr = nullptr;
        void *m_hostMemPtr = nullptr;
        void *m_customHostMemPtr = nullptr;
        tcim::TensorInfo m_tensorInfo;
        std::string m_name;
        tcim::Device m_device = tcim::CPU;
        std::shared_ptr<tcim::Tensor> m_tensorPtr;
    };

    using HMTensorPtr = std::shared_ptr<HMTensor>;

    class HMBaseModel {
        tcim::Module m_module;
        tcim::Device m_device;
        std::vector<HMTensorPtr> m_inputTensors;
        std::vector<HMTensorPtr> m_outputTensors;

    public:
        explicit HMBaseModel(const std::string &model_file, tcim::Device device = tcim::CPU) {
            m_module = tcim::Module::LoadFromFile(model_file);
            m_device = device;
            auto inputNum = m_module.GetInputNum();
            for (int i = 0; i < inputNum; i++) {
                std::string inputName = m_module.GetInputName(i);
                tcim::TensorInfo inputInfo;
                m_module.GetInputInfo(inputName, inputInfo, device);
                HMTensorPtr tensorPtr = std::make_shared<HMTensor>(inputName, inputInfo);
                m_inputTensors.push_back(tensorPtr);
                m_module.SetInput(inputName, tensorPtr->Tensor());
            }

            auto outputNum = m_module.GetOutputNum();
            for (int i = 0; i < outputNum; i++) {
                std::string outputName = m_module.GetOutputName(i);
                //std::cout << "outputName: " << outputName << std::endl;

                tcim::TensorInfo outputInfo;
                // NOTE::Output not support CPU now
                m_module.GetOutputInfo(outputName, outputInfo, tcim::HDPL, true);

                HMTensorPtr tensorPtr = std::make_shared<HMTensor>(outputName, outputInfo);
                m_module.SetOutput(outputName, tensorPtr->Tensor());
                m_outputTensors.push_back(tensorPtr);

            }
        }

        virtual ~HMBaseModel() {
            std::cout << "HMBaseModel dtor" << std::endl;
        }

        size_t InputTensorNum() {
            return m_inputTensors.size();
        }

        size_t OutputTensorNum() {
            return m_outputTensors.size();
        }

        HMTensorPtr InputTensor(int i) {
            if (i >= m_inputTensors.size()) return nullptr;
            return m_inputTensors[i];
        }

        HMTensorPtr OutputTensor(int i) {
            if (i >= m_outputTensors.size()) return nullptr;
            return m_outputTensors[i];
        }

        virtual int inference() {
            auto ret = m_module.Run();
            if (tcim::OK != ret) {
                std::cout << "module.Run ret=" << ret << std::endl;
                return ret;
            }

            ret = m_module.Sync();
            if (tcim::OK != ret) {
                std::cout << "module.Run ret=" << ret << std::endl;
                return ret;
            }

            return 0;
        };

        std::string ShapeToString(const std::vector<int64_t> &shape) {
            std::stringstream oss;
            oss << "[";
            for (int i = 0; i < shape.size(); ++i) {
                oss << shape[i];
                if (i != shape.size() - 1) oss << ",";
            }
            oss << "]";
            return oss.str();
        }

        void PrintInfo(bool is_quanted = true) {
            //std::cout << "Model Version:" << tcim::Module::GetModelVersion() << std::endl;

            auto inputNum = m_module.GetInputNum();
            auto outputNum = m_module.GetOutputNum();

            for (int i = 0; i < inputNum; ++i) {
                // input: name, tensor info
                std::string name = m_module.GetInputName(i);
                std::cout << "inputName: " << name << ", " << m_inputTensors[i]->Tensor().Info() << std::endl;
            }

            for (int i = 0; i < outputNum; ++i) {
                // output: name, tensor info
                std::string name = m_module.GetOutputName(i);
                std::cout << "output: " << name << ", " << m_outputTensors[i]->Tensor().Info() << std::endl;
            }
        }
    };
} //:~ NS(hm)



#endif //YOLOV5S_BASE_MODEL_H
