//
// Created by yuan on 4/1/25.
//

#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <backward/strstream>
#include <numeric>

#if (__GNUC__ < 8 && !defined(_MSC_VER))
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else

#include <filesystem>

namespace fs = std::filesystem;
#endif

const std::vector<std::string> common_classes = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                                                 "truck", "boat", "traffic light",
                                                 "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                                                 "dog", "horse", "sheep", "cow",
                                                 "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                                                 "handbag", "tie", "suitcase", "frisbee",
                                                 "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                                 "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                                                 "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                                                 "apple", "sandwich", "orange",
                                                 "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                                                 "couch", "potted plant", "bed",
                                                 "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                                                 "keyboard", "cell phone", "microwave", "oven",
                                                 "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                                                 "teddy bear", "hair drier", "toothbrush"};


static const std::string ShapeToString(const std::vector<int64_t>& shape) {
    std::stringstream ss;
    for(int i = 0;i < shape.size(); ++i) {
        ss << shape[i];
        if (i < shape.size()-1) {
            ss << "x";
        }
    }
    return ss.str();
}



typedef struct {
    float conf{0.0f};
    int cls{-1};  // cls index
    std::string label; // cls_name
    cv::Rect box;
} Detection;

using OneImageDetectedBoxes = std::vector<Detection>;

class Detector {
public:
    struct Config{
        bool mIsPreReload, mIsPostReload;
        std::string mPreReloadFileName, mPostReloadFileName;
        bool mIsDumpFile = false;
    };
protected:
    Ort::Env mEnv;
    Ort::SessionOptions mSessionOption;
    std::shared_ptr<Ort::Session> mSessionPtr;
    std::vector<Ort::Value> mInputs;
    std::vector<std::unique_ptr<float>>mInputsPtrs;
    std::vector<Ort::Value> mOutputs;
    std::vector<std::string> mInputNames;
    std::vector<std::string> mOutputNames;
    std::vector<const char*> mInputNamesCC;
    std::vector<const char*> mOutputNamesCC;
    std::vector<std::unique_ptr<float>> mOutputPtrs;
    bool mIsPreReload, mIsPostReload;
    std::string mPreReloadFileName, mPostReloadFileName;
    bool mIsDumpFile = false;

public:
    Detector(bool pre_reload = false, bool post_reload = false): mIsPreReload(pre_reload), mIsPostReload(post_reload) {

    }

    virtual ~Detector(){
        mSessionOption.release();
        mSessionPtr->release();
    }

    int SetConfig(Config& conf) {
        mIsPreReload = conf.mIsPreReload;
        mPreReloadFileName = conf.mPreReloadFileName;
        mIsPostReload = conf.mIsPostReload;
        mPostReloadFileName = conf.mPostReloadFileName;
        mIsDumpFile = conf.mIsDumpFile;
        return 0;
    }

    int LoadFromModel(const std::string& model_file) {
        mEnv = Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "tool");

        Ort::SessionOptions mSessionOption;
        mSessionOption.SetInterOpNumThreads(1);
        mSessionOption.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
        mSessionPtr = std::make_shared<Ort::Session>(mEnv, model_file.c_str(), mSessionOption);

        auto providers = Ort::GetAvailableProviders();
        auto isEnableCuda = (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end());
        auto isEnableCPU = (std::find(providers.begin(), providers.end(), "CPUExecutionProvider") != providers.end());
        if (isEnableCuda) {
            std::cout << "CUDAExecutionProvider is available" << std::endl;
        }

        if (isEnableCPU) {
            std::cout << "CPUExecutionProvider is available" << std::endl;
            OrtSessionOptionsAppendExecutionProvider_CPU(mSessionOption, 0);
        }

        // read onnx model info
        Ort::AllocatorWithDefaultOptions allocator;
        std::cout << "Model Information:" << std::endl;
        size_t numInputNodes = mSessionPtr->GetInputCount();
        for(int i = 0; i< numInputNodes; ++i) {
            auto input_name = mSessionPtr->GetInputNameAllocated(i, allocator);
            mInputNames.push_back(input_name.get());
            mInputNamesCC.push_back(mInputNames.rbegin()->c_str());
            auto shape = mSessionPtr->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            auto type = mSessionPtr->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
            std::cout << " Input " << i << ":" << input_name.get() << "," << ShapeToString(shape) << type << std::endl;
        }

        size_t numOutputNodes = mSessionPtr->GetOutputCount();
        for(int i = 0; i< numOutputNodes; ++i) {
            auto outputName = mSessionPtr->GetOutputNameAllocated(i, allocator);
            mOutputNames.push_back(outputName.get());
            mOutputNamesCC.push_back(mOutputNames.rbegin()->c_str());
            auto shape = mSessionPtr->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            std::cout << " Output " << i << ":" << outputName.get() << "," << ShapeToString(shape) << std::endl;
        }

        return 0;
    }

    void DumpPreRun()
    {
        if (!mIsDumpFile) return;
        std::stringstream filename;
        for(int i = 0;i < mInputNamesCC.size(); ++i) {

            float* input_data = mInputs[i].GetTensorMutableData<float>();
            auto input_shape = mInputs[i].GetTensorTypeAndShapeInfo().GetShape();
            auto input_count = mInputs[i].GetTensorTypeAndShapeInfo().GetElementCount();

            filename << "onnx_before_run_" << mInputNames[i] << "_" + ShapeToString(input_shape) << ".bin";
            FILE *fp = fopen(filename.str().c_str(), "wb");
            fwrite(input_data, sizeof(float), input_count, fp);
            fclose(fp);

            if (mIsPreReload) {
                FILE* fp2 = fopen(mPreReloadFileName.c_str(), "rb");
                fread(input_data, 1, sizeof(float)*input_count, fp2);
                fclose(fp2);
            }
        }
    }

    void DumpAfterRun()
    {
        if (!mIsDumpFile) return;
        std::stringstream filename;
        for(int i = 0;i < mOutputs.size(); ++i) {
            Ort::Value& output = mOutputs[i];
            float* output_data = output.GetTensorMutableData<float>();
            auto output_shape = mOutputs[i].GetTensorTypeAndShapeInfo().GetShape();
            auto output_count = mOutputs[i].GetTensorTypeAndShapeInfo().GetElementCount();


            filename << "onnx_after_run_" << mOutputNames[i] << "_" + ShapeToString(output_shape) << ".bin";
            FILE *fp = fopen(filename.str().c_str(), "wb");
            fwrite(output_data, sizeof(float), output_count, fp);
            fclose(fp);

            if (mIsPostReload) {
                FILE* fp2 = fopen(mPostReloadFileName.c_str(), "rb");
                fread(output_data, 1, sizeof(float)*output_count, fp2);
                fclose(fp2);
            }
        }
    }

    int Infer(std::vector<cv::Mat>& images, std::vector<OneImageDetectedBoxes> &detections) {
        preprocess(images);
        try {
            DumpPreRun();
            mOutputs = mSessionPtr->Run(Ort::RunOptions{ nullptr }, mInputNamesCC.data(),
                    mInputs.data(), mInputs.size(),
                                        mOutputNamesCC.data(), mOutputNamesCC.size());

        }
        catch (std::exception e) {
            std::cout << e.what() << std::endl;
        }

        DumpAfterRun();

        return postprocess(images, detections);
    }


public:
    virtual int preprocess(std::vector<cv::Mat>& images) = 0;
    virtual int postprocess(std::vector<cv::Mat>& images, std::vector<OneImageDetectedBoxes>& boxes) = 0;
};


// yolov8m default Shape is [1, 84, 8400]
class YOLOV8 : public Detector
{
    struct BatchInfo {
        float x_factor;
        float y_factor;
    };
    std::vector<BatchInfo> mBatchInfo;

public:
    YOLOV8() {

    }

    ~YOLOV8() {

    }

    virtual int preprocess(std::vector<cv::Mat>& images) override
    {
        auto typeInfo = mSessionPtr->GetInputTypeInfo(0);
        auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto inputDims =typeInfo.GetTensorTypeAndShapeInfo().GetShape();
        assert(inputDims[0] == images.size());
        int inputTensorSize = std::accumulate(inputDims.begin(), inputDims.end(), 1, std::multiplies<int64_t>());
        std::unique_ptr<float> inputTensorMemPtr(new float[inputTensorSize]);
        auto input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, inputTensorMemPtr.get(), inputTensorSize,
                inputDims.data(), inputDims.size());
        mInputs.push_back(std::move(input_tensor_));
        int input_h = inputDims[2];
        int input_w = inputDims[3];

        // Copy data for every batch
        int inputTensorBatchSize = inputTensorSize/inputDims[0];
        for(int i = 0; i < images.size(); ++i) {
            auto image = images[i];

            int max = MAX(image.cols, image.rows);
            cv::Mat image2 = cv::Mat::zeros(cv::Size(max, max), CV_8UC3);
            cv::Rect roi(0, 0, image.cols, image.rows);
            image.copyTo(image2(roi));

            // fix bug, boxes consistence!
            BatchInfo batchInfo;
            batchInfo.x_factor = image2.cols / static_cast<float>(input_w);
            batchInfo.y_factor = image2.rows / static_cast<float>(input_h);
            mBatchInfo.push_back(batchInfo);


            cv::Mat blob = cv::dnn::blobFromImage(image2, 1/255.0, cv::Size(input_w, input_h),
                    cv::Scalar(0,0,0), true, false);

            float *offset = inputTensorMemPtr.get() + inputTensorBatchSize*i;
            memcpy(offset, blob.ptr<float>(), inputTensorBatchSize*sizeof(float));
        }
        // Save pointer
        mInputsPtrs.push_back(std::move(inputTensorMemPtr));
    }

    virtual int postprocess(std::vector<cv::Mat>& images, std::vector<OneImageDetectedBoxes>& detections) override
    {
        //YOLOv8 only one output tensor
        assert(mOutputs.size() == 1);
        const float* output_data = mOutputs[0].GetTensorData<float>();
        auto output_shape = mOutputs[0].GetTensorTypeAndShapeInfo().GetShape();
        auto output_count = mOutputs[0].GetTensorTypeAndShapeInfo().GetElementCount();

        int batchNum = images.size();
        assert(batchNum == output_shape[0]);
        for(int batchIdx = 0; batchIdx < batchNum; ++batchIdx) {
            int output_h = output_shape[1];
            int output_w = output_shape[2];
            int batchPlaneSize = std::accumulate(++output_shape.begin(),  output_shape.end(), 1, std::multiplies<int64_t>());

            cv::Mat dout(output_h, output_w, CV_32F, (float*)output_data + batchPlaneSize*batchIdx);
            cv::Mat det_output = dout.t(); // 8400x84

            std::vector<cv::Rect> boxes;
            std::vector<int> classIds;
            std::vector<float> confidences;

            for (int i = 0; i < det_output.rows; i++) {
                cv::Mat classes_scores = det_output.row(i).colRange(4, 84);
                cv::Point classIdPoint;
                double score;
                minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

                // 置信度 0～1之间
                if (score > 0.25)
                {
                    float cx = det_output.at<float>(i, 0);
                    float cy = det_output.at<float>(i, 1);
                    float ow = det_output.at<float>(i, 2);
                    float oh = det_output.at<float>(i, 3);
                    int x = static_cast<int>((cx - 0.5 * ow) * mBatchInfo[batchIdx].x_factor);
                    int y = static_cast<int>((cy - 0.5 * oh) * mBatchInfo[batchIdx].y_factor);
                    int width = static_cast<int>(ow * mBatchInfo[batchIdx].x_factor);
                    int height = static_cast<int>(oh * mBatchInfo[batchIdx].y_factor);
                    cv::Rect box;
                    box.x = x;
                    box.y = y;
                    box.width = width;
                    box.height = height;

                    boxes.push_back(box);
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back(score);
                }
            }

            OneImageDetectedBoxes oneImageBoxes;

            // NMS
            std::vector<int> indexes;
            cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
            for (size_t i = 0; i < indexes.size(); i++) {
                int index = indexes[i];
                int cls_id = classIds[index];

                Detection bbox;
                bbox.conf = confidences[index];
                bbox.label = common_classes[cls_id];
                bbox.cls = cls_id;
                bbox.box = boxes[index];
                oneImageBoxes.push_back(bbox);
            }
            std::cout << "detected: " << indexes.size() << std::endl;

            detections.push_back(oneImageBoxes);
        }
    }
};



int main(int argc, char** argv)
{
    //User should modify this file path.
    std::vector<std::string> image_files = {
    //    "/home/yuan/test/images/bus.jpg",
          "/home/yuan/test/images/zidane.jpg"
    };
    std::string onnx_path = "/home/yuan/test/models/yolov8m_640x640.onnx";
    int image_num = 0;

    if (argc > 1){
        onnx_path = argv[1];
    }

    YOLOV8 yolov8;
    Detector::Config conf;
    conf.mIsDumpFile = true;
    conf.mIsPreReload = false;
    conf.mPreReloadFileName = "prepare.bin";
    conf.mPostReloadFileName = "post.bin";
    conf.mIsPostReload = false;
    yolov8.SetConfig(conf);
    yolov8.LoadFromModel(onnx_path);

    std::vector<OneImageDetectedBoxes> allDetections;
    std::vector<cv::Mat> images;
    for(int i = 0;i < image_files.size(); ++i) {
        cv::Mat img_bgr = cv::imread(image_files[i]);
        images.push_back(img_bgr);
    }

    yolov8.Infer(images, allDetections);

    for(int batchIdx = 0; batchIdx < allDetections.size(); ++batchIdx) {
        auto detections = allDetections[batchIdx];
        for(auto detection:detections) {
            auto cls_id = detection.cls;
            cv::rectangle(images[batchIdx], detection.box, cv::Scalar(0, 0, 255), 2, 8);
            cv::rectangle(images[batchIdx], cv::Point(detection.box.tl().x, detection.box.tl().y - 20),
            cv::Point(detection.box.br().x, detection.box.tl().y), cv::Scalar(0, 255, 255), -1);
            putText(images[batchIdx], common_classes[cls_id], cv::Point(detection.box.tl().x, detection.box.tl().y),
                    cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
        }

        //Store file
        fs::path file_path(image_files[batchIdx]);
        fs::path result_path("demo_results");
        if (!fs::exists(result_path)) {
            fs::create_directory("demo_results");
        }
        fs::path result_file = result_path / file_path.filename();
        cv::imwrite(result_file.string().c_str(), images[batchIdx]);
        printf("demo results saved to %s\n", result_file.string().c_str());
    }

    return 0;
}
