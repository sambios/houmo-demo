// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.

#include <iostream>
#include <sstream>
#include <string>

#if (__GNUC__ < 8 && !defined(_MSC_VER))
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else

#include <filesystem>

namespace fs = std::filesystem;
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "tcim/tcim_runtime.h"

#include "imageproc.hpp"
#include "imagenet.hpp"
#include "threads.hpp"
#include "utils.hpp"

//#define HM_VERSION 10400
#define HM_VERSION 20100

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



typedef struct {
    float conf{0.0f};
    int cls{-1};  // cls index
    std::string label; // cls_name
    cv::Rect box;
} Detection;


cv::Mat letterbox(cv::Mat &img, int height, int width) {
    float scale;
    int resize_rows;
    int resize_cols;
    if ((height * 1.0 / img.rows) < (width * 1.0 / img.cols)) {
        scale = height * 1.0 / img.rows;
    } else {
        scale = width * 1.0 / img.cols;
    }
    resize_cols = int(scale * img.cols);
    resize_rows = int(scale * img.rows);

    cv::resize(img, img, cv::Size(resize_cols, resize_rows));
    // Generate a gray image for letterbox using opencv
    cv::Mat img_new(width, height, CV_8UC3, cv::Scalar(0, 0, 0));
    int top = (height - resize_rows) / 2;
    int bot = (height - resize_rows + 1) / 2;
    int left = (width - resize_cols) / 2;
    int right = (width - resize_cols + 1) / 2;
    // Letterbox filling
    cv::copyMakeBorder(img, img_new, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return img_new;
}

// yolov8m default Shape is [1, 84, 8400]
void yolov8_post_process(float *pdata, std::vector<int64_t> output_shape, float x_factor, float y_factor,
        cv::Mat &frame, const std::vector<std::string> &labels, bool is_draw_box = true)
{

    int output_h = output_shape[1];
    int output_w = output_shape[2];

    cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);
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
            int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
            int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
            int width = static_cast<int>(ow * x_factor);
            int height = static_cast<int>(oh * y_factor);
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

    // NMS
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);

    std::vector<Detection> detections;
    printf("detect num: %d\n", (int) indexes.size());
    for (size_t i = 0; i < indexes.size(); i++) {
        int index = indexes[i];
        int cls_id = classIds[index];
        Detection detection;
        detection.box = boxes[index];
        detection.cls = cls_id;
        if (labels.size() > 0) {
            detection.label = labels[cls_id];
        }
        detection.conf = confidences[index];
        printf("box[%d, %d, %d, %d], conf:%f, cls:%d\n", boxes[index].x, boxes[index].y, boxes[index].width, boxes[index].height, confidences[index], cls_id);
        if (is_draw_box) {
            cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
            cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
                          cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
            putText(frame, labels[cls_id], cv::Point(boxes[index].tl().x, boxes[index].tl().y), cv::FONT_HERSHEY_PLAIN,
                    2.0, cv::Scalar(255, 0, 0), 2, 8);
        }
    }
//    std::cout << "detected: " << indexes.size() << std::endl;
//    if (is_draw_box) {
//        cv::imwrite("yolov8m_output_debug.jpg", frame);
//    }
}




int main(int argc, char *argv[]) {
    printf("\n===> yolov5s c++ example start...\n");
    printf("tcim version: %s\n", tcim::GetVersion().c_str());
    if (argc < 3) {
        printf("Usage: yolov5 model_file image_file\n");
        exit(1);
    }

    // 1. load model
    std::cout << "LoadFromFile yolov5s" << std::endl;
    std::string model_path = argv[1];
    if (!fs::exists(model_path)) {
        std::cerr << model_path << " not exist. you should run build.py in yolov5s example first." << std::endl;
        exit(-1);
    }
    auto module = tcim::Module::LoadFromFile(model_path);
    if (!module) {
        std::cerr << " load model " << model_path << " fail." << std::endl;
        exit(-1);
    }
    printf("model %s loaded.\n", model_path.c_str());

    // 2. get input info
    std::map<std::string, tcim::Tensor> input_map;
    int input_num = module.GetInputNum();
    int input_h = 0;
    int input_w = 0;
    std::cout << "Count of Input: " << input_num << std::endl;
    //only one input tensor is [images]
    assert(input_num == 1);
    for (int idx = 0; idx < input_num; idx++) {
        auto input_name = module.GetInputName(idx);
        auto input_info = module.GetInputInfo(input_name).AsContiguous();
        input_h = input_info.Shape()[2];
        input_w = input_info.Shape()[3];
        std::cout << "Input [" << input_name << "] :" << input_info << std::endl;
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        input_map.insert(std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }

    // 3. input preprocess
    std::string data_path = argv[2];
    if (!fs::exists(data_path)) {
        std::cerr << data_path << " not exist." << std::endl;
        exit(-1);
    }

    cv::Mat img_rgb;
    cv::Mat img_yuv;
    cv::Mat img_raw = cv::imread(data_path);
    cv::resize(img_raw, img_rgb, cv::Size(input_w, input_h));
    ImageProc::BgrToRgb((int8_t *) (img_rgb.data), img_rgb.rows, img_rgb.cols);
    cv::cvtColor(img_rgb, img_yuv, cv::COLOR_RGB2YUV_I420);
    int size = input_w * input_h * 3;
    ImageProc::I420To420sp((uint8_t *) input_map.at("images").Data(), (uint8_t *) img_yuv.data, size);

    // 4. get output info
    std::map<std::string, tcim::Tensor> output_map;
    int output_num = module.GetOutputNum();
    std::cout << "Count of Output: " << output_num << std::endl;
    for (int idx = 0; idx < output_num; idx++) {
        auto output_name = module.GetOutputName(idx);
        auto output_info = module.GetOutputInfo(output_name).AsContiguous().AsType(tcim::FLOAT32);
        std::cout << "Output[" << output_name << "] " << output_info << std::endl;
        auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
        output_map.insert(std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
    }

    // 5. set input
    for (const auto &input : input_map) {
        module.SetInput(input.first, input.second);
    }

    // 6. run and sync
    module.Run();
    module.Sync();

    // 7. get output
    std::vector<tcim::Tensor> output_blobs;
    for (auto &output : output_map) {
#if HM_VERSION < 20000
	module.GetOutput(output.first, output.second);
#else
        auto output_tensor = module.GetOutput(output.first);
        output_tensor.CastTo(output.second);
        output_blobs.push_back(output.second);
#endif
    }

    // 8. postprocess
    assert(output_blobs.size() == 1);
    float *output_data = (float*)output_blobs[0].Data();

    bool is_save_file = true;
    float x_factor = img_raw.cols / static_cast<float>(input_w);
    float y_factor = img_raw.rows / static_cast<float>(input_h);

    yolov8_post_process(output_data, output_blobs[0].Info().Shape(), x_factor, y_factor, img_raw, common_classes, is_save_file);
    if (is_save_file) {
        fs::path file_path(data_path);
        fs::path result_path("demo_results");
        if (!fs::exists(result_path)) {
            fs::create_directory("demo_results");
        }
        fs::path result_file = result_path / file_path.filename();
        cv::imwrite(result_file.string().c_str(), img_raw);
        printf("demo results saved to %s\n", result_file.string().c_str());
    }

    printf("<=== yolov8m c++ example completed.\n\n");
    return 0;
}
