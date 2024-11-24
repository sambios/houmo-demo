#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "dirent.h"

#ifdef __linux__
#include <unistd.h>
#endif

#include <sys/stat.h>
#include "YoloV5.h"
#include "hdpl/hdpl_runtime_api.h"

int main(int argc, char *argv[]) {
    std::cout.setf(std::ios::fixed);
    // get params
    const char *keys="{help | 0 | print help information.}"
                     "{model | ../../models/yolov5s-cars.hmm | model file path}"
                     "{dev_id | 0 | device id}"
                     "{conf_thresh | 0.57 | confidence threshold for filter boxes}"
                     "{nms_thresh | 0.5 | iou threshold for nms}"
                     "{input | ../../datasets/test | input path, images direction or video file path}"
                     "{classnames | ../../datasets/coco.names | class names file path}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    std::string model_file = parser.get<std::string>("model");
    std::cout << "Input model:" << model_file << std::endl;

    std::string input = parser.get<std::string>("input");
    std::cout << "Input dir:" << input << std::endl;

    int dev_id = parser.get<int>("dev_id");
    std::cout << "Input dev_id:" << dev_id << std::endl;

    struct stat stat;
    //Check model file whether is existed ?
    if (::stat(std::string(model_file + ".so").c_str(), &stat) != 0) {
        std::cout << "Cannot find valid model file." << std::endl;
        exit(1);
    }
    // CHeck coconames file is existed ?
    std::string coco_names = parser.get<std::string>("classnames");
    if (::stat(coco_names.c_str(), &stat) != 0) {
        std::cout << "Cannot find classnames file." << std::endl;
        exit(1);
    }

    // CHeck input directory is existed?
    if (::stat(input.c_str(), &stat) != 0){
        std::cout << "Cannot find input path." << std::endl;
        exit(1);
    }

    // creat handle
    auto ret = hdplSetDevice(dev_id);
    std::cout << "set device id: "  << dev_id << ", ret = " << ret << std::endl;

    // initialize net
    YoloV5 yolov5(model_file);
    CV_Assert(0 == yolov5.SetParams(
            parser.get<float>("conf_thresh"),
            parser.get<float>("nms_thresh"),
                    0.5,
            coco_names));

    // get batch_size
    int batch_size = yolov5.InputTensor(0)->BatchSize();

    // creat save path
    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK)
        mkdir("results/images", S_IRWXU);

    // test images
    if (stat.st_mode & S_IFDIR) {
        // get files
        std::vector<std::string> files_vector;
        DIR *pDir;
        struct dirent* ptr;
        pDir = opendir(input.c_str());
        while((ptr = readdir(pDir))!=0) {
            if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
                files_vector.push_back(input + "/" + ptr->d_name);
            }
        }
        closedir(pDir);
        std::sort(files_vector.begin(), files_vector.end());

        std::vector<hm::Bbox> boxes;
        int fileTotal = files_vector.size();
        int id = 0;
        for (auto file:files_vector){
            // file counter
            id++;
            // print current file name
            std::cout << id << "/" << fileTotal << ", img_file: " << file << std::endl;
            // read file
            cv::Mat image = cv::imread(file);
            if (image.empty()) continue;
            // predict
            CV_Assert(0 == yolov5.Detect(image, boxes));
            int i = 0;
            if (boxes.size() < 200) {
                for (auto &box: boxes) {
                    printf("[%d]:box.prob=%f, box.type = %d\n", i++, box.prob, box.type);

                    cv::rectangle(image, cv::Point(box.pos[0] * image.cols, box.pos[1] * image.rows),
                                  cv::Point(box.pos[2] * image.cols, box.pos[3] * image.rows), cv::Scalar(255, 178, 50), 2);
                }
            }else {
                // ERROR
                printf("result box is :%zu\n", boxes.size());
            }
            // save results
            size_t index = file.rfind("/");
            std::string img_name = file.substr(index + 1);
            std::string output_file = "results/images/" + img_name;
            std::cout << "output file:" << output_file << std::endl;
            bool success = cv::imwrite(output_file, image);
            if (!success) {
                std::cerr << "Failed to save image." << std::endl;
                return 1;
            }
        }


    }
    // test video
    else {
        cv::VideoCapture cap(input);
        if (!cap.isOpened()) {
            std::cout << "Open Video:" << input << " failed!" << std::endl;
            return -1;
        }

        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int fps = cap.get(cv::CAP_PROP_FPS);
        std::cout << "Capture: w=" << width << ",h=" << height << ", fps=" << fps << std::endl;

        while(true) {
            cv::Mat image;
            cap.read(image);
            if (image.empty()) {
                break;
            }

            std::vector<hm::Bbox>  bboxes;
            yolov5.Detect(image, bboxes);
            for(auto box: bboxes) {
                cv::rectangle(image, cv::Point(box.pos[0] * image.cols, box.pos[1] * image.rows),
                              cv::Point(box.pos[2] * image.cols, box.pos[3] * image.rows), cv::Scalar(255, 178, 50), 2);
            }

            cv::imshow("Test", image);
            if (cv::waitKey(40) == 'q') {
                break;
            }
        }

        cv::destroyWindow("Test");
        cap.release();
    }
    return 0;
}
