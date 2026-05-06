#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "face_recognizer.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

void InitConsoleEncoding() {
#ifdef _WIN32
    // 主判断：Windows 控制台默认可能不是 UTF-8，这里主动切换，避免中文输出乱码。
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

torch::Device ResolveDevice(const std::string& device_name) {
    if (device_name == "cpu") {
        return torch::Device(torch::kCPU);
    }
    if (device_name == "cuda") {
        // 主判断：用户显式要求 CUDA 时，如果不可用就直接报错。
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("当前环境不可用 CUDA，请改用 --device=cpu 或 --device=auto");
        }
        return torch::Device(torch::kCUDA);
    }
    // 主判断：auto 模式优先选择 CUDA，否则回退到 CPU。
    if (torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA);
    }
    return torch::Device(torch::kCPU);
}

void PrintUsage() {
    std::cout
        << "用法:\n"
        << "  face_image_demo --mtcnn_dir=<MTCNN模型目录> --model_path=<MobileFaceNet模型路径> "
           "--face_db=<人脸库目录> --image_path=<图片路径> "
           "[--device=auto|cpu|cuda] [--threshold=0.6] [--save_path=<输出路径>] [--show=0|1]\n";
}

}  // namespace

int main(int argc, char** argv) {
    InitConsoleEncoding();

    const cv::CommandLineParser parser(
        argc,
        argv,
        "{help h||显示帮助}"
        "{mtcnn_dir||MTCNN 模型目录（包含 PNet.pth/RNet.pth/ONet.pth）}"
        "{model_path||MobileFaceNet TorchScript 模型路径}"
        "{face_db||人脸库目录，每张图片只包含一张人脸，文件名即为人名}"
        "{image_path||待识别图片路径}"
        "{device|auto|推理设备，可选 auto/cpu/cuda}"
        "{threshold|0.6|人脸识别相似度阈值}"
        "{save_path||可选，保存结果图片路径}"
        "{show|1|是否显示结果窗口，1 表示显示}");

    // 主判断：缺少必要参数时先打印帮助。
    if (parser.has("help") || parser.get<std::string>("mtcnn_dir").empty() ||
        parser.get<std::string>("model_path").empty() ||
        parser.get<std::string>("face_db").empty() ||
        parser.get<std::string>("image_path").empty()) {
        PrintUsage();
        parser.printMessage();
        return 0;
    }

    try {
        const std::string mtcnn_dir = parser.get<std::string>("mtcnn_dir");
        const std::string model_path = parser.get<std::string>("model_path");
        const std::string face_db = parser.get<std::string>("face_db");
        const std::string image_path = parser.get<std::string>("image_path");
        const std::string save_path = parser.get<std::string>("save_path");
        const float threshold = parser.get<float>("threshold");
        const bool show = parser.get<int>("show") != 0;

        cv::Mat image = cv::imread(image_path);
        // 主判断：图片读取失败时直接退出。
        if (image.empty()) {
            std::cerr << "读取图片失败: " << image_path << std::endl;
            return 1;
        }

        FaceRecognizer::Options options;
        options.device = ResolveDevice(parser.get<std::string>("device"));
        options.threshold = threshold;

        FaceRecognizer recognizer(mtcnn_dir, model_path, face_db, options);

        const std::vector<FaceRecognizer::FaceRecord> results = recognizer.Recognize(image);

        cv::Mat visualized = image.clone();
        FaceRecognizer::DrawResults(visualized, results);

        std::cout << "识别到人脸数量: " << results.size() << std::endl;
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& record = results[i];
            std::cout << "face[" << i << "] name=" << record.name
                      << ", similarity=" << record.similarity
                      << ", bbox=("
                      << record.bbox.x << ", "
                      << record.bbox.y << ", "
                      << record.bbox.x + record.bbox.width << ", "
                      << record.bbox.y + record.bbox.height << ")"
                      << std::endl;
        }

        // 主判断：只有传入保存路径时才写文件。
        if (!save_path.empty()) {
            cv::imwrite(save_path, visualized);
            std::cout << "结果已保存到: " << save_path << std::endl;
        }

        // 主判断：命令行显式关闭显示时不弹窗。
        if (show) {
            cv::imshow("face_image_demo", visualized);
            cv::waitKey(0);
        }
        return 0;
    } catch (const c10::Error& error) {
        std::cerr << "Torch 推理失败: " << error.what() << std::endl;
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "运行失败: " << error.what() << std::endl;
        return 1;
    }
}
