#pragma once

#include <array>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include "mtcnn_detector.h"

class FaceRecognizer {
public:
    struct FaceRecord {
        std::string name;
        cv::Rect2f bbox;
        float score = 0.0f;
        float similarity = 0.0f;
    };

    struct Options {
        float threshold = 0.6f;
        int image_size = 112;
        torch::Device device = torch::kCPU;
    };

    FaceRecognizer(
        const std::string& mtcnn_model_dir,
        const std::string& mobilefacenet_model_path,
        const std::string& face_db_path,
        Options options = {});

    std::vector<FaceRecord> Recognize(const cv::Mat& image) const;

    static void DrawResults(cv::Mat& image, const std::vector<FaceRecord>& results);

    void UpdateFaceDb();

private:
    struct FaceDbEntry {
        std::string name;
        std::vector<float> feature;
    };

    // 人脸对齐：根据5个关键点估计相似变换矩阵并对齐人脸
    static cv::Mat EstimateNorm(const std::array<cv::Point2f, 5>& landmarks);
    static cv::Mat NormCrop(const cv::Mat& image, const std::array<cv::Point2f, 5>& landmarks, int image_size = 112);

    // 图像预处理：resize -> CHW -> 归一化
    torch::Tensor PreprocessFace(const cv::Mat& face) const;

    // 特征提取：批量推理获取特征向量
    std::vector<std::vector<float>> ExtractFeatures(const std::vector<cv::Mat>& faces) const;

    // 计算两个特征向量的余弦相似度
    static float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);

    // 加载人脸库中所有人脸图片并提取特征
    void LoadFaceDb();

    MtcnnDetector detector_;
    mutable torch::jit::script::Module model_;
    Options options_;
    std::string face_db_path_;
    std::vector<FaceDbEntry> face_db_;
};
