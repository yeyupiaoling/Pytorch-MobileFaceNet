#pragma once

#include <array>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/script.h>

class MtcnnDetector {
public:
    struct FaceInfo {
        cv::Rect2f bbox;
        float score = 0.0f;
        std::array<cv::Point2f, 5> landmarks{};
        bool has_landmarks = false;
    };

    struct Options {
        float min_face_size = 20.0f;
        float scale_factor = 0.79f;
        float pnet_threshold = 0.9f;
        float rnet_threshold = 0.6f;
        float onet_threshold = 0.7f;
        torch::Device device = torch::kCPU;
    };

    MtcnnDetector(const std::string& model_dir, Options options = {});

    std::vector<FaceInfo> Detect(const cv::Mat& image) const;

    static void DrawDetections(cv::Mat& image, const std::vector<FaceInfo>& faces);

private:
    struct CandidateBox {
        float x1 = 0.0f;
        float y1 = 0.0f;
        float x2 = 0.0f;
        float y2 = 0.0f;
        float score = 0.0f;
        std::array<float, 4> reg{};
    };

    enum class NmsMode {
        Union,
        Minimum,
    };

    struct ONetResult {
        CandidateBox box;
        std::array<cv::Point2f, 5> landmarks{};
    };

    std::vector<CandidateBox> DetectPNet(const cv::Mat& image) const;
    std::vector<CandidateBox> DetectRNet(const cv::Mat& image, const std::vector<CandidateBox>& boxes) const;
    std::vector<FaceInfo> DetectONet(const cv::Mat& image, const std::vector<CandidateBox>& boxes) const;

    torch::Tensor ProcessImage(const cv::Mat& image, float scale) const;
    torch::Tensor MatToTensor(const cv::Mat& image) const;
    cv::Mat CropAndPad(const cv::Mat& image, int x1, int y1, int x2, int y2) const;

    static float SoftmaxPositive(float negative_logit, float positive_logit);
    static int GetPNetMapHeight(const torch::Tensor& tensor);
    static int GetPNetMapWidth(const torch::Tensor& tensor);
    static float GetPNetValue(const torch::Tensor& tensor, int channel, int y, int x);
    static std::vector<CandidateBox> GenerateBBox(
        const torch::Tensor& cls_tensor,
        const torch::Tensor& reg_tensor,
        float scale,
        float threshold);
    static std::vector<int> NmsIndices(
        const std::vector<CandidateBox>& boxes,
        float threshold,
        NmsMode mode);
    static float IoU(const CandidateBox& lhs, const CandidateBox& rhs, NmsMode mode);
    static std::vector<CandidateBox> ConvertToSquare(const std::vector<CandidateBox>& boxes);
    static std::vector<CandidateBox> RoundCoordinates(const std::vector<CandidateBox>& boxes);
    static CandidateBox CalibrateBox(const CandidateBox& box, const std::array<float, 4>& reg);
    static CandidateBox CalibratePNetBox(const CandidateBox& box);

    mutable torch::jit::script::Module pnet_;
    mutable torch::jit::script::Module rnet_;
    mutable torch::jit::script::Module onet_;
    Options options_;
};
