#include "mtcnn_detector.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <stdexcept>

namespace {

constexpr float kPixelMean = 127.5f;
constexpr float kPixelStd = 128.0f;
constexpr float kPNetSize = 12.0f;
constexpr int kRNetSize = 24;
constexpr int kONetSize = 48;

}  // namespace

MtcnnDetector::MtcnnDetector(const std::string& model_dir, Options options)
    : options_(std::move(options)) {
    const std::filesystem::path base_dir(model_dir);
    pnet_ = torch::jit::load((base_dir / "PNet.pth").string(), options_.device);
    rnet_ = torch::jit::load((base_dir / "RNet.pth").string(), options_.device);
    onet_ = torch::jit::load((base_dir / "ONet.pth").string(), options_.device);
    pnet_.eval();
    rnet_.eval();
    onet_.eval();
}

std::vector<MtcnnDetector::FaceInfo> MtcnnDetector::Detect(const cv::Mat& image) const {
    // 主判断：空图像时直接返回，避免后续访问尺寸时报错。
    if (image.empty()) {
        return {};
    }

    torch::NoGradGuard no_grad;

    const std::vector<CandidateBox> pnet_boxes = DetectPNet(image);
    // 主判断：PNet 没有候选框时无需继续级联。
    if (pnet_boxes.empty()) {
        return {};
    }

    const std::vector<CandidateBox> rnet_boxes = DetectRNet(image, pnet_boxes);
    // 主判断：RNet 没有保留结果时直接结束。
    if (rnet_boxes.empty()) {
        return {};
    }

    return DetectONet(image, rnet_boxes);
}

void MtcnnDetector::DrawDetections(cv::Mat& image, const std::vector<FaceInfo>& faces) {
    for (const FaceInfo& face : faces) {
        const cv::Point top_left(
            static_cast<int>(std::round(face.bbox.x)),
            static_cast<int>(std::round(face.bbox.y)));
        const cv::Point bottom_right(
            static_cast<int>(std::round(face.bbox.x + face.bbox.width)),
            static_cast<int>(std::round(face.bbox.y + face.bbox.height)));
        cv::rectangle(image, top_left, bottom_right, cv::Scalar(255, 0, 0), 1);
        cv::putText(
            image,
            cv::format("%.2f", face.score),
            cv::Point(top_left.x, std::max(0, top_left.y - 2)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 255),
            2);

        // 主判断：只有 ONet 阶段产生的结果才绘制关键点。
        if (!face.has_landmarks) {
            continue;
        }
        for (const cv::Point2f& point : face.landmarks) {
            cv::circle(
                image,
                cv::Point(static_cast<int>(std::round(point.x)), static_cast<int>(std::round(point.y))),
                2,
                cv::Scalar(0, 0, 255),
                -1);
        }
    }
}

std::vector<MtcnnDetector::CandidateBox> MtcnnDetector::DetectPNet(const cv::Mat& image) const {
    float current_scale = kPNetSize / options_.min_face_size;
    std::vector<CandidateBox> all_boxes;

    // 主循环：按图像金字塔逐层缩放，与 Python 版 detect_pnet 保持一致。
    while (std::min(image.rows * current_scale, image.cols * current_scale) > kPNetSize) {
        torch::Tensor input = ProcessImage(image, current_scale).unsqueeze(0).to(options_.device);
        torch::IValue output = pnet_.forward(std::vector<torch::jit::IValue>{input});
        auto output_tuple = output.toTuple();
        const auto& outputs = output_tuple->elements();
        torch::Tensor cls_tensor = outputs[0].toTensor().squeeze().to(torch::kCPU).contiguous();
        torch::Tensor reg_tensor = outputs[1].toTensor().squeeze().to(torch::kCPU).contiguous();

        std::vector<CandidateBox> scale_boxes =
            GenerateBBox(cls_tensor, reg_tensor, current_scale, options_.pnet_threshold);

        // 主判断：当前尺度无候选框时仅继续下一层金字塔。
        if (!scale_boxes.empty()) {
            const std::vector<int> keep = NmsIndices(scale_boxes, 0.5f, NmsMode::Union);
            for (int index : keep) {
                all_boxes.push_back(scale_boxes[static_cast<size_t>(index)]);
            }
        }

        current_scale *= options_.scale_factor;
    }

    // 主判断：所有尺度都没有检测到候选框时直接返回空结果。
    if (all_boxes.empty()) {
        return {};
    }

    const std::vector<int> keep = NmsIndices(all_boxes, 0.7f, NmsMode::Union);
    std::vector<CandidateBox> calibrated_boxes;
    calibrated_boxes.reserve(keep.size());
    for (int index : keep) {
        calibrated_boxes.push_back(CalibratePNetBox(all_boxes[static_cast<size_t>(index)]));
    }
    return calibrated_boxes;
}

std::vector<MtcnnDetector::CandidateBox> MtcnnDetector::DetectRNet(
    const cv::Mat& image,
    const std::vector<CandidateBox>& boxes) const {
    const std::vector<CandidateBox> square_boxes = RoundCoordinates(ConvertToSquare(boxes));
    std::vector<CandidateBox> valid_boxes;
    std::vector<torch::Tensor> inputs;
    valid_boxes.reserve(square_boxes.size());
    inputs.reserve(square_boxes.size());

    for (const CandidateBox& box : square_boxes) {
        const int width = static_cast<int>(std::round(box.x2 - box.x1 + 1.0f));
        const int height = static_cast<int>(std::round(box.y2 - box.y1 + 1.0f));

        // 主判断：与 Python 版一致，过小候选框直接丢弃。
        if (std::min(width, height) < 20) {
            continue;
        }

        cv::Mat cropped = CropAndPad(
            image,
            static_cast<int>(box.x1),
            static_cast<int>(box.y1),
            static_cast<int>(box.x2),
            static_cast<int>(box.y2));
        cv::Mat resized;
        cv::resize(cropped, resized, cv::Size(kRNetSize, kRNetSize), 0.0, 0.0, cv::INTER_LINEAR);
        inputs.push_back(MatToTensor(resized));
        valid_boxes.push_back(box);
    }

    // 主判断：没有合法候选框时，RNet 阶段直接结束。
    if (inputs.empty()) {
        return {};
    }

    const torch::Tensor batch = torch::stack(inputs).to(options_.device);
    torch::IValue output = rnet_.forward(std::vector<torch::jit::IValue>{batch});
    auto output_tuple = output.toTuple();
    const auto& outputs = output_tuple->elements();
    const torch::Tensor cls_tensor =
        torch::softmax(outputs[0].toTensor(), -1).to(torch::kCPU).contiguous();
    const torch::Tensor reg_tensor = outputs[1].toTensor().to(torch::kCPU).contiguous();

    const auto cls_accessor = cls_tensor.accessor<float, 2>();
    const auto reg_accessor = reg_tensor.accessor<float, 2>();
    std::vector<CandidateBox> candidate_boxes;
    candidate_boxes.reserve(valid_boxes.size());

    for (int64_t i = 0; i < cls_tensor.size(0); ++i) {
        const float score = cls_accessor[i][1];
        // 主判断：只保留置信度高于阈值的候选框。
        if (score <= options_.rnet_threshold) {
            continue;
        }

        CandidateBox candidate = valid_boxes[static_cast<size_t>(i)];
        candidate.score = score;
        candidate.reg = {
            reg_accessor[i][0],
            reg_accessor[i][1],
            reg_accessor[i][2],
            reg_accessor[i][3],
        };
        candidate_boxes.push_back(candidate);
    }

    // 主判断：阈值筛选后为空时不再继续做 NMS。
    if (candidate_boxes.empty()) {
        return {};
    }

    const std::vector<int> keep = NmsIndices(candidate_boxes, 0.4f, NmsMode::Union);
    std::vector<CandidateBox> calibrated_boxes;
    calibrated_boxes.reserve(keep.size());
    for (int index : keep) {
        const CandidateBox& box = candidate_boxes[static_cast<size_t>(index)];
        calibrated_boxes.push_back(CalibrateBox(box, box.reg));
    }
    return calibrated_boxes;
}

std::vector<MtcnnDetector::FaceInfo> MtcnnDetector::DetectONet(
    const cv::Mat& image,
    const std::vector<CandidateBox>& boxes) const {
    const std::vector<CandidateBox> square_boxes = RoundCoordinates(ConvertToSquare(boxes));
    std::vector<CandidateBox> valid_boxes;
    std::vector<torch::Tensor> inputs;
    valid_boxes.reserve(square_boxes.size());
    inputs.reserve(square_boxes.size());

    for (const CandidateBox& box : square_boxes) {
        const int width = static_cast<int>(std::round(box.x2 - box.x1 + 1.0f));
        const int height = static_cast<int>(std::round(box.y2 - box.y1 + 1.0f));

        // 主判断：非法框直接过滤，避免 resize 时出现异常。
        if (width <= 0 || height <= 0) {
            continue;
        }

        cv::Mat cropped = CropAndPad(
            image,
            static_cast<int>(box.x1),
            static_cast<int>(box.y1),
            static_cast<int>(box.x2),
            static_cast<int>(box.y2));
        cv::Mat resized;
        cv::resize(cropped, resized, cv::Size(kONetSize, kONetSize), 0.0, 0.0, cv::INTER_LINEAR);
        inputs.push_back(MatToTensor(resized));
        valid_boxes.push_back(box);
    }

    // 主判断：没有合法输入时直接返回空结果。
    if (inputs.empty()) {
        return {};
    }

    const torch::Tensor batch = torch::stack(inputs).to(options_.device);
    torch::IValue output = onet_.forward(std::vector<torch::jit::IValue>{batch});
    auto output_tuple = output.toTuple();
    const auto& outputs = output_tuple->elements();
    const torch::Tensor cls_tensor =
        torch::softmax(outputs[0].toTensor(), -1).to(torch::kCPU).contiguous();
    const torch::Tensor reg_tensor = outputs[1].toTensor().to(torch::kCPU).contiguous();
    const torch::Tensor landmark_tensor = outputs[2].toTensor().to(torch::kCPU).contiguous();

    const auto cls_accessor = cls_tensor.accessor<float, 2>();
    const auto reg_accessor = reg_tensor.accessor<float, 2>();
    const auto landmark_accessor = landmark_tensor.accessor<float, 2>();
    std::vector<ONetResult> candidate_results;
    candidate_results.reserve(valid_boxes.size());

    for (int64_t i = 0; i < cls_tensor.size(0); ++i) {
        const float score = cls_accessor[i][1];
        // 主判断：只保留高置信度的人脸候选框。
        if (score <= options_.onet_threshold) {
            continue;
        }

        CandidateBox original_box = valid_boxes[static_cast<size_t>(i)];
        original_box.score = score;
        const std::array<float, 4> reg = {
            reg_accessor[i][0],
            reg_accessor[i][1],
            reg_accessor[i][2],
            reg_accessor[i][3],
        };

        const float width = original_box.x2 - original_box.x1 + 1.0f;
        const float height = original_box.y2 - original_box.y1 + 1.0f;
        std::array<cv::Point2f, 5> landmarks{};
        for (int point_index = 0; point_index < 5; ++point_index) {
            landmarks[static_cast<size_t>(point_index)] = cv::Point2f(
                landmark_accessor[i][point_index * 2] * width + original_box.x1 - 1.0f,
                landmark_accessor[i][point_index * 2 + 1] * height + original_box.y1 - 1.0f);
        }

        ONetResult result;
        result.box = CalibrateBox(original_box, reg);
        result.box.score = score;
        result.landmarks = landmarks;
        candidate_results.push_back(result);
    }

    // 主判断：ONet 阈值筛选后为空时返回空结果。
    if (candidate_results.empty()) {
        return {};
    }

    std::vector<CandidateBox> nms_boxes;
    nms_boxes.reserve(candidate_results.size());
    for (const ONetResult& result : candidate_results) {
        nms_boxes.push_back(result.box);
    }

    const std::vector<int> keep = NmsIndices(nms_boxes, 0.6f, NmsMode::Minimum);
    std::vector<FaceInfo> faces;
    faces.reserve(keep.size());
    for (int index : keep) {
        const ONetResult& result = candidate_results[static_cast<size_t>(index)];
        FaceInfo face;
        face.bbox = cv::Rect2f(
            result.box.x1,
            result.box.y1,
            result.box.x2 - result.box.x1,
            result.box.y2 - result.box.y1);
        face.score = result.box.score;
        face.landmarks = result.landmarks;
        face.has_landmarks = true;
        faces.push_back(face);
    }
    return faces;
}

torch::Tensor MtcnnDetector::ProcessImage(const cv::Mat& image, float scale) const {
    const int new_height = std::max(1, static_cast<int>(image.rows * scale));
    const int new_width = std::max(1, static_cast<int>(image.cols * scale));
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height), 0.0, 0.0, cv::INTER_LINEAR);
    return MatToTensor(resized);
}

torch::Tensor MtcnnDetector::MatToTensor(const cv::Mat& image) const {
    cv::Mat bgr_image;
    if (image.channels() == 3) {
        bgr_image = image;
    } else if (image.channels() == 4) {
        cv::cvtColor(image, bgr_image, cv::COLOR_BGRA2BGR);
    } else {
        cv::cvtColor(image, bgr_image, cv::COLOR_GRAY2BGR);
    }

    cv::Mat float_image;
    bgr_image.convertTo(float_image, CV_32FC3);
    float_image = (float_image - cv::Scalar(kPixelMean, kPixelMean, kPixelMean)) / kPixelStd;
    return torch::from_blob(
               float_image.data,
               {float_image.rows, float_image.cols, 3},
               torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .permute({2, 0, 1});
}

cv::Mat MtcnnDetector::CropAndPad(const cv::Mat& image, int x1, int y1, int x2, int y2) const {
    const int width = std::max(1, x2 - x1 + 1);
    const int height = std::max(1, y2 - y1 + 1);
    cv::Mat cropped = cv::Mat::zeros(height, width, CV_8UC3);

    const int src_x1 = std::max(0, x1);
    const int src_y1 = std::max(0, y1);
    const int src_x2 = std::min(image.cols - 1, x2);
    const int src_y2 = std::min(image.rows - 1, y2);

    // 主判断：如果和原图没有交集，则直接返回全黑图块。
    if (src_x1 > src_x2 || src_y1 > src_y2) {
        return cropped;
    }

    const cv::Rect src_roi(src_x1, src_y1, src_x2 - src_x1 + 1, src_y2 - src_y1 + 1);
    const cv::Rect dst_roi(std::max(0, -x1), std::max(0, -y1), src_roi.width, src_roi.height);
    image(src_roi).copyTo(cropped(dst_roi));
    return cropped;
}

float MtcnnDetector::SoftmaxPositive(float negative_logit, float positive_logit) {
    const float max_logit = std::max(negative_logit, positive_logit);
    const float negative_exp = std::exp(negative_logit - max_logit);
    const float positive_exp = std::exp(positive_logit - max_logit);
    return positive_exp / (negative_exp + positive_exp);
}

int MtcnnDetector::GetPNetMapHeight(const torch::Tensor& tensor) {
    if (tensor.dim() >= 3) {
        return static_cast<int>(tensor.size(1));
    }
    if (tensor.dim() == 2 || tensor.dim() == 1) {
        return 1;
    }
    return 0;
}

int MtcnnDetector::GetPNetMapWidth(const torch::Tensor& tensor) {
    if (tensor.dim() >= 3) {
        return static_cast<int>(tensor.size(2));
    }
    if (tensor.dim() == 2) {
        return static_cast<int>(tensor.size(1));
    }
    if (tensor.dim() == 1) {
        return 1;
    }
    return 0;
}

float MtcnnDetector::GetPNetValue(const torch::Tensor& tensor, int channel, int y, int x) {
    const float* data = tensor.data_ptr<float>();
    if (tensor.dim() >= 3) {
        const int height = static_cast<int>(tensor.size(1));
        const int width = static_cast<int>(tensor.size(2));
        return data[channel * height * width + y * width + x];
    }
    if (tensor.dim() == 2) {
        const int width = static_cast<int>(tensor.size(1));
        return data[channel * width + x];
    }
    if (tensor.dim() == 1) {
        return data[channel];
    }
    throw std::runtime_error("PNet 输出维度不合法");
}

std::vector<MtcnnDetector::CandidateBox> MtcnnDetector::GenerateBBox(
    const torch::Tensor& cls_tensor,
    const torch::Tensor& reg_tensor,
    float scale,
    float threshold) {
    const int height = GetPNetMapHeight(cls_tensor);
    const int width = GetPNetMapWidth(cls_tensor);
    std::vector<CandidateBox> boxes;

    // 主判断：输出形状异常时直接返回空结果，避免数组越界。
    if (height <= 0 || width <= 0) {
        return boxes;
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float negative_logit = GetPNetValue(cls_tensor, 0, y, x);
            const float positive_logit = GetPNetValue(cls_tensor, 1, y, x);
            const float score = SoftmaxPositive(negative_logit, positive_logit);

            // 主判断：只对高于阈值的位置生成候选框。
            if (score <= threshold) {
                continue;
            }

            CandidateBox box;
            box.x1 = std::round((2.0f * x) / scale);
            box.y1 = std::round((2.0f * y) / scale);
            box.x2 = std::round((2.0f * x + kPNetSize) / scale);
            box.y2 = std::round((2.0f * y + kPNetSize) / scale);
            box.score = score;
            box.reg = {
                GetPNetValue(reg_tensor, 0, y, x),
                GetPNetValue(reg_tensor, 1, y, x),
                GetPNetValue(reg_tensor, 2, y, x),
                GetPNetValue(reg_tensor, 3, y, x),
            };
            boxes.push_back(box);
        }
    }
    return boxes;
}

std::vector<int> MtcnnDetector::NmsIndices(
    const std::vector<CandidateBox>& boxes,
    float threshold,
    NmsMode mode) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&boxes](int lhs, int rhs) {
        return boxes[static_cast<size_t>(lhs)].score > boxes[static_cast<size_t>(rhs)].score;
    });

    std::vector<int> keep;
    while (!order.empty()) {
        const int current = order.front();
        keep.push_back(current);

        std::vector<int> next_order;
        for (size_t i = 1; i < order.size(); ++i) {
            const int candidate = order[i];
            // 主判断：只有重叠度不超过阈值的框才会保留到下一轮。
            if (IoU(boxes[static_cast<size_t>(current)], boxes[static_cast<size_t>(candidate)], mode) <=
                threshold) {
                next_order.push_back(candidate);
            }
        }
        order.swap(next_order);
    }
    return keep;
}

float MtcnnDetector::IoU(const CandidateBox& lhs, const CandidateBox& rhs, NmsMode mode) {
    const float xx1 = std::max(lhs.x1, rhs.x1);
    const float yy1 = std::max(lhs.y1, rhs.y1);
    const float xx2 = std::min(lhs.x2, rhs.x2);
    const float yy2 = std::min(lhs.y2, rhs.y2);
    const float width = std::max(0.0f, xx2 - xx1 + 1.0f);
    const float height = std::max(0.0f, yy2 - yy1 + 1.0f);
    const float inter = width * height;
    const float area_lhs = (lhs.x2 - lhs.x1 + 1.0f) * (lhs.y2 - lhs.y1 + 1.0f);
    const float area_rhs = (rhs.x2 - rhs.x1 + 1.0f) * (rhs.y2 - rhs.y1 + 1.0f);

    if (mode == NmsMode::Minimum) {
        return inter / std::min(area_lhs, area_rhs);
    }
    return inter / (area_lhs + area_rhs - inter + 1e-10f);
}

std::vector<MtcnnDetector::CandidateBox> MtcnnDetector::ConvertToSquare(
    const std::vector<CandidateBox>& boxes) {
    std::vector<CandidateBox> square_boxes;
    square_boxes.reserve(boxes.size());
    for (const CandidateBox& box : boxes) {
        const float height = box.y2 - box.y1 + 1.0f;
        const float width = box.x2 - box.x1 + 1.0f;
        const float max_side = std::max(width, height);

        CandidateBox square_box = box;
        square_box.x1 = box.x1 + width * 0.5f - max_side * 0.5f;
        square_box.y1 = box.y1 + height * 0.5f - max_side * 0.5f;
        square_box.x2 = square_box.x1 + max_side - 1.0f;
        square_box.y2 = square_box.y1 + max_side - 1.0f;
        square_boxes.push_back(square_box);
    }
    return square_boxes;
}

std::vector<MtcnnDetector::CandidateBox> MtcnnDetector::RoundCoordinates(
    const std::vector<CandidateBox>& boxes) {
    std::vector<CandidateBox> rounded_boxes = boxes;
    for (CandidateBox& box : rounded_boxes) {
        box.x1 = std::round(box.x1);
        box.y1 = std::round(box.y1);
        box.x2 = std::round(box.x2);
        box.y2 = std::round(box.y2);
    }
    return rounded_boxes;
}

MtcnnDetector::CandidateBox MtcnnDetector::CalibrateBox(
    const CandidateBox& box,
    const std::array<float, 4>& reg) {
    const float width = box.x2 - box.x1 + 1.0f;
    const float height = box.y2 - box.y1 + 1.0f;
    CandidateBox calibrated_box = box;
    calibrated_box.x1 += reg[0] * width;
    calibrated_box.y1 += reg[1] * height;
    calibrated_box.x2 += reg[2] * width;
    calibrated_box.y2 += reg[3] * height;
    return calibrated_box;
}

MtcnnDetector::CandidateBox MtcnnDetector::CalibratePNetBox(const CandidateBox& box) {
    return CalibrateBox(box, box.reg);
}
