#include "face_recognizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

// MobileFaceNet 模型的像素归一化参数，与 Python 版 process 函数保持一致
constexpr float kPixelMean = 127.5f;
constexpr float kPixelStd = 127.5f;

// 标准人脸5个关键点坐标，与 Python 版 estimate_norm 中的 src 数组一致
constexpr float kSrcLandmarks[5][2] = {
    {38.2946f, 51.6963f},
    {73.5318f, 51.5014f},
    {56.0252f, 71.7366f},
    {41.5493f, 92.3655f},
    {70.7299f, 92.2041f},
};

std::vector<cv::Point2f> GetCanonicalLandmarks() {
    return {
        cv::Point2f(kSrcLandmarks[0][0], kSrcLandmarks[0][1]),
        cv::Point2f(kSrcLandmarks[1][0], kSrcLandmarks[1][1]),
        cv::Point2f(kSrcLandmarks[2][0], kSrcLandmarks[2][1]),
        cv::Point2f(kSrcLandmarks[3][0], kSrcLandmarks[3][1]),
        cv::Point2f(kSrcLandmarks[4][0], kSrcLandmarks[4][1]),
    };
}

std::wstring Utf8ToWide(const std::string& text) {
#ifdef _WIN32
    // 主判断：空字符串直接返回，避免后续转换申请 0 长度缓冲区。
    if (text.empty()) {
        return {};
    }
    const int wide_length = MultiByteToWideChar(
        CP_UTF8,
        0,
        text.c_str(),
        static_cast<int>(text.size()),
        nullptr,
        0);
    std::wstring wide(static_cast<size_t>(wide_length), L'\0');
    MultiByteToWideChar(
        CP_UTF8,
        0,
        text.c_str(),
        static_cast<int>(text.size()),
        wide.data(),
        wide_length);
    return wide;
#else
    return std::wstring(text.begin(), text.end());
#endif
}

bool ContainsNonAscii(const std::string& text) {
    return std::any_of(text.begin(), text.end(), [](unsigned char c) {
        return c > 0x7F;
    });
}

// 对齐矩阵方向必须与 Python 版 tform.estimate(lmk, src) 保持一致：
// 输入关键点 -> 标准模板关键点。
cv::Mat EstimateSimilarityTransform(const std::array<cv::Point2f, 5>& landmarks) {
    const std::vector<cv::Point2f> target_points = GetCanonicalLandmarks();
    cv::Mat A(10, 4, CV_64F, cv::Scalar(0.0));
    cv::Mat b(10, 1, CV_64F, cv::Scalar(0.0));

    for (int i = 0; i < 5; ++i) {
        const double x = static_cast<double>(landmarks[static_cast<size_t>(i)].x);
        const double y = static_cast<double>(landmarks[static_cast<size_t>(i)].y);
        const double tx = static_cast<double>(target_points[static_cast<size_t>(i)].x);
        const double ty = static_cast<double>(target_points[static_cast<size_t>(i)].y);

        A.at<double>(2 * i, 0) = x;
        A.at<double>(2 * i, 1) = -y;
        A.at<double>(2 * i, 2) = 1.0;
        A.at<double>(2 * i, 3) = 0.0;
        b.at<double>(2 * i, 0) = tx;

        A.at<double>(2 * i + 1, 0) = y;
        A.at<double>(2 * i + 1, 1) = x;
        A.at<double>(2 * i + 1, 2) = 0.0;
        A.at<double>(2 * i + 1, 3) = 1.0;
        b.at<double>(2 * i + 1, 0) = ty;
    }

    cv::Mat solution;
    // 主判断：求解失败时回退为单位变换，避免 warpAffine 崩溃。
    if (!cv::solve(A, b, solution, cv::DECOMP_SVD)) {
        return cv::Mat::eye(2, 3, CV_64F);
    }

    cv::Mat transform = cv::Mat::eye(2, 3, CV_64F);
    transform.at<double>(0, 0) = solution.at<double>(0, 0);
    transform.at<double>(0, 1) = -solution.at<double>(1, 0);
    transform.at<double>(0, 2) = solution.at<double>(2, 0);
    transform.at<double>(1, 0) = solution.at<double>(1, 0);
    transform.at<double>(1, 1) = solution.at<double>(0, 0);
    transform.at<double>(1, 2) = solution.at<double>(3, 0);
    return transform;
}

std::string PathToUtf8(const std::filesystem::path& path) {
#ifdef _WIN32
    const std::wstring wide = path.wstring();
    // 主判断：空字符串直接返回，避免后续转换申请 0 长度缓冲区。
    if (wide.empty()) {
        return {};
    }
    const int utf8_length = WideCharToMultiByte(
        CP_UTF8,
        0,
        wide.c_str(),
        static_cast<int>(wide.size()),
        nullptr,
        0,
        nullptr,
        nullptr);
    std::string utf8(static_cast<size_t>(utf8_length), '\0');
    WideCharToMultiByte(
        CP_UTF8,
        0,
        wide.c_str(),
        static_cast<int>(wide.size()),
        utf8.data(),
        utf8_length,
        nullptr,
        nullptr);
    return utf8;
#else
    return path.string();
#endif
}

cv::Mat ReadImageWithUnicodePath(const std::filesystem::path& image_path) {
    std::ifstream input(image_path, std::ios::binary);
    // 主判断：文件打开失败时直接返回空图。
    if (!input) {
        return {};
    }

    std::vector<unsigned char> buffer(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    // 主判断：空文件时直接返回空图。
    if (buffer.empty()) {
        return {};
    }

    return cv::imdecode(buffer, cv::IMREAD_COLOR);
}

#ifdef _WIN32
cv::Size MeasureUtf8Text(const std::string& text, int font_height) {
    const std::wstring wide_text = Utf8ToWide(text);
    // 主判断：空文本时直接返回 0 尺寸。
    if (wide_text.empty()) {
        return cv::Size();
    }

    HDC screen_dc = GetDC(nullptr);
    HDC memory_dc = CreateCompatibleDC(screen_dc);
    HFONT font = CreateFontW(
        -font_height,
        0,
        0,
        0,
        FW_NORMAL,
        FALSE,
        FALSE,
        FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE,
        L"Microsoft YaHei UI");
    HGDIOBJ old_font = SelectObject(memory_dc, font);

    SIZE size{};
    GetTextExtentPoint32W(memory_dc, wide_text.c_str(), static_cast<int>(wide_text.size()), &size);

    SelectObject(memory_dc, old_font);
    DeleteObject(font);
    DeleteDC(memory_dc);
    ReleaseDC(nullptr, screen_dc);
    return cv::Size(size.cx, size.cy);
}

void DrawUtf8Text(
    cv::Mat& image,
    const std::string& text,
    const cv::Point& top_left,
    const cv::Scalar& color,
    int font_height) {
    const std::wstring wide_text = Utf8ToWide(text);
    // 主判断：空文本时无需绘制。
    if (wide_text.empty()) {
        return;
    }

    cv::Mat bgra_image;
    cv::cvtColor(image, bgra_image, cv::COLOR_BGR2BGRA);

    BITMAPINFO bitmap_info{};
    bitmap_info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmap_info.bmiHeader.biWidth = bgra_image.cols;
    bitmap_info.bmiHeader.biHeight = -bgra_image.rows;
    bitmap_info.bmiHeader.biPlanes = 1;
    bitmap_info.bmiHeader.biBitCount = 32;
    bitmap_info.bmiHeader.biCompression = BI_RGB;

    void* dib_bits = nullptr;
    HDC screen_dc = GetDC(nullptr);
    HDC memory_dc = CreateCompatibleDC(screen_dc);
    HBITMAP dib_bitmap =
        CreateDIBSection(memory_dc, &bitmap_info, DIB_RGB_COLORS, &dib_bits, nullptr, 0);
    HGDIOBJ old_bitmap = SelectObject(memory_dc, dib_bitmap);

    std::memcpy(dib_bits, bgra_image.data, bgra_image.total() * bgra_image.elemSize());

    HFONT font = CreateFontW(
        -font_height,
        0,
        0,
        0,
        FW_NORMAL,
        FALSE,
        FALSE,
        FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE,
        L"Microsoft YaHei UI");
    HGDIOBJ old_font = SelectObject(memory_dc, font);

    SetBkMode(memory_dc, TRANSPARENT);
    SetTextColor(memory_dc, RGB(
        static_cast<int>(color[2]),
        static_cast<int>(color[1]),
        static_cast<int>(color[0])));
    TextOutW(
        memory_dc,
        top_left.x,
        top_left.y,
        wide_text.c_str(),
        static_cast<int>(wide_text.size()));
    GdiFlush();

    std::memcpy(bgra_image.data, dib_bits, bgra_image.total() * bgra_image.elemSize());
    cv::cvtColor(bgra_image, image, cv::COLOR_BGRA2BGR);

    SelectObject(memory_dc, old_font);
    SelectObject(memory_dc, old_bitmap);
    DeleteObject(font);
    DeleteObject(dib_bitmap);
    DeleteDC(memory_dc);
    ReleaseDC(nullptr, screen_dc);
}
#endif

}  // namespace

FaceRecognizer::FaceRecognizer(
    const std::string& mtcnn_model_dir,
    const std::string& mobilefacenet_model_path,
    const std::string& face_db_path,
    Options options)
    : detector_(mtcnn_model_dir, MtcnnDetector::Options{
                                     20.0f,
                                     0.79f,
                                     0.9f,
                                     0.6f,
                                     0.7f,
                                     options.device}),
      options_(std::move(options)),
      face_db_path_(face_db_path) {
    // 加载 MobileFaceNet TorchScript 模型
    model_ = torch::jit::load(mobilefacenet_model_path, options_.device);
    model_.eval();

    // 加载人脸库
    LoadFaceDb();
}

std::vector<FaceRecognizer::FaceRecord> FaceRecognizer::Recognize(const cv::Mat& image) const {
    std::vector<FaceRecord> results;

    // 主判断：空图像直接返回空结果。
    if (image.empty()) {
        return results;
    }

    // 步骤1：使用 MTCNN 检测人脸并获取关键点
    const std::vector<MtcnnDetector::FaceInfo> faces = detector_.Detect(image);

    // 主判断：没有检测到人脸时直接返回。
    if (faces.empty()) {
        return results;
    }

    // 步骤2：对每张人脸进行对齐和裁剪
    std::vector<cv::Mat> aligned_faces;
    aligned_faces.reserve(faces.size());
    for (const MtcnnDetector::FaceInfo& face : faces) {
        // 主判断：没有关键点时无法对齐，跳过该人脸。
        if (!face.has_landmarks) {
            continue;
        }
        cv::Mat aligned = NormCrop(image, face.landmarks, options_.image_size);
        aligned_faces.push_back(aligned);
    }

    // 主判断：没有可对齐的人脸时返回空结果。
    if (aligned_faces.empty()) {
        return results;
    }

    // 步骤3：批量提取人脸特征向量
    const std::vector<std::vector<float>> features = ExtractFeatures(aligned_faces);

    // 主判断：人脸库为空时，所有人脸标记为未知。
    if (face_db_.empty()) {
        for (size_t i = 0; i < faces.size(); ++i) {
            if (!faces[i].has_landmarks) {
                continue;
            }
            FaceRecord record;
            record.name = "unknown";
            record.bbox = faces[i].bbox;
            record.score = faces[i].score;
            record.similarity = 0.0f;
            results.push_back(record);
        }
        return results;
    }

    // 步骤4：将每张人脸特征与人脸库中所有特征计算余弦相似度
    size_t face_idx = 0;
    for (const MtcnnDetector::FaceInfo& face : faces) {
        // 主判断：跳过没有关键点的人脸。
        if (!face.has_landmarks) {
            continue;
        }

        const std::vector<float>& query_feature = features[face_idx];
        ++face_idx;

        // 遍历人脸库，找最大相似度
        float max_sim = -1.0f;
        std::string best_name = "unknown";

        for (const FaceDbEntry& entry : face_db_) {
            const float sim = CosineSimilarity(query_feature, entry.feature);
            // 主判断：找到更高相似度时更新最佳匹配。
            if (sim > max_sim) {
                max_sim = sim;
                best_name = entry.name;
            }
        }

        FaceRecord record;
        // 主判断：相似度低于阈值时标记为未知。
        if (max_sim > options_.threshold) {
            record.name = best_name;
        } else {
            record.name = "unknown";
        }
        record.bbox = face.bbox;
        record.score = face.score;
        record.similarity = max_sim;
        results.push_back(record);
    }

    return results;
}

void FaceRecognizer::DrawResults(cv::Mat& image, const std::vector<FaceRecord>& results) {
    for (const FaceRecord& record : results) {
        const cv::Point top_left(
            static_cast<int>(std::round(record.bbox.x)),
            static_cast<int>(std::round(record.bbox.y)));
        const cv::Point bottom_right(
            static_cast<int>(std::round(record.bbox.x + record.bbox.width)),
            static_cast<int>(std::round(record.bbox.y + record.bbox.height)));

        // 主判断：已知人脸用绿色框，未知人脸用红色框。
        const cv::Scalar box_color = (record.name == "unknown")
            ? cv::Scalar(0, 0, 255)
            : cv::Scalar(0, 255, 0);
        cv::rectangle(image, top_left, bottom_right, box_color, 2);

        std::string label = record.name + " " + cv::format("%.2f", record.similarity);
        if (ContainsNonAscii(label)) {
#ifdef _WIN32
            const int font_height = 18;
            const cv::Size text_size = MeasureUtf8Text(label, font_height);
            const int text_top = std::max(0, top_left.y - text_size.height - 6);
            cv::rectangle(
                image,
                cv::Point(top_left.x, text_top),
                cv::Point(top_left.x + text_size.width + 6, text_top + text_size.height + 6),
                box_color,
                cv::FILLED);
            DrawUtf8Text(
                image,
                label,
                cv::Point(top_left.x + 3, text_top + 3),
                cv::Scalar(255, 255, 255),
                font_height);
#else
            cv::putText(
                image,
                label,
                cv::Point(top_left.x, std::max(0, top_left.y - 5)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255),
                1);
#endif
        } else {
            int baseline = 0;
            cv::Size text_size =
                cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            const cv::Point text_origin(top_left.x, std::max(top_left.y - 5, text_size.height + 2));
            cv::rectangle(
                image,
                cv::Point(text_origin.x, text_origin.y - text_size.height - 2),
                cv::Point(text_origin.x + text_size.width, text_origin.y + 2),
                box_color,
                cv::FILLED);
            cv::putText(
                image,
                label,
                text_origin,
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255),
                1);
        }
    }
}

void FaceRecognizer::UpdateFaceDb() {
    LoadFaceDb();
}

cv::Mat FaceRecognizer::EstimateNorm(const std::array<cv::Point2f, 5>& landmarks) {
    return EstimateSimilarityTransform(landmarks);
}

cv::Mat FaceRecognizer::NormCrop(
    const cv::Mat& image,
    const std::array<cv::Point2f, 5>& landmarks,
    int image_size) {
    cv::Mat M = EstimateNorm(landmarks);
    cv::Mat warped;
    cv::warpAffine(image, warped, M, cv::Size(image_size, image_size), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0.0));
    return warped;
}

torch::Tensor FaceRecognizer::PreprocessFace(const cv::Mat& face) const {
    cv::Mat resized;
    // 主判断：尺寸不一致时需要 resize 到模型输入尺寸。
    if (face.cols != options_.image_size || face.rows != options_.image_size) {
        cv::resize(face, resized, cv::Size(options_.image_size, options_.image_size), 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        resized = face;
    }

    // BGR -> float32，归一化方式与 Python 版一致：(img - 127.5) / 127.5
    cv::Mat float_image;
    resized.convertTo(float_image, CV_32FC3);
    float_image = (float_image - cv::Scalar(kPixelMean, kPixelMean, kPixelMean)) / kPixelStd;

    // HWC -> CHW
    return torch::from_blob(
               float_image.data,
               {float_image.rows, float_image.cols, 3},
               torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .permute({2, 0, 1});
}

std::vector<std::vector<float>> FaceRecognizer::ExtractFeatures(
    const std::vector<cv::Mat>& faces) const {
    // 主判断：没有人脸时返回空结果。
    if (faces.empty()) {
        return {};
    }

    torch::NoGradGuard no_grad;

    // 批量预处理所有人脸
    std::vector<torch::Tensor> tensors;
    tensors.reserve(faces.size());
    for (const cv::Mat& face : faces) {
        tensors.push_back(PreprocessFace(face));
    }

    // 组成 batch 并送入模型
    const torch::Tensor batch = torch::stack(tensors).to(options_.device);
    torch::IValue output = model_.forward(std::vector<torch::jit::IValue>{batch});
    const torch::Tensor features = output.toTensor().to(torch::kCPU).contiguous();

    // 将每张人脸的特征向量转为 std::vector<float>
    const int64_t num_faces = features.size(0);
    const int64_t feature_dim = features.size(1);
    const auto accessor = features.accessor<float, 2>();

    std::vector<std::vector<float>> result;
    result.reserve(static_cast<size_t>(num_faces));
    for (int64_t i = 0; i < num_faces; ++i) {
        std::vector<float> feature(feature_dim);
        for (int64_t j = 0; j < feature_dim; ++j) {
            feature[static_cast<size_t>(j)] = accessor[i][j];
        }
        result.push_back(std::move(feature));
    }

    return result;
}

float FaceRecognizer::CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    // 主判断：维度不一致时返回0，避免越界访问。
    if (a.size() != b.size() || a.empty()) {
        return 0.0f;
    }

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    // 主判断：分母接近0时返回0，避免除零。
    const float denominator = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denominator < 1e-8f) {
        return 0.0f;
    }

    return dot / denominator;
}

void FaceRecognizer::LoadFaceDb() {
    face_db_.clear();

    // 主判断：人脸库目录不存在时跳过加载。
    if (!std::filesystem::exists(face_db_path_)) {
        return;
    }

    // 遍历人脸库目录下的图片文件，提取特征
    for (const auto& entry : std::filesystem::directory_iterator(face_db_path_)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        // 主判断：只处理常见图片格式。
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp") {
            continue;
        }

        const std::string name = PathToUtf8(entry.path().stem());
        const std::filesystem::path image_path = entry.path();

        cv::Mat img = ReadImageWithUnicodePath(image_path);
        // 主判断：图片读取失败时跳过。
        if (img.empty()) {
            continue;
        }

        // 检测人脸
        const std::vector<MtcnnDetector::FaceInfo> detected_faces = detector_.Detect(img);

        // 主判断：不是恰好1张人脸时跳过，与 Python 版保持一致。
        if (detected_faces.size() != 1) {
            continue;
        }

        const MtcnnDetector::FaceInfo& face = detected_faces[0];

        // 主判断：没有关键点时无法对齐，跳过。
        if (!face.has_landmarks) {
            continue;
        }

        // 对齐人脸并提取特征
        cv::Mat aligned = NormCrop(img, face.landmarks, options_.image_size);
        const std::vector<std::vector<float>> features = ExtractFeatures({aligned});

        // 主判断：特征提取失败时跳过。
        if (features.empty()) {
            continue;
        }

        FaceDbEntry db_entry;
        db_entry.name = name;
        db_entry.feature = features[0];
        face_db_.push_back(std::move(db_entry));
    }
}
