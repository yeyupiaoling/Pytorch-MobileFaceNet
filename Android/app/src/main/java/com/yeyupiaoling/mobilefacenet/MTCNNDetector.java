package com.yeyupiaoling.mobilefacenet;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MTCNNDetector {
    private static final String TAG = "MTCNNDetector";
    // 像素均值，用于图像归一化
    private static final float[] PIXEL_MEAN = new float[]{127.5f, 127.5f, 127.5f};
    // 像素标准差，用于图像归一化
    private static final float PIXEL_STD = 128f;
    // 各网络模型的检测阈值
    private final float pnetThreshold = 0.9f;
    private final float rnetThreshold = 0.6f;
    private final float onetThreshold = 0.7f;
    // 最小人脸尺寸
    private final float minFaceSize = 20f;
    // 图像金字塔缩放因子
    private final float scaleFactor = 0.79f;

    // PNet、RNet、ONet 三个神经网络模型
    private Module pnet;
    private Module rnet;
    private Module onet;

    // 构造函数，初始化模型
    public MTCNNDetector(Context context) {
        try {
            pnet = Module.load(assetFilePath(context, "mtcnn/PNet.pt"));
            rnet = Module.load(assetFilePath(context, "mtcnn/RNet.pt"));
            onet = Module.load(assetFilePath(context, "mtcnn/ONet.pt"));
            new Handler(Looper.getMainLooper()).post(() -> {
                Toast.makeText(context, "MTCNN 模型加载成功", Toast.LENGTH_SHORT).show();
            });
        } catch (Exception e) {
            new Handler(Looper.getMainLooper()).post(() -> {
                Toast.makeText(context, "MTCNN 模型加载失败", Toast.LENGTH_LONG).show();
            });
            Log.e(TAG, "模型加载失败", e);
        }
    }

    // 获取 assets 目录下文件的实际路径
    private String assetFilePath(Context context, String assetPath) throws Exception {
        File file = new File(context.getFilesDir(), assetPath.replace('/', '_'));
        // 主判断：如果内部存储已经存在可用文件，则直接复用。
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream inputStream = context.getAssets().open(assetPath);
             OutputStream outputStream = new FileOutputStream(file)) {
            byte[] buffer = new byte[4 * 1024];
            int readSize;
            // 循环写入，直到整个模型文件复制完成。
            while ((readSize = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, readSize);
            }
            outputStream.flush();
        }

        return file.getAbsolutePath();
    }

    // 主检测函数，输入 Bitmap 图像，输出人脸数据列表
    public List<FaceData> detect(Bitmap bitmap) {
        List<FaceData> faces = new ArrayList<>();

        // 主判断：模型未正确加载时直接返回空结果。
        if (pnet == null || rnet == null || onet == null || bitmap == null) {
            return faces;
        }

        // 确保 Bitmap 格式为 ARGB_8888
        Bitmap rgbaBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);

        try {
            // 第一阶段：使用 PNet 检测
            float[][] pnetBoxes = detectPNet(bitmapToFloatArray(rgbaBitmap));

            // 主判断：没有候选框时无需继续后续阶段。
            if (pnetBoxes.length == 0) {
                return faces;
            }

            // 第二阶段：使用 RNet 精炼检测
            float[][] rnetBoxes = detectRNet(rgbaBitmap, pnetBoxes);

            // 主判断：RNet 没有保留结果时直接结束。
            if (rnetBoxes.length == 0) {
                return faces;
            }

            // 第三阶段：使用 ONet 最终检测，同时检测关键点
            float[][] onetResult = detectONet(rgbaBitmap, rnetBoxes);

            // 主判断：ONet 没有最终结果时直接返回。
            if (onetResult.length == 0) {
                return faces;
            }

            int width = rgbaBitmap.getWidth();
            int height = rgbaBitmap.getHeight();

            // 转换检测结果为 FaceData 格式
            for (float[] box : onetResult) {
                float left = box[0];
                float top = box[1];
                float right = box[2];
                float bottom = box[3];
                float score = box[4];

                // 主判断：确保边界框不越界。
                if (left < 0) {
                    left = 0;
                }
                if (top < 0) {
                    top = 0;
                }
                if (right > width) {
                    right = width;
                }
                if (bottom > height) {
                    bottom = height;
                }

                float[] landmarks = null;
                // 主判断：只有关键点完整时才拷贝关键点。
                if (box.length > 14) {
                    landmarks = new float[10];
                    System.arraycopy(box, 5, landmarks, 0, 10);
                }

                faces.add(new FaceData(left, top, right, bottom, score, landmarks));
            }
            return faces;
        } finally {
            rgbaBitmap.recycle();
        }
    }

    // PNet 检测阶段，快速生成候选框
    private float[][] detectPNet(float[][][] data) {
        int height = data.length;
        int width = data[0].length;
        List<float[]> allBoxes = new ArrayList<>();

        // 构建图像金字塔
        float currentScale = 12f / minFaceSize;

        while (Math.min(height * currentScale, width * currentScale) > 12) {
            int h = (int) (height * currentScale);
            int w = (int) (width * currentScale);

            Tensor resizedInput = resizeImage(data, h, w);
            IValue[] resizedOutputs = pnet.forward(IValue.from(resizedInput)).toTuple();

            Tensor clsTensor = resizedOutputs[0].toTensor();
            Tensor regTensor = resizedOutputs[1].toTensor();
            float[] resizedCls = clsTensor.getDataAsFloatArray();
            float[] resizedReg = regTensor.getDataAsFloatArray();

            int clsHeight = getPNetMapHeight(clsTensor.shape());
            int clsWidth = getPNetMapWidth(clsTensor.shape());

            // 主判断：输出维度异常时跳过当前尺度，避免越界。
            if (clsHeight <= 0 || clsWidth <= 0) {
                currentScale *= scaleFactor;
                continue;
            }

            List<float[]> scaleBoxes = new ArrayList<>();
            for (int y = 0; y < clsHeight; y++) {
                for (int x = 0; x < clsWidth; x++) {
                    float backgroundLogit = getPNetOutputValue(resizedCls, clsTensor.shape(), 0, y, x);
                    float faceLogit = getPNetOutputValue(resizedCls, clsTensor.shape(), 1, y, x);
                    float score = softmaxPositive(backgroundLogit, faceLogit);

                    // 主判断：只有分数超过阈值时才生成候选框。
                    if (score > pnetThreshold) {
                        float dx = getPNetOutputValue(resizedReg, regTensor.shape(), 0, y, x);
                        float dy = getPNetOutputValue(resizedReg, regTensor.shape(), 1, y, x);
                        float dw = getPNetOutputValue(resizedReg, regTensor.shape(), 2, y, x);
                        float dh = getPNetOutputValue(resizedReg, regTensor.shape(), 3, y, x);

                        float left = Math.round((2f * x) / currentScale);
                        float top = Math.round((2f * y) / currentScale);
                        float right = Math.round((2f * x + 12f) / currentScale);
                        float bottom = Math.round((2f * y + 12f) / currentScale);

                        scaleBoxes.add(new float[]{left, top, right, bottom, score, dx, dy, dw, dh});
                    }
                }
            }

            // 主判断：当前尺度存在候选框时，先做一次 NMS。
            if (!scaleBoxes.isEmpty()) {
                float[][] scaleBoxesArray = scaleBoxes.toArray(new float[0][]);
                int[] keep = nmsIndices(scaleBoxesArray, 0.5f, 'u');
                for (int index : keep) {
                    allBoxes.add(scaleBoxesArray[index]);
                }
            }

            currentScale *= scaleFactor;
        }

        // 主判断：完全没有候选框时直接返回空数组。
        if (allBoxes.isEmpty()) {
            return new float[0][];
        }

        float[][] allBoxesArray = allBoxes.toArray(new float[0][]);
        int[] keep = nmsIndices(allBoxesArray, 0.7f, 'u');
        float[][] selectedBoxes = selectBoxes(allBoxesArray, keep);
        return calibratePNetBoxes(selectedBoxes);
    }

    // 获取 PNet 特征图高度，兼容被 squeeze 后的张量形状
    private int getPNetMapHeight(long[] shape) {
        if (shape.length >= 4) {
            return (int) shape[2];
        }
        if (shape.length == 3 || shape.length == 2) {
            return 1;
        }
        return 0;
    }

    // 获取 PNet 特征图宽度，兼容被 squeeze 后的张量形状
    private int getPNetMapWidth(long[] shape) {
        if (shape.length >= 4) {
            return (int) shape[3];
        }
        if (shape.length == 3) {
            return (int) shape[2];
        }
        if (shape.length == 2) {
            return 1;
        }
        return 0;
    }

    // 按当前张量形状读取 PNet 输出值
    private float getPNetOutputValue(float[] data, long[] shape, int channel, int y, int x) {
        if (shape.length >= 4) {
            int width = (int) shape[3];
            int height = (int) shape[2];
            int index = channel * height * width + y * width + x;
            return data[index];
        }
        if (shape.length == 3) {
            int width = (int) shape[2];
            int index = channel * width + x;
            return data[index];
        }
        if (shape.length == 2) {
            return data[channel];
        }
        throw new IllegalArgumentException("PNet 输出形状不支持: " + Arrays.toString(shape));
    }

    // RNet 检测阶段，精炼候选框
    private float[][] detectRNet(Bitmap bitmap, float[][] boxes) {
        // 主判断：没有输入候选框时直接返回。
        if (boxes == null || boxes.length == 0) {
            return new float[0][];
        }

        float[][] squareBoxes = roundBoxCoordinates(convertToSquare(boxes));
        int numBoxes = squareBoxes.length;
        float[][] croppedData = new float[numBoxes][3 * 24 * 24];
        int validCount = 0;
        float[][] validBoxes = new float[numBoxes][];

        for (int i = 0; i < numBoxes; i++) {
            float[] box = squareBoxes[i];
            int left = Math.round(box[0]);
            int top = Math.round(box[1]);
            int right = Math.round(box[2]);
            int bottom = Math.round(box[3]);

            int cropWidth = right - left + 1;
            int cropHeight = bottom - top + 1;
            // 主判断：过小候选框直接跳过。
            if (Math.min(cropWidth, cropHeight) < 20) {
                continue;
            }

            Bitmap cropped = cropAndPadBitmap(bitmap, left, top, right, bottom);
            Bitmap resized = Bitmap.createScaledBitmap(cropped, 24, 24, true);
            croppedData[validCount] = getBitmapPixels(resized);
            validBoxes[validCount] = box.clone();
            validCount++;

            if (!cropped.isRecycled()) {
                cropped.recycle();
            }
            if (!resized.isRecycled()) {
                resized.recycle();
            }
        }

        // 主判断：没有有效候选框时直接返回。
        if (validCount == 0) {
            return new float[0][];
        }

        float[] validData = flattenBatchData(croppedData, validCount);
        Tensor inputTensor = Tensor.fromBlob(validData, new long[]{validCount, 3, 24, 24});

        IValue[] outputs = rnet.forward(IValue.from(inputTensor)).toTuple();

        float[] clsData = outputs[0].toTensor().getDataAsFloatArray();
        float[] regData = outputs[1].toTensor().getDataAsFloatArray();

        List<float[]> candidateBoxes = new ArrayList<>();
        List<float[]> candidateRegs = new ArrayList<>();

        for (int i = 0; i < validCount; i++) {
            float score = softmaxPositive(clsData[i * 2], clsData[i * 2 + 1]);

            // 主判断：分数超过阈值才保留到下一阶段。
            if (score > rnetThreshold) {
                float[] box = validBoxes[i].clone();
                box[4] = score;
                candidateBoxes.add(box);
                candidateRegs.add(new float[]{
                        regData[i * 4],
                        regData[i * 4 + 1],
                        regData[i * 4 + 2],
                        regData[i * 4 + 3]
                });
            }
        }

        // 主判断：全部被过滤掉时直接返回空结果。
        if (candidateBoxes.isEmpty()) {
            return new float[0][];
        }

        float[][] candidateBoxesArray = candidateBoxes.toArray(new float[0][]);
        float[][] candidateRegsArray = candidateRegs.toArray(new float[0][]);
        int[] keep = nmsIndices(candidateBoxesArray, 0.6f, 'u');
        return calibrateBoxes(selectBoxes(candidateBoxesArray, keep), selectBoxes(candidateRegsArray, keep));
    }

    // ONet 检测阶段，输出最终检测框和关键点
    private float[][] detectONet(Bitmap bitmap, float[][] boxes) {
        // 主判断：没有输入候选框时直接返回。
        if (boxes == null || boxes.length == 0) {
            return new float[0][];
        }

        float[][] squareBoxes = roundBoxCoordinates(convertToSquare(boxes));
        int numBoxes = squareBoxes.length;
        float[][] croppedData = new float[numBoxes][3 * 48 * 48];
        float[][] validBoxes = new float[numBoxes][];
        int validCount = 0;

        for (int i = 0; i < numBoxes; i++) {
            float[] box = squareBoxes[i];
            int left = Math.round(box[0]);
            int top = Math.round(box[1]);
            int right = Math.round(box[2]);
            int bottom = Math.round(box[3]);

            int cropWidth = right - left + 1;
            int cropHeight = bottom - top + 1;
            // 主判断：非法尺寸直接跳过，避免 Bitmap 创建失败。
            if (cropWidth <= 0 || cropHeight <= 0) {
                continue;
            }

            Bitmap cropped = cropAndPadBitmap(bitmap, left, top, right, bottom);
            Bitmap resized = Bitmap.createScaledBitmap(cropped, 48, 48, true);
            croppedData[validCount] = getBitmapPixels(resized);
            validBoxes[validCount] = box.clone();
            validCount++;

            if (!cropped.isRecycled()) {
                cropped.recycle();
            }
            if (!resized.isRecycled()) {
                resized.recycle();
            }
        }

        // 主判断：没有有效输入时直接返回。
        if (validCount == 0) {
            return new float[0][];
        }

        float[] inputData = flattenBatchData(croppedData, validCount);
        Tensor inputTensor = Tensor.fromBlob(inputData, new long[]{validCount, 3, 48, 48});

        IValue[] outputs = onet.forward(IValue.from(inputTensor)).toTuple();

        float[] clsData = outputs[0].toTensor().getDataAsFloatArray();
        float[] regData = outputs[1].toTensor().getDataAsFloatArray();
        float[] landmarkData = outputs[2].toTensor().getDataAsFloatArray();

        List<float[]> candidateResults = new ArrayList<>();

        for (int i = 0; i < validCount; i++) {
            float score = softmaxPositive(clsData[i * 2], clsData[i * 2 + 1]);

            // 主判断：只有 ONet 判定为人脸时才输出最终结果。
            if (score > onetThreshold) {
                float[] box = validBoxes[i].clone();
                box[4] = score;
                float[] reg = new float[]{
                        regData[i * 4],
                        regData[i * 4 + 1],
                        regData[i * 4 + 2],
                        regData[i * 4 + 3]
                };

                float[] landmarks = new float[10];
                float boxWidth = box[2] - box[0] + 1f;
                float boxHeight = box[3] - box[1] + 1f;
                for (int j = 0; j < 5; j++) {
                    landmarks[j * 2] = landmarkData[i * 10 + j * 2] * boxWidth + box[0] - 1f;
                    landmarks[j * 2 + 1] = landmarkData[i * 10 + j * 2 + 1] * boxHeight + box[1] - 1f;
                }

                float[] calibratedBox = applyRegression(box, reg[0], reg[1], reg[2], reg[3]);
                candidateResults.add(new float[]{
                        calibratedBox[0], calibratedBox[1], calibratedBox[2], calibratedBox[3], calibratedBox[4],
                        landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4],
                        landmarks[5], landmarks[6], landmarks[7], landmarks[8], landmarks[9]
                });
            }
        }

        // 主判断：没有最终结果时直接返回。
        if (candidateResults.isEmpty()) {
            return new float[0][];
        }

        float[][] resultArray = candidateResults.toArray(new float[0][]);
        int[] keep = nmsIndices(resultArray, 0.6f, 'm');
        return selectBoxes(resultArray, keep);
    }

    // 非极大值抑制，返回保留的下标
    private int[] nmsIndices(float[][] boxes, float threshold, char mode) {
        // 主判断：空输入直接返回空下标。
        if (boxes == null || boxes.length == 0) {
            return new int[0];
        }

        Integer[] order = new Integer[boxes.length];
        for (int i = 0; i < boxes.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Float.compare(boxes[b][4], boxes[a][4]));

        List<Integer> remaining = new ArrayList<>(Arrays.asList(order));
        List<Integer> keep = new ArrayList<>();

        while (!remaining.isEmpty()) {
            int current = remaining.get(0);
            keep.add(current);

            List<Integer> nextRemaining = new ArrayList<>();
            for (int i = 1; i < remaining.size(); i++) {
                int candidate = remaining.get(i);
                float overlap = calculateIoU(boxes[current], boxes[candidate], mode);
                // 主判断：只保留重叠度不超过阈值的候选框。
                if (overlap <= threshold) {
                    nextRemaining.add(candidate);
                }
            }
            remaining = nextRemaining;
        }

        int[] keepIndices = new int[keep.size()];
        for (int i = 0; i < keep.size(); i++) {
            keepIndices[i] = keep.get(i);
        }
        return keepIndices;
    }

    // 计算两个边界框的 IoU
    private float calculateIoU(float[] box1, float[] box2, char mode) {
        float left1 = box1[0];
        float top1 = box1[1];
        float right1 = box1[2];
        float bottom1 = box1[3];
        float left2 = box2[0];
        float top2 = box2[1];
        float right2 = box2[2];
        float bottom2 = box2[3];

        float left = Math.max(left1, left2);
        float top = Math.max(top1, top2);
        float right = Math.min(right1, right2);
        float bottom = Math.min(bottom1, bottom2);

        float interWidth = Math.max(0f, right - left + 1f);
        float interHeight = Math.max(0f, bottom - top + 1f);
        // 主判断：没有交集时直接返回 0。
        if (interWidth <= 0f || interHeight <= 0f) {
            return 0f;
        }

        float interArea = interWidth * interHeight;
        float box1Area = (right1 - left1 + 1f) * (bottom1 - top1 + 1f);
        float box2Area = (right2 - left2 + 1f) * (bottom2 - top2 + 1f);

        // 主判断：mode 为 m 时按最小面积归一，否则按并集归一。
        if (mode == 'm') {
            return interArea / Math.min(box1Area, box2Area);
        }
        return interArea / (box1Area + box2Area - interArea + 1e-10f);
    }

    // 将 Bitmap 转换为浮点数组并进行归一化
    private float[][][] bitmapToFloatArray(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        float[][][] result = new float[height][width][3];

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = pixels[y * width + x];
                int r = (pixel >> 16) & 0xff;
                int g = (pixel >> 8) & 0xff;
                int b = pixel & 0xff;

                // 按 BGR 顺序归一化，与 Python/OpenCV 版本保持一致。
                result[y][x][0] = (b - PIXEL_MEAN[0]) / PIXEL_STD;
                result[y][x][1] = (g - PIXEL_MEAN[1]) / PIXEL_STD;
                result[y][x][2] = (r - PIXEL_MEAN[2]) / PIXEL_STD;
            }
        }
        return result;
    }

    // 从 Bitmap 获取归一化像素数据
    private float[] getBitmapPixels(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        float[] result = new float[3 * width * height];

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        int planeSize = width * height;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = pixels[y * width + x];
                int r = (pixel >> 16) & 0xff;
                int g = (pixel >> 8) & 0xff;
                int b = pixel & 0xff;

                int pixelIndex = y * width + x;
                result[pixelIndex] = (b - PIXEL_MEAN[0]) / PIXEL_STD;
                result[planeSize + pixelIndex] = (g - PIXEL_MEAN[1]) / PIXEL_STD;
                result[planeSize * 2 + pixelIndex] = (r - PIXEL_MEAN[2]) / PIXEL_STD;
            }
        }
        return result;
    }

    // 将批次图像数据展平为一维数组
    private float[] flattenBatchData(float[][] batchData, int batchSize) {
        // 主判断：空批次直接返回空数组。
        if (batchSize == 0) {
            return new float[0];
        }

        int sampleSize = batchData[0].length;
        float[] flatData = new float[batchSize * sampleSize];
        for (int i = 0; i < batchSize; i++) {
            System.arraycopy(batchData[i], 0, flatData, i * sampleSize, sampleSize);
        }
        return flatData;
    }

    // 将三维浮点数组转换为 PyTorch 张量
    private Tensor floatArrayToTensor(float[][][] data, long[] shape) {
        int size = (int) (shape[1] * shape[2] * shape[3]);
        float[] flat = new float[size];

        for (int c = 0; c < shape[1]; c++) {
            for (int y = 0; y < shape[2]; y++) {
                for (int x = 0; x < shape[3]; x++) {
                    flat[(int) (c * shape[2] * shape[3] + y * shape[3] + x)] = data[y][x][c];
                }
            }
        }

        return Tensor.fromBlob(flat, shape);
    }

    // 双线性插值缩放图像
    private Tensor resizeImage(float[][][] data, int targetHeight, int targetWidth) {
        int srcHeight = data.length;
        int srcWidth = data[0].length;
        int channels = data[0][0].length;

        float[][][] result = new float[targetHeight][targetWidth][channels];
        float scaleY = (float) srcHeight / targetHeight;
        float scaleX = (float) srcWidth / targetWidth;

        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                float srcY = y * scaleY;
                float srcX = x * scaleX;

                int y0 = (int) srcY;
                int x0 = (int) srcX;
                int y1 = Math.min(y0 + 1, srcHeight - 1);
                int x1 = Math.min(x0 + 1, srcWidth - 1);

                float yFrac = srcY - y0;
                float xFrac = srcX - x0;

                for (int c = 0; c < channels; c++) {
                    float v00 = data[y0][x0][c];
                    float v01 = data[y0][x1][c];
                    float v10 = data[y1][x0][c];
                    float v11 = data[y1][x1][c];

                    float v0 = v00 * (1 - xFrac) + v01 * xFrac;
                    float v1 = v10 * (1 - xFrac) + v11 * xFrac;
                    result[y][x][c] = v0 * (1 - yFrac) + v1 * yFrac;
                }
            }
        }

        return floatArrayToTensor(result, new long[]{1, channels, targetHeight, targetWidth});
    }

    // 只返回人脸类别的 softmax 概率
    private float softmaxPositive(float negativeLogit, float positiveLogit) {
        float maxLogit = Math.max(negativeLogit, positiveLogit);
        float negativeExp = (float) Math.exp(negativeLogit - maxLogit);
        float positiveExp = (float) Math.exp(positiveLogit - maxLogit);
        return positiveExp / (negativeExp + positiveExp);
    }

    // 将检测框转换为正方形
    private float[][] convertToSquare(float[][] boxes) {
        float[][] squareBoxes = new float[boxes.length][];
        for (int i = 0; i < boxes.length; i++) {
            float[] box = boxes[i];
            float[] squareBox = box.clone();
            float height = box[3] - box[1] + 1f;
            float width = box[2] - box[0] + 1f;
            float maxSide = Math.max(width, height);
            squareBox[0] = box[0] + width * 0.5f - maxSide * 0.5f;
            squareBox[1] = box[1] + height * 0.5f - maxSide * 0.5f;
            squareBox[2] = squareBox[0] + maxSide - 1f;
            squareBox[3] = squareBox[1] + maxSide - 1f;
            squareBoxes[i] = squareBox;
        }
        return squareBoxes;
    }

    // 将检测框坐标四舍五入
    private float[][] roundBoxCoordinates(float[][] boxes) {
        float[][] roundedBoxes = new float[boxes.length][];
        for (int i = 0; i < boxes.length; i++) {
            float[] roundedBox = boxes[i].clone();
            roundedBox[0] = Math.round(roundedBox[0]);
            roundedBox[1] = Math.round(roundedBox[1]);
            roundedBox[2] = Math.round(roundedBox[2]);
            roundedBox[3] = Math.round(roundedBox[3]);
            roundedBoxes[i] = roundedBox;
        }
        return roundedBoxes;
    }

    // 对单个检测框应用边框回归
    private float[] applyRegression(float[] box, float dx, float dy, float dw, float dh) {
        float width = box[2] - box[0] + 1f;
        float height = box[3] - box[1] + 1f;
        float[] calibratedBox = box.clone();
        calibratedBox[0] = calibratedBox[0] + dx * width;
        calibratedBox[1] = calibratedBox[1] + dy * height;
        calibratedBox[2] = calibratedBox[2] + dw * width;
        calibratedBox[3] = calibratedBox[3] + dh * height;
        return calibratedBox;
    }

    // 对一组检测框应用边框回归
    private float[][] calibrateBoxes(float[][] boxes, float[][] regs) {
        float[][] calibratedBoxes = new float[boxes.length][];
        for (int i = 0; i < boxes.length; i++) {
            calibratedBoxes[i] = applyRegression(boxes[i], regs[i][0], regs[i][1], regs[i][2], regs[i][3]);
        }
        return calibratedBoxes;
    }

    // PNet 候选框做一次统一校准
    private float[][] calibratePNetBoxes(float[][] boxes) {
        float[][] calibratedBoxes = new float[boxes.length][5];
        for (int i = 0; i < boxes.length; i++) {
            float[] calibrated = applyRegression(boxes[i], boxes[i][5], boxes[i][6], boxes[i][7], boxes[i][8]);
            System.arraycopy(calibrated, 0, calibratedBoxes[i], 0, 5);
        }
        return calibratedBoxes;
    }

    // 按下标提取检测框
    private float[][] selectBoxes(float[][] boxes, int[] indices) {
        float[][] selectedBoxes = new float[indices.length][];
        for (int i = 0; i < indices.length; i++) {
            selectedBoxes[i] = boxes[indices[i]].clone();
        }
        return selectedBoxes;
    }

    // 对越界区域补黑边后再裁剪
    private Bitmap cropAndPadBitmap(Bitmap bitmap, int left, int top, int right, int bottom) {
        int cropWidth = right - left + 1;
        int cropHeight = bottom - top + 1;
        Bitmap paddedBitmap = Bitmap.createBitmap(cropWidth, cropHeight, Bitmap.Config.ARGB_8888);

        int srcLeft = Math.max(0, left);
        int srcTop = Math.max(0, top);
        int srcRight = Math.min(bitmap.getWidth() - 1, right);
        int srcBottom = Math.min(bitmap.getHeight() - 1, bottom);
        // 主判断：与原图没有交集时直接返回黑色图块。
        if (srcLeft > srcRight || srcTop > srcBottom) {
            return paddedBitmap;
        }

        Bitmap sourceRegion = Bitmap.createBitmap(
                bitmap,
                srcLeft,
                srcTop,
                srcRight - srcLeft + 1,
                srcBottom - srcTop + 1
        );
        Canvas canvas = new Canvas(paddedBitmap);
        float dstLeft = Math.max(0, -left);
        float dstTop = Math.max(0, -top);
        canvas.drawBitmap(sourceRegion, dstLeft, dstTop, null);
        sourceRegion.recycle();
        return paddedBitmap;
    }

    // 释放 PyTorch 模型资源
    public void release() {
        if (pnet != null) {
            pnet.destroy();
            pnet = null;
        }
        if (rnet != null) {
            rnet.destroy();
            rnet = null;
        }
        if (onet != null) {
            onet.destroy();
            onet = null;
        }
    }
}
