package com.yeyupiaoling.mobilefacenet;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
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
import java.util.List;
import java.util.Locale;

public class FaceRecognizer {
    private static final String TAG = "FaceRecognizer";
    private static final float PIXEL_MEAN = 127.5f;
    private static final float PIXEL_STD = 127.5f;
    private static final int IMAGE_SIZE = 112;
    private static final float DEFAULT_THRESHOLD = 0.6f;
    // 使用 4 个关键点估计人脸对齐变换，兼容 Android Matrix.setPolyToPoly 的点数限制
    private static final float[] ALIGNMENT_TEMPLATE = new float[]{
            38.2946f, 51.6963f,
            73.5318f, 51.5014f,
            41.5493f, 92.3655f,
            70.7299f, 92.2041f
    };

    private final Context context;
    private MTCNNDetector mtcnnDetector;
    private final float threshold;
    private Module mobileFaceNetModule;
    private final List<FaceDbEntry> faceDbEntries = new ArrayList<>();

    public static class FaceDbEntry {
        public String name;
        public float[] feature;
        public File file;
    }

    public List<FaceDbEntry> getFaceDbEntries() {
        return faceDbEntries;
    }

    public FaceRecognizer(Context context) {
        this(context, DEFAULT_THRESHOLD);
    }

    public FaceRecognizer(Context context, float threshold) {
        this.context = context.getApplicationContext();
        this.threshold = threshold;
        this.mtcnnDetector = new MTCNNDetector(this.context);
        try {
            mobileFaceNetModule = Module.load(assetFilePath(this.context, "mobilefacenet.pt"));
            updateFaceDb();
            new Handler(Looper.getMainLooper()).post(() -> {
                Toast.makeText(this.context,
                        String.format(Locale.CHINA, "人脸库加载完成，共 %d 人", faceDbEntries.size()),
                        Toast.LENGTH_SHORT).show();
            });
        } catch (Exception e) {
            Log.e(TAG, "加载识别模型失败", e);
            new Handler(Looper.getMainLooper()).post(() -> {
                Toast.makeText(this.context, "MobileFaceNet 模型加载失败", Toast.LENGTH_LONG).show();
            });
        }
    }

    // 识别人脸并返回结果
    public List<RecognitionResult> recognize(Bitmap bitmap) {
        List<RecognitionResult> results = new ArrayList<>();

        // 主判断：模型或输入无效时直接返回空结果。
        if (bitmap == null || mobileFaceNetModule == null) {
            return results;
        }

        List<FaceData> faces = mtcnnDetector.detect(bitmap);
        // 主判断：没有检测到人脸时直接返回。
        if (faces.isEmpty()) {
            return results;
        }

        for (FaceData face : faces) {
            Bitmap alignedFace = alignFace(bitmap, face);
            // 主判断：对齐失败时回退到按检测框裁剪，尽量保证仍可识别。
            if (alignedFace == null) {
                alignedFace = cropFace(bitmap, face);
            }

            // 主判断：裁剪或对齐依然失败时，返回 unknown 结果。
            if (alignedFace == null) {
                results.add(new RecognitionResult(face, "unknown", 0f));
                continue;
            }

            float[] feature = extractFeature(alignedFace);
            alignedFace.recycle();

            // 主判断：特征提取失败时直接标记为 unknown。
            if (feature == null || feature.length == 0) {
                results.add(new RecognitionResult(face, "unknown", 0f));
                continue;
            }

            String bestName = "unknown";
            float bestSimilarity = -1f;
            for (FaceDbEntry entry : faceDbEntries) {
                float similarity = cosineSimilarity(feature, entry.feature);
                // 主判断：找到更高相似度时更新最佳匹配。
                if (similarity > bestSimilarity) {
                    bestSimilarity = similarity;
                    bestName = entry.name;
                }
            }

            // 主判断：人脸库为空或相似度低于阈值时，统一标记为 unknown。
            if (bestSimilarity < threshold || faceDbEntries.isEmpty()) {
                bestName = "unknown";
            }
            if (bestSimilarity < 0f) {
                bestSimilarity = 0f;
            }
            results.add(new RecognitionResult(face, bestName, bestSimilarity));
        }
        return results;
    }

    // 重新加载人脸库
    public void updateFaceDb() {
        faceDbEntries.clear();
        File dbDir = new File(context.getFilesDir(), "face_db");
        // 主判断：如果人脸库目录不存在，则创建并从 assets 拷贝初始数据
        if (!dbDir.exists()) {
            dbDir.mkdirs();
            AssetManager assetManager = context.getAssets();
            try {
                String[] fileNames = assetManager.list("face_db");
                if (fileNames != null) {
                    for (String fileName : fileNames) {
                        try (InputStream is = assetManager.open("face_db/" + fileName);
                             OutputStream os = new FileOutputStream(new File(dbDir, fileName))) {
                            byte[] buffer = new byte[1024];
                            int length;
                            while ((length = is.read(buffer)) > 0) {
                                os.write(buffer, 0, length);
                            }
                        }
                    }
                }
            } catch (Exception e) {
                Log.e(TAG, "拷贝初始人脸库失败", e);
            }
        }

        try {
            File[] files = dbDir.listFiles();
            // 主判断：人脸库目录为空时直接结束，避免空指针。
            if (files == null || files.length == 0) {
                Log.w(TAG, "内部存储 face_db 目录为空");
                return;
            }

            for (File file : files) {
                String fileName = file.getName();
                String lowerName = fileName.toLowerCase(Locale.ROOT);
                // 主判断：只处理常见图片文件。
                if (!lowerName.endsWith(".jpg")
                        && !lowerName.endsWith(".jpeg")
                        && !lowerName.endsWith(".png")
                        && !lowerName.endsWith(".bmp")
                        && !lowerName.endsWith(".webp")) {
                    continue;
                }

                try {
                    Bitmap bitmap = BitmapFactory.decodeFile(file.getAbsolutePath());
                    // 主判断：图片解码失败时跳过该人脸库条目。
                    if (bitmap == null) {
                        continue;
                    }

                    List<FaceData> faces = mtcnnDetector.detect(bitmap);
                    // 主判断：人脸库图片必须恰好只有一张人脸，否则跳过。
                    if (faces.size() != 1) {
                        bitmap.recycle();
                        continue;
                    }

                    FaceData face = faces.get(0);
                    Bitmap alignedFace = alignFace(bitmap, face);
                    // 主判断：没有关键点时再回退到检测框裁剪。
                    if (alignedFace == null) {
                        alignedFace = cropFace(bitmap, face);
                    }
                    bitmap.recycle();

                    // 主判断：仍无法得到有效人脸图时跳过。
                    if (alignedFace == null) {
                        continue;
                    }

                    float[] feature = extractFeature(alignedFace);
                    alignedFace.recycle();

                    // 主判断：特征提取失败时跳过该条目。
                    if (feature == null || feature.length == 0) {
                        continue;
                    }

                    FaceDbEntry entry = new FaceDbEntry();
                    entry.name = removeExtension(fileName);
                    entry.feature = feature;
                    entry.file = file;
                    faceDbEntries.add(entry);
                } catch (Exception e) {
                    Log.w(TAG, "加载人脸库条目失败: " + fileName, e);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "加载人脸库失败", e);
        }
    }

    // 从 assets 中拷贝文件到应用私有目录，供 PyTorch Android 加载
    private String assetFilePath(Context context, String assetPath) throws Exception {
        File file = new File(context.getFilesDir(), assetPath.replace('/', '_'));
        // 主判断：文件已存在且非空时，直接复用以避免重复拷贝。
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream inputStream = context.getAssets().open(assetPath);
             OutputStream outputStream = new FileOutputStream(file)) {
            byte[] buffer = new byte[4 * 1024];
            int readSize;
            while ((readSize = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, readSize);
            }
            outputStream.flush();
        }
        return file.getAbsolutePath();
    }

    // 基于关键点对齐人脸到标准输入尺寸
    private Bitmap alignFace(Bitmap sourceBitmap, FaceData faceData) {
        // 主判断：关键点不完整时无法做对齐，返回空让上层回退。
        if (faceData.landmarks == null || faceData.landmarks.length < 10) {
            return null;
        }

        float[] sourcePoints = new float[]{
                faceData.landmarks[0], faceData.landmarks[1],
                faceData.landmarks[2], faceData.landmarks[3],
                faceData.landmarks[6], faceData.landmarks[7],
                faceData.landmarks[8], faceData.landmarks[9]
        };
        Matrix matrix = new Matrix();
        // 主判断：变换矩阵构建失败时返回空，避免生成畸形图像。
        if (!matrix.setPolyToPoly(sourcePoints, 0, ALIGNMENT_TEMPLATE, 0, 4)) {
            return null;
        }

        Bitmap alignedBitmap = Bitmap.createBitmap(IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(alignedBitmap);
        canvas.drawColor(Color.BLACK);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        canvas.drawBitmap(sourceBitmap, matrix, paint);
        return alignedBitmap;
    }

    // 按检测框裁剪人脸，作为对齐失败时的兜底方案
    private Bitmap cropFace(Bitmap sourceBitmap, FaceData faceData) {
        int left = Math.max(0, Math.round(faceData.left));
        int top = Math.max(0, Math.round(faceData.top));
        int right = Math.min(sourceBitmap.getWidth(), Math.round(faceData.right));
        int bottom = Math.min(sourceBitmap.getHeight(), Math.round(faceData.bottom));
        int width = right - left;
        int height = bottom - top;

        // 主判断：裁剪区域非法时直接返回空。
        if (width <= 0 || height <= 0) {
            return null;
        }

        Bitmap croppedBitmap = Bitmap.createBitmap(sourceBitmap, left, top, width, height);
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(croppedBitmap, IMAGE_SIZE, IMAGE_SIZE, true);
        croppedBitmap.recycle();
        return resizedBitmap;
    }

    // 提取单张人脸特征
    private float[] extractFeature(Bitmap faceBitmap) {
        // 主判断：输入为空时直接返回空。
        if (faceBitmap == null || mobileFaceNetModule == null) {
            return null;
        }

        Bitmap resizedBitmap;
        // 主判断：输入尺寸不一致时先缩放到模型要求的 112x112。
        if (faceBitmap.getWidth() != IMAGE_SIZE || faceBitmap.getHeight() != IMAGE_SIZE) {
            resizedBitmap = Bitmap.createScaledBitmap(faceBitmap, IMAGE_SIZE, IMAGE_SIZE, true);
        } else {
            resizedBitmap = faceBitmap;
        }

        float[] inputData = bitmapToTensorData(resizedBitmap);
        Tensor inputTensor = Tensor.fromBlob(inputData, new long[]{1, 3, IMAGE_SIZE, IMAGE_SIZE});
        Tensor outputTensor = mobileFaceNetModule.forward(IValue.from(inputTensor)).toTensor();
        float[] features = outputTensor.getDataAsFloatArray();

        // 主判断：如果中间创建了临时缩放图，提取完成后立即释放。
        if (resizedBitmap != faceBitmap && !resizedBitmap.isRecycled()) {
            resizedBitmap.recycle();
        }
        return features;
    }

    // 将 Bitmap 转成模型需要的 CHW 浮点数组
    private float[] bitmapToTensorData(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int planeSize = width * height;
        float[] result = new float[3 * planeSize];
        int[] pixels = new int[planeSize];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = pixels[y * width + x];
                int r = (pixel >> 16) & 0xff;
                int g = (pixel >> 8) & 0xff;
                int b = pixel & 0xff;
                int pixelIndex = y * width + x;
                // 按 BGR 顺序写入，和桌面端推理预处理保持一致。
                result[pixelIndex] = (b - PIXEL_MEAN) / PIXEL_STD;
                result[planeSize + pixelIndex] = (g - PIXEL_MEAN) / PIXEL_STD;
                result[planeSize * 2 + pixelIndex] = (r - PIXEL_MEAN) / PIXEL_STD;
            }
        }
        return result;
    }

    // 计算余弦相似度
    private float cosineSimilarity(float[] a, float[] b) {
        // 主判断：维度不一致或为空时返回 0，避免越界或除零。
        if (a == null || b == null || a.length == 0 || a.length != b.length) {
            return 0f;
        }

        float dot = 0f;
        float normA = 0f;
        float normB = 0f;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        float denominator = (float) (Math.sqrt(normA) * Math.sqrt(normB));
        // 主判断：分母接近 0 时返回 0，避免异常结果。
        if (denominator < 1e-8f) {
            return 0f;
        }
        return dot / denominator;
    }

    private String removeExtension(String fileName) {
        int index = fileName.lastIndexOf('.');
        // 主判断：没有扩展名时直接返回原文件名。
        if (index <= 0) {
            return fileName;
        }
        return fileName.substring(0, index);
    }

    // 增加人脸到人脸库
    public boolean addFace(String name, Bitmap bitmap) {
        // 主判断：参数不合法时直接返回失败
        if (bitmap == null || name == null || name.isEmpty()) {
            return false;
        }
        File dbDir = new File(context.getFilesDir(), "face_db");
        // 主判断：确保人脸库目录存在
        if (!dbDir.exists()) {
            dbDir.mkdirs();
        }
        File newFile = new File(dbDir, name + ".jpg");
        try (FileOutputStream out = new FileOutputStream(newFile)) {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
            out.flush();
            updateFaceDb(); // 重新加载人脸库
            return true;
        } catch (Exception e) {
            Log.e(TAG, "添加人脸失败", e);
            return false;
        }
    }

    // 删除人脸
    public boolean deleteFace(FaceDbEntry entry) {
        // 主判断：文件不存在时无法删除
        if (entry == null || entry.file == null || !entry.file.exists()) {
            return false;
        }
        boolean deleted = entry.file.delete();
        // 主判断：只有文件真实删除成功才从内存中移除
        if (deleted) {
            faceDbEntries.remove(entry);
        }
        return deleted;
    }

    // 重命名人脸
    public boolean renameFace(FaceDbEntry entry, String newName) {
        // 主判断：参数不合法或原文件不存在时返回失败
        if (entry == null || entry.file == null || !entry.file.exists() || newName == null || newName.isEmpty()) {
            return false;
        }
        File newFile = new File(entry.file.getParent(), newName + ".jpg");
        // 主判断：新名字已存在时不允许重命名
        if (newFile.exists()) {
            return false; // 新名字已存在
        }
        boolean renamed = entry.file.renameTo(newFile);
        // 主判断：重命名成功后更新内存中的文件引用与名称
        if (renamed) {
            entry.file = newFile;
            entry.name = newName;
        }
        return renamed;
    }

    public void release() {
        if (mobileFaceNetModule != null) {
            mobileFaceNetModule.destroy();
            mobileFaceNetModule = null;
        }
        if (mtcnnDetector != null) {
            mtcnnDetector.release();
            mtcnnDetector = null;
        }
    }
}
