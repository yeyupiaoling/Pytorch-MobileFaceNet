package com.yeyupiaoling.mobilefacenet;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.Button;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    // 相机权限请求码
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 100;
    // 图像分析分辨率
    private static final Size ANALYSIS_SIZE = new Size(480, 640);

    private ProcessCameraProvider cameraProvider;
    private ImageAnalysis imageAnalysis;
    private ExecutorService cameraExecutor;
    private FaceRecognizer faceRecognizer;
    private FaceOverlayView faceOverlayView;

    // 当前摄像头方向，默认前置
    private int currentLensFacing = CameraSelector.LENS_FACING_FRONT;
    // 防止重复处理同一帧
    private boolean isProcessingFrame = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        faceOverlayView = findViewById(R.id.faceOverlayView);
        Button switchCameraButton = findViewById(R.id.switchCameraButton);
        switchCameraButton.setOnClickListener(v -> switchCamera());
        
        Button startFaceDbButton = findViewById(R.id.startFaceDbButton);
        startFaceDbButton.setOnClickListener(v -> {
            android.content.Intent intent = new android.content.Intent(MainActivity.this, FaceDbActivity.class);
            startActivity(intent);
        });

        cameraExecutor = Executors.newSingleThreadExecutor();
        new Thread(() -> {
            faceRecognizer = new FaceRecognizer(this);
        }).start();

        // 主判断：已经授权时直接启动相机，否则先请求权限。
        if (checkCameraPermission()) {
            startCamera();
        } else {
            requestCameraPermission();
        }
    }

    // 检查相机权限
    private boolean checkCameraPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    // 请求相机权限
    private void requestCameraPermission() {
        ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.CAMERA},
                CAMERA_PERMISSION_REQUEST_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        // 主判断：只有当前权限请求成功时才启动相机。
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Toast.makeText(this, "相机权限被拒绝", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    // 启动相机
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindCameraUseCases();
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "启动相机失败", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    // 绑定预览与分析用例
    private void bindCameraUseCases() {
        // 主判断：相机提供者未准备好时直接返回。
        if (cameraProvider == null) {
            return;
        }

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(currentLensFacing)
                .build();

        Preview preview = new Preview.Builder().build();
        imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(ANALYSIS_SIZE)
                // 只保留最新帧，避免识别速度跟不上时不断积压。
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
        imageAnalysis.setAnalyzer(cameraExecutor, this::analyzeFrame);

        cameraProvider.unbindAll();
        try {
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
            androidx.camera.view.PreviewView previewView = findViewById(R.id.previewView);
            preview.setSurfaceProvider(previewView.getSurfaceProvider());
        } catch (Exception e) {
            Log.e(TAG, "绑定相机用例失败", e);
        }
    }

    // 分析每一帧图像
    private void analyzeFrame(ImageProxy imageProxy) {
        // 主判断：上一帧还在处理时直接丢弃当前帧，避免重复堆积。
        if (isProcessingFrame) {
            imageProxy.close();
            return;
        }

        isProcessingFrame = true;
        try (imageProxy) {
            Bitmap bitmap = imageProxyToBitmap(imageProxy);
            // 主判断：转换失败时不做识别。
            if (bitmap == null) {
                return;
            }
            if (faceRecognizer == null) {
                return;
            }

            long startTime = System.currentTimeMillis();
            List<RecognitionResult> results = faceRecognizer.recognize(bitmap);
            Log.d(TAG, "识别耗时: " + (System.currentTimeMillis() - startTime) + "ms，结果数 " + results);

            int sourceImageWidth = bitmap.getWidth();
            int sourceImageHeight = bitmap.getHeight();
            runOnUiThread(() -> faceOverlayView.setResults(results, sourceImageWidth, sourceImageHeight));
            bitmap.recycle();
        } catch (Exception e) {
            Log.e(TAG, "分析图像失败", e);
        } finally {
            isProcessingFrame = false;
        }
    }

    // 将 ImageProxy 的 YUV 图像转换为 Bitmap
    private Bitmap imageProxyToBitmap(ImageProxy imageProxy) {
        ImageProxy.PlaneProxy[] planes = imageProxy.getPlanes();
        // 主判断：YUV 三平面不完整时直接返回空。
        if (planes.length < 3) {
            return null;
        }

        int width = imageProxy.getWidth();
        int height = imageProxy.getHeight();

        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        byte[] yBytes = new byte[yBuffer.remaining()];
        byte[] uBytes = new byte[uBuffer.remaining()];
        byte[] vBytes = new byte[vBuffer.remaining()];
        yBuffer.get(yBytes);
        uBuffer.get(uBytes);
        vBuffer.get(vBytes);

        int yRowStride = planes[0].getRowStride();
        int yPixelStride = planes[0].getPixelStride();
        int uRowStride = planes[1].getRowStride();
        int uPixelStride = planes[1].getPixelStride();
        int vRowStride = planes[2].getRowStride();
        int vPixelStride = planes[2].getPixelStride();

        int[] argb = new int[width * height];
        for (int y = 0; y < height; y++) {
            int yRowOffset = y * yRowStride;
            int uvRowOffset = y / 2;
            int uRowOffset = uvRowOffset * uRowStride;
            int vRowOffset = uvRowOffset * vRowStride;

            for (int x = 0; x < width; x++) {
                int yValue = yBytes[yRowOffset + x * yPixelStride] & 0xff;
                int uValue = uBytes[uRowOffset + (x / 2) * uPixelStride] & 0xff;
                int vValue = vBytes[vRowOffset + (x / 2) * vPixelStride] & 0xff;

                int c = Math.max(0, yValue - 16);
                int d = uValue - 128;
                int e = vValue - 128;

                int r = clampToByte((298 * c + 409 * e + 128) >> 8);
                int g = clampToByte((298 * c - 100 * d - 208 * e + 128) >> 8);
                int b = clampToByte((298 * c + 516 * d + 128) >> 8);
                argb[y * width + x] = 0xff000000 | (r << 16) | (g << 8) | b;
            }
        }

        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        bitmap.setPixels(argb, 0, width, 0, 0, width, height);

        int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();
        // 主判断：需要旋转或镜像时，转换出与预览一致方向的新图像。
        if (rotationDegrees != 0 || currentLensFacing == CameraSelector.LENS_FACING_FRONT) {
            Matrix matrix = new Matrix();
            if (rotationDegrees != 0) {
                matrix.postRotate(rotationDegrees);
            }
            if (currentLensFacing == CameraSelector.LENS_FACING_FRONT) {
                float centerX = rotationDegrees % 180 == 0 ? width / 2f : height / 2f;
                float centerY = rotationDegrees % 180 == 0 ? height / 2f : width / 2f;
                matrix.postScale(-1, 1, centerX, centerY);
            }
            Bitmap transformedBitmap = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
            bitmap.recycle();
            return transformedBitmap;
        }
        return bitmap;
    }

    // 限制像素值到 0-255
    private int clampToByte(int value) {
        return Math.max(0, Math.min(255, value));
    }

    // 切换前后摄像头
    private void switchCamera() {
        // 主判断：相机尚未初始化时不做切换。
        if (cameraProvider == null) {
            return;
        }
        currentLensFacing = currentLensFacing == CameraSelector.LENS_FACING_FRONT
                ? CameraSelector.LENS_FACING_BACK
                : CameraSelector.LENS_FACING_FRONT;
        bindCameraUseCases();
    }

    @Override
    protected void onResume() {
        super.onResume();
        // 恢复时更新人脸库
        if (faceRecognizer != null) {
            faceRecognizer.updateFaceDb();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (faceRecognizer != null) {
            faceRecognizer.release();
        }
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
    }
}
