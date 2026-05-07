package com.yeyupiaoling.mobilefacenet;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.NonNull;

import java.util.List;
import java.util.Locale;

public class FaceOverlayView extends View {
    // 绘制人脸框的画笔
    private Paint facePaint;
    // 绘制关键点的画笔
    private Paint landmarkPaint;
    // 绘制标签文本的画笔
    private Paint textPaint;
    // 绘制标签背景的画笔
    private Paint textBackgroundPaint;
    // 当前识别结果
    private List<RecognitionResult> recognitionResults;
    // 源图宽度
    private int sourceImageWidth;
    // 源图高度
    private int sourceImageHeight;

    public FaceOverlayView(Context context) {
        super(context);
        init();
    }

    public FaceOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public FaceOverlayView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    // 初始化画笔
    private void init() {
        facePaint = new Paint();
        facePaint.setColor(Color.GREEN);
        facePaint.setStyle(Paint.Style.STROKE);
        facePaint.setStrokeWidth(5f);

        landmarkPaint = new Paint();
        landmarkPaint.setColor(Color.RED);
        landmarkPaint.setStyle(Paint.Style.FILL);
        landmarkPaint.setStrokeWidth(8f);

        textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(38f);
        textPaint.setStyle(Paint.Style.FILL);

        textBackgroundPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        textBackgroundPaint.setColor(Color.argb(180, 0, 0, 0));
        textBackgroundPaint.setStyle(Paint.Style.FILL);
    }

    // 设置识别结果和源图尺寸
    public void setResults(List<RecognitionResult> recognitionResults, int sourceImageWidth, int sourceImageHeight) {
        this.recognitionResults = recognitionResults;
        this.sourceImageWidth = sourceImageWidth;
        this.sourceImageHeight = sourceImageHeight;
        invalidate();
    }

    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);

        // 主判断：没有识别结果时无需绘制。
        if (recognitionResults == null) {
            return;
        }

        // 主判断：源图尺寸无效时无法做坐标映射。
        if (sourceImageWidth <= 0 || sourceImageHeight <= 0) {
            return;
        }

        // 主判断：当前画布尺寸无效时直接返回。
        if (getWidth() <= 0 || getHeight() <= 0) {
            return;
        }

        float scale = Math.max(getWidth() / (float) sourceImageWidth,
                getHeight() / (float) sourceImageHeight);
        float offsetX = (getWidth() - sourceImageWidth * scale) / 2f;
        float offsetY = (getHeight() - sourceImageHeight * scale) / 2f;

        for (RecognitionResult result : recognitionResults) {
            FaceData face = result.faceData;
            float left = face.left * scale + offsetX;
            float top = face.top * scale + offsetY;
            float right = face.right * scale + offsetX;
            float bottom = face.bottom * scale + offsetY;

            // 主判断：未知人脸用红框，已知人脸用绿框，便于快速区分。
            if ("unknown".equals(result.name)) {
                facePaint.setColor(Color.RED);
            } else {
                facePaint.setColor(Color.GREEN);
            }
            canvas.drawRect(left, top, right, bottom, facePaint);

            String label = String.format(Locale.CHINA, "%s %.2f", result.name, result.similarity);
            float textWidth = textPaint.measureText(label);
            float textTop = Math.max(44f, top - 48f);
            canvas.drawRect(left, textTop - 34f, left + textWidth + 16f, textTop + 8f, textBackgroundPaint);
            canvas.drawText(label, left + 8f, textTop, textPaint);

            // 主判断：只有关键点完整时才绘制关键点。
            if (face.landmarks != null && face.landmarks.length >= 10) {
                for (int i = 0; i < 5; i++) {
                    float x = face.landmarks[i * 2] * scale + offsetX;
                    float y = face.landmarks[i * 2 + 1] * scale + offsetY;
                    canvas.drawCircle(x, y, 6f, landmarkPaint);
                }
            }
        }
    }
}
