package com.yeyupiaoling.mobilefacenet;

import androidx.annotation.NonNull;

import java.util.Arrays;

// 人脸数据结构
public class FaceData {
    // 人脸框左边界
    public float left;
    // 人脸框上边界
    public float top;
    // 人脸框右边界
    public float right;
    // 人脸框下边界
    public float bottom;
    // 检测置信度
    public float score;
    // 5 个关键点坐标，按 x、y 交替排列
    public float[] landmarks;

    public FaceData(float left, float top, float right, float bottom, float score, float[] landmarks) {
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
        this.score = score;
        this.landmarks = landmarks;
    }

    @NonNull
    @Override
    public String toString() {
        return "FaceData{"
                + "left=" + left
                + ", top=" + top
                + ", right=" + right
                + ", bottom=" + bottom
                + ", score=" + score
                + ", landmarks=" + Arrays.toString(landmarks)
                + '}';
    }
}
