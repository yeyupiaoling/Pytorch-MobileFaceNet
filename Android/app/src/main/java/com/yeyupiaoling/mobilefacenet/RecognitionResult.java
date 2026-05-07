package com.yeyupiaoling.mobilefacenet;

import androidx.annotation.NonNull;

// 单张人脸的识别结果
public class RecognitionResult {
    // 检测到的人脸信息
    public final FaceData faceData;
    // 识别名称
    public final String name;
    // 与人脸库最佳匹配的相似度
    public final float similarity;

    public RecognitionResult(FaceData faceData, String name, float similarity) {
        this.faceData = faceData;
        this.name = name;
        this.similarity = similarity;
    }

    @NonNull
    @Override
    public String toString() {
        return "RecognitionResult{" +
                "name='" + name + '\'' +
                ", similarity=" + similarity +
                ", faceData=" + faceData +
                '}';
    }
}
