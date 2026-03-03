#!/usr/bin/env python3

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import os


def load_reference_face(reference_img_path):
    """
    加载本地参考人脸图片，提取人脸特征向量
    :param reference_img_path: 参考图片路径
    :return: 参考人脸特征向量（None表示加载失败）
    """
    # 初始化InsightFace
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 检查文件是否存在
    if not os.path.exists(reference_img_path):
        print(f"错误：参考图片 {reference_img_path} 不存在！")
        return None

    # 读取图片
    img = cv2.imread(reference_img_path)
    if img is None:
        print(f"错误：无法读取参考图片 {reference_img_path}！")
        return None

    # 检测人脸
    faces = app.get(img)
    if len(faces) == 0:
        print("错误：参考图片中未检测到人脸！")
        return None
    elif len(faces) > 1:
        print("警告：参考图片中检测到多张人脸，将使用第一张人脸作为基准")

    # 返回第一张人脸的特征向量
    reference_embedding = faces[0].embedding
    print(f"✅ 参考人脸加载成功，特征向量维度：{len(reference_embedding)}")
    return reference_embedding


def realtime_face_compare(reference_embedding):
    """
    打开摄像头，实时采集人脸并与参考人脸比对
    :param reference_embedding: 参考人脸特征向量
    """
    try:
        app = FaceAnalysis()
        app.prepare(ctx_id=0, det_size=(640, 640))

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测摄像头中的人脸
            faces = app.get(frame)
            result_img = frame.copy()

            # 绘制检测结果和比对信息
            if len(faces) > 0:
                # 遍历检测到的每一张人脸
                for idx, face in enumerate(faces):
                    # 获取当前人脸特征向量
                    cur_embedding = face.embedding
                    # 计算余弦相似度（值越大越相似，0-1之间）
                    similarity = np.dot(reference_embedding, cur_embedding) / (norm(reference_embedding) * norm(cur_embedding))
                    # 判断是否为同一人（阈值可调整，一般0.5-0.6为宜）
                    is_same_person = similarity > 0.6
                    color = (0, 255, 0) if is_same_person else (0, 0, 255)
                    label = f"Match: {similarity:.4f} ({'YES' if is_same_person else 'NO'})"

                    # 绘制人脸框
                    bbox = face.bbox.astype(int)
                    cv2.rectangle(result_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    # 绘制106个关键点
                    if hasattr(face, 'landmark_2d_106'):
                        for point in face.landmark_2d_106.astype(int):
                            cv2.circle(result_img, tuple(point), 2, (255, 0, 0), -1)
                    # 绘制性别、年龄、相似度信息
                    gender = 'Male' if face.gender == 1 else 'Female'
                    text_y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[3] + 20
                    cv2.putText(result_img, f"Age: {face.age}, Sex: {gender}",
                                (bbox[0], text_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(result_img, label,
                                (bbox[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 绘制总检测人数
                cv2.putText(
                    result_img,
                    f"Total Faces: {len(faces)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2
                )
            else:
                # 未检测到人脸时的提示
                cv2.putText(
                    result_img,
                    "No face detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            # 显示画面
            cv2.imshow('InsightFace Real-time Compare', result_img)

            # 按ESC退出
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"摄像头比对失败: {e}")


def main():
    reference_img_path = "./test.jpg"
    # 加载参考人脸
    reference_embedding = load_reference_face(reference_img_path)
    if reference_embedding is None:
        print("参考人脸加载失败，程序退出！")
        return

    # 2. 启动实时比对
    realtime_face_compare(reference_embedding)


if __name__ == "__main__":
    main()
