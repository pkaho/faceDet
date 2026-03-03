import cv2
from insightface.app import FaceAnalysis


def test_with_webcam():
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

            # 检测人脸
            faces = app.get(frame)

            # 绘制结果
            if len(faces) > 0:
                result_img = app.draw_on(frame, faces)
            else:
                result_img = frame

            # 显示统计信息
            for face in faces:
                # 绘制点
                if hasattr(face, 'landmark_2d_106'):
                    for point in face.landmark_2d_106.astype(int):
                        cv2.circle(result_img, tuple(point), 2, (0, 0, 255), -1)

                gender = 'man' if face.gender == 1 else 'woman'
                cv2.putText(
                    result_img,
                    f'Faces: {len(faces)}, Conf: {face.det_score:.2f}, Sex: {gender}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            cv2.imshow('InsightFace Webcam', result_img)

            # 按ESC退出
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"摄像头测试失败: {e}")


def main():
    """主函数"""
    test_with_webcam()

if __name__ == "__main__":
    main()
