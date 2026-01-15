import cv2
import dlib
import time

# 人脸关键点检测预训练模型路径
# 下载地址: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
PREDICTOR_PATH = "./weights/shape_predictor_68_face_landmarks.dat"

# 定义各部位的都关键点索引范围 (dlib 68-point)
FACIAL_LANDMARKS_68_IDXS = {
    "jaw": (0, 17),
    "right_eyebrow": (17, 22),
    "left_eyebrow": (22, 27),
    "nose_bridge": (27, 31),
    "nose_tip": (31, 36),
    "right_eye": (36, 42),
    "left_eye": (42, 48),
    "outer_lip": (48, 60),
    "inner_lip": (60, 68)
}


class DlibDetector:
    def __init__(self, weights=PREDICTOR_PATH, is_box=True, is_point=True, is_line=True, upsample=1) -> None:
        """初始化Dlib检测器

        Args:
            weights: 预训练模型路径
            is_box: 是否绘制边界框
            is_point: 是否绘制关键点
            is_line: 是否绘制连线
            upsample: 上采样次数，提高检测小脸的能力
        """
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(weights)
        self.draw_box = is_box
        self.draw_point = is_point
        self.draw_line = is_line
        self.upsample = upsample

    def detect(self, img):
        """检测人脸

        Args:
            img: 输入图像

        Returns:
            检测到的人脸区域列表
        """
        return self.face_detector(img, self.upsample)

    def _get_facial_landmarks(self, img, face_region):
        """获取人脸关键点坐标

        Args:
            img: 输入图像
            face_region: 人脸区域

        Returns:
            关键点坐标列表
        """
        landmarks = self.landmark_predictor(img, face_region)
        return [[point.x, point.y] for point in landmarks.parts()]

    def _draw_face_box(self, img, face_region):
        """绘制人脸边界框

        Args:
            img: 要绘制的图像
            face_region: 人脸区域
        """
        left = face_region.left()
        top = face_region.top()
        right = face_region.right()
        bottom = face_region.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    def _draw_landmark_points(self, img, facial_landmarks):
        """绘制关键点及索引编号

        Args:
            img: 要绘制的图像
            facial_landmarks: 关键点坐标列表
        """
        for landmark_index, landmark_point in enumerate(facial_landmarks):
            point_coordinates = (landmark_point[0], landmark_point[1])

            # 绘制关键点（红色圆点）
            cv2.circle(img, point_coordinates, 5, color=(0, 0, 255), thickness=-1)

            # 设置字体并绘制关键点索引
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(img,
                       str(landmark_index),
                       point_coordinates,
                       font,
                       0.5,
                       (0, 255, 0),
                       1,
                       cv2.LINE_AA)

    def _draw_landmark_lines(self, img, facial_landmarks):
        """绘制关键点连线

        Args:
            img: 要绘制的图像
            facial_landmarks: 关键点坐标列表
        """
        # 绘制连续线段
        for (start, end) in FACIAL_LANDMARKS_68_IDXS.values():
            for i in range(start, end - 1):
                pt1 = tuple(facial_landmarks[i])
                pt2 = tuple(facial_landmarks[i + 1])
                cv2.line(img, pt1, pt2, color=(255, 0, 0), thickness=1)

        # 特别处理闭合轮廓（眼睛和嘴唇需要首尾相连）
        connection_points = [
            (41, 36),  # 右眼
            (47, 42),  # 左眼
            (59, 48),  # 外唇
            (67, 60)   # 内唇
        ]

        for start_idx, end_idx in connection_points:
            cv2.line(img,
                    tuple(facial_landmarks[start_idx]),
                    tuple(facial_landmarks[end_idx]),
                    (255, 255, 255), 1)

    def draw_face_info(self, img, detected_faces):
        """绘制人脸信息（兼容旧接口）

        Args:
            img: 要绘制的图像
            detected_faces: 检测到的人脸区域列表
        """
        for face_region in detected_faces:
            # 绘制人脸边界框
            if self.draw_box:
                self._draw_face_box(img, face_region)

            # 绘制关键点
            if self.draw_point or self.draw_line:
                facial_landmarks = self._get_facial_landmarks(img, face_region)

                if self.draw_point:
                    self._draw_landmark_points(img, facial_landmarks)

                if self.draw_line:
                    self._draw_landmark_lines(img, facial_landmarks)


def process_camera(detector, camera_id=0):
    """处理摄像头视频流

    Args:
        detector: Dlib检测器实例
        camera_id: 摄像头ID（0为默认摄像头）
    """
    cap = cv2.VideoCapture(camera_id)
    frame_count = 0
    t2, t1 = 0, 0

    try:
        t1 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头画面")
                break

            detected_faces = detector.detect(frame.copy())
            detector.draw_face_info(frame, detected_faces)

            cv2.imshow('Face Landmark Detection - Camera', frame)

            key = cv2.waitKey(1) & 0xFF

            frame_count += 1

            if key == ord('q'):
                print("退出摄像头模式")
                break
        t2 = time.time()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"摄像头模式结束，共处理 {frame_count} 帧")
        print(f"总耗时：{(t2 - t1):.3f}，单帧耗时：{((t2 - t1) / frame_count):.3f}")


if __name__ == "__main__":
    detector = DlibDetector(is_point=True)
    process_camera(detector, 0)
