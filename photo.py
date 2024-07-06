import cv2
import numpy as np
import mediapipe as mp
import os

# 调用关键点检测模型
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,  # 使用静态图像时应设为True
                                  max_num_faces=1,         
                                  refine_landmarks=True,  
                                  min_detection_confidence=0.5, 
                                  min_tracking_confidence=0.5) 

# mediapipe提供的绘制模块
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 读取图像
image_path = "image.jpg"
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Unable to read the image file {image_path}.")
else:
    # 将BGR图像转为RGB图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 使用模型获取关键点
    results = face_mesh.process(img_rgb)

    # 获取图像的高度和宽度
    h, w, _ = img.shape

    # 定义要提取的面部区域的关键点索引
    LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153, 154, 155]
    RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 373, 374, 380, 381, 382]
    LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    RIGHT_EYEBROW_INDICES = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    LIPS_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
    NOSE_INDICES = [1, 2, 98, 327, 168]

    # 定义保存抠图的函数
    def save_cropped_image(image, indices, filename, w, h):
        # 创建一个透明背景的图像
        transparent_image = np.zeros((h, w, 4), dtype=np.uint8)
        
        # 创建一个掩码
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices]
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

        # 将原始图像区域复制到透明背景图像上
        bgr_image = cv2.bitwise_and(image, image, mask=mask)
        alpha_channel = np.zeros_like(mask)
        alpha_channel[mask == 255] = 255
        transparent_image[:, :, :3] = bgr_image
        transparent_image[:, :, 3] = alpha_channel

        # 保存图像
        x, y, w, h = cv2.boundingRect(points)
        cropped_image = transparent_image[y:y+h, x:x+w]
        cv2.imwrite(filename, cropped_image)

    # 如果检测到关键点
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks

            # 保存左眼图像
            save_cropped_image(img, LEFT_EYE_INDICES, "images/left_eye.png", w, h)

            # 保存右眼图像
            save_cropped_image(img, RIGHT_EYE_INDICES, "images/right_eye.png", w, h)

            # 保存左眉毛图像
            save_cropped_image(img, LEFT_EYEBROW_INDICES, "images/left_eyebrow.png", w, h)

            # 保存右眉毛图像
            save_cropped_image(img, RIGHT_EYEBROW_INDICES, "images/right_eyebrow.png", w, h)

            # 保存嘴唇图像
            save_cropped_image(img, LIPS_INDICES, "images/lips.png", w, h)

            # 保存鼻子图像
            save_cropped_image(img, NOSE_INDICES, "images/nose.png", w, h)

            # 绘制眼睛、瞳孔、眉毛、嘴巴和脸颊
            mp_drawing.draw_landmarks(image=img, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_IRISES,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

            mp_drawing.draw_landmarks(image=img, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

    # 显示图像
    cv2.imshow('MediaPipe Face Mesh', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()