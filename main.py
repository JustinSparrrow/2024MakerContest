import cv2
import numpy as np
import mediapipe as mp
import time

# 调用关键点检测模型
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,  # 使用视频流时应设为False
                                  max_num_faces=1,         
                                  refine_landmarks=True,  
                                  min_detection_confidence=0.5, 
                                  min_tracking_confidence=0.5) 

# mediapipe提供的绘制模块
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 定义左右眼、左右眉毛、嘴唇和鼻子的连接
LEFT_EYE = mp_face_mesh.FACEMESH_LEFT_EYE
RIGHT_EYE = mp_face_mesh.FACEMESH_RIGHT_EYE
LEFT_EYEBROW = mp_face_mesh.FACEMESH_LEFT_EYEBROW
RIGHT_EYEBROW = mp_face_mesh.FACEMESH_RIGHT_EYEBROW
LIPS = mp_face_mesh.FACEMESH_LIPS
NOSE = mp_face_mesh.FACEMESH_NOSE

# 加载替换图片
left_eye_img = cv2.imread('images/left_eye.png', cv2.IMREAD_UNCHANGED)
right_eye_img = cv2.imread('images/right_eye.png', cv2.IMREAD_UNCHANGED)
left_eyebrow_img = cv2.imread('images/left_eyebrow.png', cv2.IMREAD_UNCHANGED)
right_eyebrow_img = cv2.imread('images/right_eyebrow.png', cv2.IMREAD_UNCHANGED)
lips_img = cv2.imread('images/lips.png', cv2.IMREAD_UNCHANGED)
nose_img = cv2.imread('images/nose.png', cv2.IMREAD_UNCHANGED)

# 添加alpha通道的函数
def add_alpha_channel(image):
    b_channel, g_channel, r_channel = cv2.split(image)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建一个全白的Alpha通道
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

# 检查并添加Alpha通道
def ensure_alpha_channel(image):
    if image is not None and image.shape[2] != 4:
        return add_alpha_channel(image)
    return image

left_eye_img = ensure_alpha_channel(left_eye_img)
right_eye_img = ensure_alpha_channel(right_eye_img)
left_eyebrow_img = ensure_alpha_channel(left_eyebrow_img)
right_eyebrow_img = ensure_alpha_channel(right_eyebrow_img)
lips_img = ensure_alpha_channel(lips_img)
nose_img = ensure_alpha_channel(nose_img)

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0表示默认摄像头

# 初始化帧率计算变量
prev_frame_time = 0
new_frame_time = 0

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` on top of `img` at (x, y) and blend using `alpha_mask`.

    Alpha mask must contain values within the range [0, 1] and be the same size as `img_overlay`.
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    img[y1:y2, x1:x2] = (1. - alpha) * img_crop + alpha * img_overlay_crop

# 循环读取视频每一帧
while True:
    success, frame = cap.read()
    if not success:
        print("忽略空帧")
        continue

    # 将BGR图像转为RGB图像
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 使用模型获取关键点
    results = face_mesh.process(frame_rgb)

    # 计算帧率
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # 将帧率转换为字符串
    fps_text = f'FPS: {int(fps)}'

    # 如果检测到关键点
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 获取关键点坐标
            h, w, c = frame.shape
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
            
            # 绘制左右眼、左右眉毛、嘴唇和鼻子
            mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks,
                                      connections=LEFT_EYE,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks,
                                      connections=RIGHT_EYE,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks,
                                      connections=LEFT_EYEBROW,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks,
                                      connections=RIGHT_EYEBROW,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks,
                                      connections=LIPS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks,
                                      connections=NOSE,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # 获取左眼区域并替换
            if left_eye_img is not None:
                left_eye_points = [landmarks[i] for i in [33, 133, 160, 159, 158, 144, 145, 153, 154, 155]]
                left_eye_x = min([pt[0] for pt in left_eye_points])
                left_eye_y = min([pt[1] for pt in left_eye_points])
                left_eye_w = max([pt[0] for pt in left_eye_points]) - left_eye_x
                left_eye_h = max([pt[1] for pt in left_eye_points]) - left_eye_y
                left_eye_resized = cv2.resize(left_eye_img, (left_eye_w, left_eye_h))
                overlay_image_alpha(frame, left_eye_resized[:, :, :3], left_eye_x, left_eye_y, left_eye_resized[:, :, 3] / 255.0)

            # 获取右眼区域并替换
            if right_eye_img is not None:
                right_eye_points = [landmarks[i] for i in [362, 263, 387, 386, 385, 373, 374, 380, 381, 382]]
                right_eye_x = min([pt[0] for pt in right_eye_points])
                right_eye_y = min([pt[1] for pt in right_eye_points])
                right_eye_w = max([pt[0] for pt in right_eye_points]) - right_eye_x
                right_eye_h = max([pt[1] for pt in right_eye_points]) - right_eye_y
                right_eye_resized = cv2.resize(right_eye_img, (right_eye_w, right_eye_h))
                overlay_image_alpha(frame, right_eye_resized[:, :, :3], right_eye_x, right_eye_y, right_eye_resized[:, :, 3] / 255.0)

            # 获取左眉毛区域并替换
            if left_eyebrow_img is not None:
                left_eyebrow_points = [landmarks[i] for i in [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]]
                left_eyebrow_x = min([pt[0] for pt in left_eyebrow_points])
                left_eyebrow_y = min([pt[1] for pt in left_eyebrow_points])
                left_eyebrow_w = max([pt[0] for pt in left_eyebrow_points]) - left_eyebrow_x
                left_eyebrow_h = max([pt[1] for pt in left_eyebrow_points]) - left_eyebrow_y
                left_eyebrow_resized = cv2.resize(left_eyebrow_img, (left_eyebrow_w, left_eyebrow_h))
                overlay_image_alpha(frame, left_eyebrow_resized[:, :, :3], left_eyebrow_x, left_eyebrow_y, left_eyebrow_resized[:, :, 3] / 255.0)

            # 获取右眉毛区域并替换
            if right_eyebrow_img is not None:
                right_eyebrow_points = [landmarks[i] for i in [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]]
                right_eyebrow_x = min([pt[0] for pt in right_eyebrow_points])
                right_eyebrow_y = min([pt[1] for pt in right_eyebrow_points])
                right_eyebrow_w = max([pt[0] for pt in right_eyebrow_points]) - right_eyebrow_x
                right_eyebrow_h = max([pt[1] for pt in right_eyebrow_points]) - right_eyebrow_y
                right_eyebrow_resized = cv2.resize(right_eyebrow_img, (right_eyebrow_w, right_eyebrow_h))
                overlay_image_alpha(frame, right_eyebrow_resized[:, :, :3], right_eyebrow_x, right_eyebrow_y, right_eyebrow_resized[:, :, 3] / 255.0)

            # 获取嘴唇区域并替换
            if lips_img is not None:
                lips_points = [landmarks[i] for i in [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]]
                lips_x = min([pt[0] for pt in lips_points])
                lips_y = min([pt[1] for pt in lips_points])
                lips_w = max([pt[0] for pt in lips_points]) - lips_x
                lips_h = max([pt[1] for pt in lips_points]) - lips_y
                lips_resized = cv2.resize(lips_img, (lips_w, lips_h))
                overlay_image_alpha(frame, lips_resized[:, :, :3], lips_x, lips_y, lips_resized[:, :, 3] / 255.0)

            # 获取鼻子区域并替换
            if nose_img is not None:
                nose_points = [landmarks[i] for i in [1, 2, 98, 327, 168]]
                nose_x = min([pt[0] for pt in nose_points])
                nose_y = min([pt[1] for pt in nose_points])
                nose_w = max([pt[0] for pt in nose_points]) - nose_x
                nose_h = max([pt[1] for pt in nose_points]) - nose_y
                nose_resized = cv2.resize(nose_img, (nose_w, nose_h))
                overlay_image_alpha(frame, nose_resized[:, :, :3], nose_x, nose_y, nose_resized[:, :, 3] / 255.0)

    # 在帧上绘制帧率
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示图像
    cv2.imshow('MediaPipe Face Mesh', frame)

    # 按下'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()