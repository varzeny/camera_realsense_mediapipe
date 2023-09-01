# 영상에서 객체의 깊이값 추출

import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 바인딩 속성
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)
origin_x, origin_y = 0, 0
width, height = 0, 0

#깊이 추출 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

# 인식된 부분 바인딩 함수
def visualize(image, detection_result) -> np.ndarray:
    global origin_x, origin_y, width, height, category_name
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        origin_x, origin_y = bbox.origin_x, bbox.origin_y
        width, height = bbox.width, bbox.height 
        start_point = origin_x, origin_y
        end_point = origin_x + width, origin_y + height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + origin_x,
                         MARGIN + ROW_SIZE + origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
        
    return image

cap = cv2.VideoCapture(1)

model_path = 'data/model/efficientdet_lite2_int8.tflite'
base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.ObjectDetectorOptions(base_options=base_options,
                                       running_mode=vision.RunningMode.IMAGE,
                                       max_results=1,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

while True: # image는 mediapipe 관련 frame은 realsense관련

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    depth_info = depth_frame.as_depth_frame()

    ret, images = cap.read()
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=images)
    if not ret:
        print("카메라를 찾을 수 없습니다.")
        continue
    
    detection_result = detector.detect(image)
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    
    cv2.imshow('annotated_image', annotated_image)
    key = cv2.waitKey(1)
    if key & 0xff == ord('q') or key == 27: 
        cv2.destroyAllWindows()
        break

    if key == ord("s"):
        cv2.imwrite('data/image/out.png', annotated_image)
    
    if key == ord("g"):
        # if category_name == 'dining table':
        #     mx = origin_x + int(width / 2)
        #     my = origin_y + int(height / 2)
        #     print(detection_result)
            
        #     print('category_name : ', category_name)
        #     print('x : ', mx, ' y : ', my, 'z : ', (depth_info.get_distance(mx, my)) * 100, 'cm')
        mx = origin_x + int(width / 2)
        my = origin_y + int(height / 2)
        print(detection_result)
        
        print('category_name : ', category_name)
        print('x : ', mx, ' y : ', my, 'z : ', (depth_info.get_distance(mx, my)) * 100, 'cm')

cv2.destroyAllWindows()