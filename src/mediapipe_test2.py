# 영상에서 객체인식

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)

def visualize(image, detection_result) -> np.ndarray:
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
        
    return image



    
cap = cv2.VideoCapture(1)

model_path = 'data/model/efficientdet_lite0_int8.tflite'
base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.ObjectDetectorOptions(base_options=base_options,
                                       running_mode=vision.RunningMode.IMAGE,
                                       max_results=5,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

while True:
    ret, frame = cap.read()
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
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
    
    if key == ord("g"):
        print(detection_result)

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()

