# 예제 이미지에서 객체인식

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

image_file = 'data/image/dogcat.jpg'

# img = cv2.imread(image_file)
# cv2.imshow('img', img)
# cv2.waitKey()
# cv2.destroyAllWindows()

model_path = 'data/model/efficientdet_lite0_int8.tflite'
base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.ObjectDetectorOptions(base_options=base_options,
                                       max_results=5,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

image = mp.Image.create_from_file(image_file)

detection_result = detector.detect(image)
print(detection_result)
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
cv2.imshow('annotated_image', rgb_annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
