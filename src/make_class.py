import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class Camera_xyz:

    def __init__(self):
        global pipeline, align, category_name
        self.MARGIN = 10
        self.ROW_SIZE = 10
        self.FONT_SIZE = 1
        self.FONT_THICKNESS = 1
        self.TEXT_COLOR = (0, 0, 255)
        category_name = ''
        
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)

        align_to = rs.stream.color
        align = rs.align(align_to)    


    def settng(self):
        global detector, cap
        model_path = 'data/model/efficientdet_lite0_int8.tflite'
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.ObjectDetectorOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, max_results=5, score_threshold=0.5)
        detector = vision.ObjectDetector.create_from_options(options)

        cap = cv2.VideoCapture(1)


    def visualize(self, image, detection_result) -> np.ndarray:
        global origin_x, origin_y, width, height, category_name
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            origin_x, origin_y = bbox.origin_x, bbox.origin_y
            width, height = bbox.width, bbox.height 
            start_point = origin_x, origin_y
            end_point = origin_x + width, origin_y + height
            cv2.rectangle(image, start_point, end_point, self.TEXT_COLOR, 3)

            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (self.MARGIN + origin_x, self.MARGIN + self.ROW_SIZE + origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, self.FONT_SIZE, self.TEXT_COLOR, self.FONT_THICKNESS)
            
        return image
    

    def object_detected(self, category_name_check):
        self.settng()
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            depth_info = depth_frame.as_depth_frame()
        
            ret, images = cap.read()
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=images)
            if not ret:
                continue
            
            detection_result = detector.detect(image)
            image_copy = np.copy(image.numpy_view())
            annotated_image = self.visualize(image_copy, detection_result)
            
            cv2.imshow('annotated_image', annotated_image)
            cv2.waitKey(1)

            # if category_name == category_name_check:
            #     x = origin_x + int(width / 2)
            #     y = origin_y + int(height / 2)
            #     z = depth_info.get_distance(x, y) * 100
                
            #     cv2.destroyAllWindows()
            #     break

        return category_name, x, y, z

