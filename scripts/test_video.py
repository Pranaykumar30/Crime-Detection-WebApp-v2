from ultralytics import YOLO
import tensorflow as tf
import cv2
import numpy as np
import os
from collections import deque

def process_video(yolo_model_path, mobilenet_model_path, video_path, output_path):
    yolo_model = YOLO(yolo_model_path)
    mobilenet_model = tf.keras.models.load_model(mobilenet_model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError('Cannot open video file')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    class_names = ['handguns', 'knives', 'sharp-edged-weapons', 'masked-intruders', 'violence', 'normal-behavior']

    frame_count = 0
    log_file = open('data/videos/output/prediction_log.txt', 'w')
    mobilenet_buffer = deque(maxlen=10)  # 10-frame buffer

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        yolo_detections = []  # ✅ Initialize this variable before use

        # YOLOv8 prediction
        yolo_results = yolo_model.predict(frame, conf=0.2, imgsz=640, verbose=True)
        if yolo_results and len(yolo_results[0].boxes) > 0:
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = float(box.conf)
                label = f'{class_names[cls]} {conf:.2f}'
                yolo_detections.append(label)  # ✅ Store detections in list
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"Frame {frame_count}: Detected {label} at {x1, y1, x2, y2}")
        else:
            print(f"Frame {frame_count}: YOLO detected nothing")

        # MobileNet prediction with majority vote
        mobilenet_input = cv2.resize(frame, (224, 224))
        mobilenet_input = np.expand_dims(mobilenet_input / 255.0, axis=0)
        mobilenet_pred = mobilenet_model.predict(mobilenet_input, verbose=0)
        mobilenet_cls = np.argmax(mobilenet_pred)
        mobilenet_conf = float(mobilenet_pred[0][mobilenet_cls])
        mobilenet_buffer.append((mobilenet_cls, mobilenet_conf))

        if len(mobilenet_buffer) == 10:
            class_counts = np.bincount([cls for cls, _ in mobilenet_buffer])
            majority_cls = np.argmax(class_counts)
            majority_conf = np.mean([conf for cls, conf in mobilenet_buffer if cls == majority_cls])
        else:
            majority_cls, majority_conf = mobilenet_cls, mobilenet_conf

        mobilenet_label = f'MobileNet: {class_names[majority_cls]} {majority_conf:.2f}'
        cv2.putText(frame, mobilenet_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # ✅ Now yolo_detections is always defined before logging
        log_file.write(f'Frame {frame_count}: YOLOv8: {yolo_detections}, MobileNet: {mobilenet_label}\n')
        
        out.write(frame)

    cap.release()
    out.release()
    log_file.close()

    print(f'Processed video saved to {output_path}, log saved to data/videos/output/prediction_log.txt')

if __name__ == '__main__':
    os.makedirs('data/videos/output', exist_ok=True)
    process_video(
        '/workspaces/Crime-Detection-WebApp-v2/models/yolo/yolov8_custom/weights/best.pt',
        'models/mobilenet/mobilenet_custom.h5',
        'data/videos/sample_2.mp4',
        'data/videos/output/annotated_video.mp4'
    )
