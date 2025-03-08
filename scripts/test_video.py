from ultralytics import YOLO
import tensorflow as tf
import cv2
import numpy as np
import os

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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # YOLOv8 prediction
        yolo_results = yolo_model.predict(frame, verbose=False)
        for result in yolo_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = float(box.conf)
                label = f'{class_names[cls]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # MobileNet prediction
        mobilenet_input = cv2.resize(frame, (224, 224))
        mobilenet_input = np.expand_dims(mobilenet_input / 255.0, axis=0)
        mobilenet_pred = mobilenet_model.predict(mobilenet_input, verbose=0)
        mobilenet_cls = np.argmax(mobilenet_pred)
        mobilenet_conf = float(mobilenet_pred[0][mobilenet_cls])
        mobilenet_label = f'MobileNet: {class_names[mobilenet_cls]} {mobilenet_conf:.2f}'
        cv2.putText(frame, mobilenet_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        out.write(frame)
    cap.release()
    out.release()
    print(f'Processed video saved to {output_path}')

if __name__ == '__main__':
    os.makedirs('data/videos/output', exist_ok=True)
    process_video(
        'models/yolo/yolov8_custom/weights/best.pt',
        'models/mobilenet/mobilenet_custom.h5',
        'data/videos/sample_video.mp4',
        'data/videos/output/annotated_video.mp4'
    )
