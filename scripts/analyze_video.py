from ultralytics import YOLO
import tensorflow as tf
import cv2
import numpy as np
import os

def analyze_video(yolo_model_path, mobilenet_model_path, video_path):
    yolo_model = YOLO(yolo_model_path)
    mobilenet_model = tf.keras.models.load_model(mobilenet_model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError('Cannot open video file')
    class_names = ['handguns', 'knives', 'sharp-edged-weapons', 'masked-intruders', 'violence', 'normal-behavior']
    yolo_tp, yolo_fp, yolo_fn = 0, 0, 0
    mobile_tp, mobile_fp = 0, 0
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # YOLOv8 analysis
        yolo_results = yolo_model.predict(frame, conf=0.2, imgsz=(640, 640), verbose=False)
        yolo_detections = [int(box.cls) for result in yolo_results for box in result.boxes]
        has_handgun = 0 in yolo_detections  # Class 0 is handguns
        wrong_classes = len([cls for cls in yolo_detections if cls != 0])
        yolo_tp += 1 if has_handgun else 0
        yolo_fp += wrong_classes
        yolo_fn += 1 if not has_handgun else 0
        # MobileNet analysis
        mobilenet_input = cv2.resize(frame, (224, 224))
        mobilenet_input = np.expand_dims(mobilenet_input / 255.0, axis=0)
        mobilenet_pred = mobilenet_model.predict(mobilenet_input, verbose=0)
        mobilenet_cls = np.argmax(mobilenet_pred)
        mobile_tp += 1 if mobilenet_cls == 0 else 0  # 0 is handguns
        mobile_fp += 1 if mobilenet_cls != 0 else 0
    cap.release()
    # Calculate metrics
    yolo_precision = yolo_tp / (yolo_tp + yolo_fp) if (yolo_tp + yolo_fp) > 0 else 0
    yolo_recall = yolo_tp / (yolo_tp + yolo_fn) if (yolo_tp + yolo_fn) > 0 else 0
    mobile_accuracy = mobile_tp / frame_count if frame_count > 0 else 0
    results = {
        'yolo': {'true_positives': yolo_tp, 'false_positives': yolo_fp, 'false_negatives': yolo_fn,
                 'precision': yolo_precision, 'recall': yolo_recall},
        'mobilenet': {'true_positives': mobile_tp, 'false_positives': mobile_fp, 'accuracy': mobile_accuracy}
    }
    with open('data/videos/output/video_analysis.txt', 'w') as f:
        f.write(f'YOLOv8 - TP: {yolo_tp}, FP: {yolo_fp}, FN: {yolo_fn}, Precision: {yolo_precision:.2f}, Recall: {yolo_recall:.2f}')
        f.write(f'MobileNet - TP: {mobile_tp}, FP: {mobile_fp}, Accuracy: {mobile_accuracy:.2f}')
    print(f'Analysis saved to data/videos/output/video_analysis.txt')
    return results

if __name__ == '__main__':
    os.makedirs('data/videos/output', exist_ok=True)
    analyze_video(
        'models/yolo/yolov8_custom/weights/best.pt',
        'models/mobilenet/mobilenet_custom.h5',
        'data/videos/sample.mp4'
    )
