from ultralytics import YOLO
import os

def train_yolo():
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='/workspaces/Crime-Detection-WebApp-v2/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        name='yolov8_custom',
        project='models/yolo',
        exist_ok=True)
    return results

if __name__ == '__main__':
    os.makedirs('models/yolo', exist_ok=True)
    train_yolo()
