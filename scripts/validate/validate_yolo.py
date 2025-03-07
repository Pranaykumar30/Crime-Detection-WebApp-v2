from ultralytics import YOLO

def validate_yolo(model_path, data_yaml):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split='test')
    results = {'mAP50': metrics.box.map50, 'mAP50-95': metrics.box.map}
    print(f'YOLOv8 Validation Metrics: {results}')
    with open('models/yolo_validation_results.txt', 'w') as f:
        f.write(f'YOLOv8: {results}')
    return results

if __name__ == '__main__':
    validate_yolo('models/yolo/yolov8_custom/weights/best.pt', 'data.yaml')
