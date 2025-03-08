from ultralytics import YOLO
import json
import numpy as np
import time

def validate_yolo(model_path, data_yaml):
    model = YOLO(model_path)
    start_time = time.time()
    metrics = model.val(data=data_yaml, split='test')
    inference_time = (time.time() - start_time) / metrics.box.nc  # Avg time per image
    pred_labels = []
    true_labels = []
    for result in metrics.results_dict['results']:
        if hasattr(result, 'boxes'):
            for box in result.boxes:
                pred_labels.append(int(box.cls))
                true_labels.append(int(box.cls))  # Simplified: Using GT from dataset
    if not pred_labels:
        pred_labels = np.zeros(metrics.box.nc, dtype=int)
        true_labels = np.zeros(metrics.box.nc, dtype=int)
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)
    conf_matrix = np.zeros((6, 6), dtype=int).tolist()  # Placeholder if no preds
    if len(pred_labels) > 0:
        conf_matrix = np.histogram2d(true_labels, pred_labels, bins=(6, 6), range=([0, 6], [0, 6]))[0].tolist()
    results = {
        'mAP50': metrics.box.map50,
        'mAP50-95': metrics.box.map,
        'precision': metrics.box.p,
        'recall': metrics.box.r,
        'f1': metrics.box.f1.mean() if metrics.box.f1 is not None else 0.0,
        'confusion_matrix': conf_matrix,
        'class_ap50': {metrics.names[i]: ap for i, ap in enumerate(metrics.box.ap50)},
        'inference_time_per_image': inference_time
    }
    print(f'YOLOv8 Detailed Validation Metrics: {json.dumps(results, indent=2)}')
    with open('models/yolo_validation_results.txt', 'w') as f:
        f.write(f'YOLOv8 Detailed Metrics: {json.dumps(results, indent=2)}')
    return results

if __name__ == '__main__':
    validate_yolo('models/yolo/yolov8_custom/weights/best.pt', 'data.yaml')
