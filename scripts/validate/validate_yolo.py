from ultralytics import YOLO
import json
import numpy as np
import os
import time
import cv2

def validate_yolo(model_path, data_dir):
    model = YOLO(model_path)
    img_dir = os.path.join(data_dir, 'test', 'images')
    lbl_dir = os.path.join(data_dir, 'test', 'labels')
    save_dir = 'models/yolo/yolometrics'
    os.makedirs(save_dir, exist_ok=True)
    
    images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
    true_labels = []
    pred_labels = []
    start_time = time.time()
    results = model.predict(images, verbose=False)
    inference_time = (time.time() - start_time) / len(images)  # Avg time per image
    
    for img_path, result in zip(images, results):
        lbl_file = os.path.basename(img_path).replace('.jpg', '.txt')
        lbl_path = os.path.join(lbl_dir, lbl_file)
        
        with open(lbl_path, 'r') as f:
            gt_classes = [int(line.split()[0]) for line in f if line.strip()]
        
        pred_classes = [int(box.cls) for box in result.boxes] if result.boxes else []
        true_label = gt_classes[0] if gt_classes else 5  # Default to 'normal-behavior'
        pred_label = pred_classes[0] if pred_classes else 5  # Default to 'normal-behavior'
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        
        # Save annotated images
        img = cv2.imread(img_path)
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{int(box.cls.item())}: {box.conf.item():.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, img)
    
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    conf_matrix = np.histogram2d(true_labels, pred_labels, bins=(6, 6), range=([0, 6], [0, 6]))[0].tolist()
    
    metrics = model.val(data='data.yaml', split='test')
    results_dict = {
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.p.mean()) if metrics.box.p is not None else 0.0,
        'recall': float(metrics.box.r.mean()) if metrics.box.r is not None else 0.0,
        'f1': float(metrics.box.f1.mean()) if metrics.box.f1 is not None else 0.0,
        'confusion_matrix': conf_matrix,
        'IoU': float(metrics.box.iou.mean()) if metrics.box.iou is not None else 0.0,
        'class_ap50': {metrics.names[i]: float(ap) for i, ap in enumerate(metrics.box.ap50)},
        'class_recall': {metrics.names[i]: float(rec) for i, rec in enumerate(metrics.box.r)},
        'class_precision': {metrics.names[i]: float(prec) for i, prec in enumerate(metrics.box.p)},
        'inference_time_per_image': float(inference_time)
    }
    
    print(f'YOLOv8 Detailed Validation Metrics: {json.dumps(results_dict, indent=2)}')
    
    with open('models/yolo_validation_results.txt', 'w') as f:
        f.write(f'YOLOv8 Detailed Metrics: {json.dumps(results_dict, indent=2)}')
    
    return results_dict

if __name__ == '__main__':
    validate_yolo('models/yolo/yolov8_custom/weights/best.pt', 'data/split')
