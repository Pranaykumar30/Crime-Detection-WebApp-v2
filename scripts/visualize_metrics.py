import matplotlib.pyplot as plt
import numpy as np

def visualize_metrics(log_path, output_path):
    frames, yolo_preds, mobile_preds = [], [], []
    class_names = ['handguns', 'knives', 'sharp-edged-weapons', 'masked-intruders', 'violence', 'normal-behavior']
    with open(log_path, 'r') as f:
        for line in f:
            frame = int(line.split('Frame ')[1].split(':')[0])
            yolo_part = line.split('YOLOv8: ')[1].split(', MobileNet: ')[0].strip()
            mobile_part = line.split('MobileNet: ')[1].strip()
            # YOLOv8: 1 if handguns detected, 0 if not
            yolo_preds.append(1 if 'handguns' in yolo_part else 0)
            # MobileNet: Class index
            mobile_class = mobile_part.split(' ')[0]
            mobile_preds.append(class_names.index(mobile_class))
            frames.append(frame)
    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(frames, yolo_preds, label='YOLOv8 Handguns (1=Detected)', color='green')
    plt.title('YOLOv8 Handguns Detection')
    plt.xlabel('Frame')
    plt.ylabel('Detected (1/0)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(frames, mobile_preds, label='MobileNet Class Index', color='red')
    plt.title('MobileNet Class Predictions')
    plt.xlabel('Frame')
    plt.ylabel('Class Index')
    plt.yticks(range(len(class_names)), class_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f'Plot saved to {output_path}')

if __name__ == '__main__':
    visualize_metrics(
        'data/videos/output/prediction_log.txt',
        'data/videos/output/performance_plot.png'
    )
