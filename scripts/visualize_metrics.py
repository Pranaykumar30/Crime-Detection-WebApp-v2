import matplotlib.pyplot as plt
import numpy as np

def visualize_metrics(log_path, output_path):
    frames, yolo_preds, mobile_preds = [], [], []
    class_names = ['handguns', 'knives', 'sharp-edged-weapons', 'masked-intruders', 'violence', 'normal-behavior']
    with open(log_path, 'r') as f:
        lines = f.readlines()
    if not lines:
        print('Error: Log file is empty. Exiting.')
        return
    for line in lines:
        if 'Frame' not in line or 'YOLOv8' not in line or 'MobileNet' not in line:
            print(f'Warning: Skipping malformed line: {line.strip()}')
            continue
        try:
            frame = int(line.split('Frame ')[1].split(':')[0])
            yolo_part = line.split('YOLOv8: ')[1].split(', MobileNet: ')[0].strip()
            mobile_part = line.split('MobileNet: ')[2].strip()  # Second MobileNet: split
            # YOLOv8: 1 if handguns detected, 0 if not
            yolo_preds.append(1 if 'handguns' in yolo_part else 0)
            # MobileNet: Class index
            mobile_tokens = mobile_part.split(' ')
            if not mobile_tokens or len(mobile_tokens) < 2:
                print(f'Warning: Invalid MobileNet format in line: {line.strip()}')
                continue
            mobile_class = mobile_tokens[0]
            if mobile_class not in class_names:
                print(f'Warning: Unrecognized class \'{mobile_class}\' in line: {line.strip()}')
                continue
            mobile_preds.append(class_names.index(mobile_class))
            frames.append(frame)
        except Exception as e:
            print(f'Warning: Failed to parse line: {line.strip()} - Error: {e}')
            continue
    if not frames:
        print('Error: No valid data found in log file. Exiting.')
        return
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
