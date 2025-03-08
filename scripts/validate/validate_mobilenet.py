import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time

def validate_mobilenet(model_path, data_dir):
    model = tf.keras.models.load_model(model_path)
    images, true_labels = [], []
    class_names = ['handguns', 'knives', 'sharp-edged-weapons', 'masked-intruders', 'violence', 'normal-behavior']
    img_dir = os.path.join(data_dir, 'test', 'images')
    lbl_dir = os.path.join(data_dir, 'test', 'labels')
    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        lbl_file = img_file.replace('.jpg', '.txt')
        lbl_path = os.path.join(lbl_dir, lbl_file)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        with open(lbl_path, 'r') as f:
            class_id = int(f.readline().split()[0])
        images.append(img)
        true_labels.append(class_id)
    images = np.array(images)
    true_labels = np.array(true_labels)  # Fixed: Convert to NumPy array
    true_labels_one_hot = tf.keras.utils.to_categorical(true_labels, num_classes=6)
    start_time = time.time()
    predictions = model.predict(images, verbose=0)
    inference_time = (time.time() - start_time) / len(images)  # Avg time per image
    pred_labels = np.argmax(predictions, axis=1)
    loss, acc = model.evaluate(images, true_labels_one_hot, verbose=0)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(true_labels, pred_labels).tolist()  # Convert to list for JSON
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        class_mask = true_labels == i
        if class_mask.sum() > 0:
            per_class_acc[class_name] = float(np.mean(pred_labels[class_mask] == true_labels[class_mask]))
        else:
            per_class_acc[class_name] = 0.0
    results = {
        'accuracy': acc,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': conf_matrix,
        'per_class_accuracy': per_class_acc,
        'inference_time_per_image': inference_time
    }
    print(f'MobileNet Detailed Validation Metrics: {results}')
    with open('models/mobilenet_validation_results.txt', 'w') as f:
        f.write(f'MobileNet Detailed Metrics: {results}')
    return results

if __name__ == '__main__':
    validate_mobilenet('models/mobilenet/mobilenet_custom.h5', 'data/split')
