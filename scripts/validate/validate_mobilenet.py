import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
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
    true_labels = np.array(true_labels)
    true_labels_one_hot = tf.keras.utils.to_categorical(true_labels, num_classes=6)
    
    start_time = time.time()
    predictions = model.predict(images, verbose=0)
    inference_time = (time.time() - start_time) / len(images)
    
    pred_labels = np.argmax(predictions, axis=1)
    loss, acc = model.evaluate(images, true_labels_one_hot, verbose=0)
    
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        class_mask = true_labels == i
        if class_mask.sum() > 0:
            per_class_acc[class_name] = float(np.mean(pred_labels[class_mask] == true_labels[class_mask]))
        else:
            per_class_acc[class_name] = 0.0
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('/workspaces/Crime-Detection-WebApp-v2/models/mobilenet/mobilenet_metrics/confusion_matrix.png')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    for i in range(6):
        fpr, tpr, _ = roc_curve(true_labels_one_hot[:, i], predictions[:, i])
        plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('/workspaces/Crime-Detection-WebApp-v2/models/mobilenet/mobilenet_metrics/roc_curve.png')
    plt.show()
    
    # Metrics Summary Plot
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [acc, precision, recall, f1]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=metrics, y=values, hue=metrics, palette='coolwarm', legend=False)
    plt.ylim(0, 1)
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.savefig('/workspaces/Crime-Detection-WebApp-v2/models/mobilenet/mobilenet_metrics/performance_metrics.png')
    plt.show()
    
    results = {
        'accuracy': acc,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': conf_matrix.tolist(),
        'per_class_accuracy': per_class_acc,
        'inference_time_per_image': inference_time
    }
    
    return results

if __name__ == '__main__':
    validate_mobilenet('models/mobilenet/mobilenet_custom.h5', 'data/split')
