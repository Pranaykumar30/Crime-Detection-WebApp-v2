import tensorflow as tf
import numpy as np
import os

def validate_mobilenet(model_path, data_dir):
    model = tf.keras.models.load_model(model_path)
    images, labels = [], []
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
        labels.append(class_id)
    images = np.array(images)
    labels = tf.keras.utils.to_categorical(labels, num_classes=6)
    loss, acc = model.evaluate(images, labels, verbose=0)
    results = {'accuracy': acc}
    print(f'MobileNet Validation Metrics: {results}')
    with open('models/mobilenet_validation_results.txt', 'w') as f:
        f.write(f'MobileNet: {results}')
    return results

if __name__ == '__main__':
    validate_mobilenet('models/mobilenet/mobilenet_custom.h5', 'data/split')
