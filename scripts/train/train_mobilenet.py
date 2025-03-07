import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
import numpy as np

def load_data(data_dir):
    images, labels = [], []
    class_names = ['handguns', 'knives', 'sharp-edged-weapons', 'masked-intruders', 'violence', 'normal-behavior']
    for split in ['train', 'val']:
        img_dir = os.path.join(data_dir, split, 'images')
        lbl_dir = os.path.join(data_dir, split, 'labels')
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
    return np.array(images), tf.keras.utils.to_categorical(labels, num_classes=6)

def train_mobilenet():
    train_images, train_labels = load_data('data/split')
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(6, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10, batch_size=8, validation_split=0.2)
    os.makedirs('models/mobilenet', exist_ok=True)
    model.save('models/mobilenet/mobilenet_custom.h5')

if __name__ == '__main__':
    train_mobilenet()
