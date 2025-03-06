import cv2
import numpy as np
import albumentations as A
import os
from skimage import exposure, filters

def preprocess_images(image_dir, output_dir):
    """Preprocess images with augmentations, edge enhancement, and normalization."""
    os.makedirs(output_dir, exist_ok=True)

    augment = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomCrop(height=640, width=640, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.ToGray(p=0.2),
        A.GaussNoise(std_range=(0.1, 0.3), mean_range=(0.0, 0.0), per_channel=True, p=0.3),  # ✅ Fixed GaussNoise
        A.MotionBlur(blur_limit=5, p=0.2),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),
        A.RandomFog(alpha_coef=0.1, p=0.2)  # ✅ Corrected fog_coef_range
    ])

    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping {img_file} - invalid image")
            continue

        img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
        img_normalized = img_resized / 255.0  # Normalize pixel values
        augmented = augment(image=(img_normalized * 255).astype(np.uint8))
        img_aug = augmented['image']

        # Convert to grayscale and detect edges
        img_gray = cv2.cvtColor(img_aug, cv2.COLOR_BGR2GRAY)
        edges = filters.sobel(img_gray)
        img_edges = cv2.convertScaleAbs(edges * 255)

        # Merge edge-detection with the original image
        img_aug = cv2.addWeighted(img_aug, 0.8, cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR), 0.2, 0)

        # Adjust intensity for better contrast
        img_aug = exposure.rescale_intensity(img_aug, in_range='image', out_range=(0, 255)).astype(np.uint8)

        # Save processed image
        cv2.imwrite(os.path.join(output_dir, img_file), img_aug)

if __name__ == '__main__':
    for split in ['train', 'val', 'test']:
        input_dir = f'data/split/{split}/images'
        preprocess_images(input_dir, input_dir)
        print(f'Preprocessed {split} images')
