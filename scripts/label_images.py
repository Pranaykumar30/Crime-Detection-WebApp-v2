from ultralytics import YOLO
import os

def label_images(image_dir, label_dir):
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Ensure directories exist
    os.makedirs(label_dir, exist_ok=True)

    # Class mapping
    class_map = {
        "handgun": 0,
        "knife": 1,
        "sharp": 2,
        "masked": 3,
        "violence": 4
    }
    default_class = 5  # Default: normal behavior

    # Iterate through images
    for img_file in os.listdir(image_dir):
        if not img_file.endswith(".jpg"):
            continue

        img_path = os.path.join(image_dir, img_file)

        if not os.path.exists(img_path):
            print(f"Skipping {img_file} - Image not found")
            continue

        try:
            # Run YOLOv8 inference
            results = model.predict(img_path, save_txt=True, conf=0.05)

            # Define label file paths
            label_file = img_file.replace(".jpg", ".txt")
            src_path = os.path.join("runs/detect/predict/labels", label_file)
            dest_path = os.path.join(label_dir, label_file)

            # Move YOLO labels if available
            if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
                os.rename(src_path, dest_path)
                print(f"Labeled {img_file} with detections")

            else:
                # No YOLO detection → Assign class based on filename keywords
                assigned_class = default_class
                for keyword, class_id in class_map.items():
                    if keyword in img_file.lower():
                        assigned_class = class_id
                        break

                # Save the manually assigned label
                with open(dest_path, "w") as f:
                    f.write(f"{assigned_class} 0.5 0.5 0.2 0.2\n")

                print(f"No detections for {img_file} - Assigned class {assigned_class}")

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    # Clean up YOLO's temporary output
    os.system("rm -rf runs/detect/predict")

if __name__ == "__main__":
    label_images("data/images", "data/labels")
