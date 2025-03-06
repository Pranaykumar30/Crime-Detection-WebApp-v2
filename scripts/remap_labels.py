import os

def remap_labels(label_dir):
    class_map = {
        46: 0,  # COCO 'gun' -> handguns
        43: 1,  # COCO 'knife' -> knives
        0: 5    # COCO 'person' -> normal-behavior (default)
    }
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        if not os.path.isfile(label_path):
            continue
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Determine class from filename
        fname_lower = label_file.lower()
        if 'handgun' in fname_lower:
            fname_class = 0
        elif 'knife' in fname_lower:
            fname_class = 1
        elif 'sharp' in fname_lower:
            fname_class = 2
        elif 'masked' in fname_lower:
            fname_class = 3
        elif 'violence' in fname_lower:
            fname_class = 4
        else:
            fname_class = 5

        with open(label_path, 'w') as f:
            if not lines:
                f.write(f'{fname_class} 0.5 0.5 0.2 0.2')
                print(f'Empty label for {label_file} - assigned class {fname_class}')
                continue
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                old_class = int(parts[0])
                coco_class = class_map.get(old_class, 5)  # Default to 5 if unmapped
                # Prioritize filename over COCO if mismatched
                new_class = fname_class if coco_class != fname_class else coco_class
                f.write(f"{new_class} {' '.join(parts[1:])}")
            print(f'Remapped {label_file} to class {new_class}')

if __name__ == '__main__':
    remap_labels('data/labels')
