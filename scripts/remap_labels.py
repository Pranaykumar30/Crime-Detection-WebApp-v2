import os

def remap_labels(label_dir):
    """
    Remaps object detection labels to custom class IDs based on predefined mappings.
    If no valid mapping is found, assigns a default class based on filename keywords.

    Class Mapping:
        0 - handguns
        1 - knives
        2 - sharp-edged-weapons
        3 - masked-intruders
        4 - violence
        5 - normal-behavior (default)
    """

    # COCO class mappings (original dataset classes to custom classes)
    class_map = {
        46: 0,  # COCO 'gun' -> handguns
        43: 1,  # COCO 'knife' -> knives
        0: 3    # COCO 'person' -> masked-intruders
    }
    
    # Default label mapping based on filename keywords
    keyword_map = {
        "handgun": 0,
        "knife": 1,
        "sharp": 2,
        "masked": 3,
        "violence": 4
    }
    
    default_class = 5  # Normal behavior (default class)

    # Process all label files
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        if not os.path.isfile(label_path):
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            # If label file is empty, assign class based on filename keywords
            assigned_class = default_class
            for keyword, class_id in keyword_map.items():
                if keyword in label_file.lower():
                    assigned_class = class_id
                    break

            with open(label_path, "w") as f:
                f.write(f"{assigned_class} 0.5 0.5 0.2 0.2\n")
            print(f"Empty label for {label_file} - Assigned class {assigned_class}")
            continue

        # Update label file with remapped classes
        updated_lines = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            
            old_class = int(parts[0])  # Original COCO class
            
            # Apply class mapping
            new_class = class_map.get(old_class, default_class)

            # If old class is 0 (person), refine classification based on filename
            if old_class == 0:
                for keyword, class_id in keyword_map.items():
                    if keyword in label_file.lower():
                        new_class = class_id
                        break

            updated_lines.append(f"{new_class} {' '.join(parts[1:])}\n")

        # Write updated labels back to file
        with open(label_path, "w") as f:
            f.writelines(updated_lines)

if __name__ == "__main__":
    remap_labels("data/labels")
