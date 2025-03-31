import os
import shutil
import re
import random
from pathlib import Path


def split_dataset(images_dir, labels_dir, output_dir, val_ratio=0.2, seed=42):
    random.seed(seed)
    
    train_images_dir = Path(output_dir) / 'train' / 'images'
    val_images_dir = Path(output_dir) / 'val' / 'images'
    train_labels_dir = Path(output_dir) / 'train' / 'labels'
    val_labels_dir = Path(output_dir) / 'val' / 'labels'

    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(Path(images_dir).glob('*.jpg'))
    base_image_groups = {}
    for img_path in image_files:
        base_name = re.match(r'(.+?-\d+)', img_path.stem)
        if base_name:
            base_name = base_name.group(1)
            if base_name not in base_image_groups:
                base_image_groups[base_name] = []
            base_image_groups[base_name].append(img_path)
    
    base_image_keys = list(base_image_groups.keys())
    random.shuffle(base_image_keys)
    val_count = int(len(base_image_keys) * val_ratio)
    
    val_base_images = set(base_image_keys[:val_count])
    train_base_images = set(base_image_keys[val_count:])
    
    def copy_images_and_labels(base_images, target_images_dir, target_labels_dir):
        for base_name in base_images:
            for img_path in base_image_groups[base_name]:
                target_img_path = target_images_dir / img_path.name
                shutil.copy(img_path, target_img_path)
                
                label_file = Path(labels_dir) / f"{img_path.stem}.txt"
                if label_file.exists():
                    target_label_path = target_labels_dir / label_file.name
                    shutil.copy(label_file, target_label_path)
    
    copy_images_and_labels(train_base_images, train_images_dir, train_labels_dir)
    copy_images_and_labels(val_base_images, val_images_dir, val_labels_dir)
    
    print(f"Dataset split completed. Train and validation sets are saved in '{output_dir}'.")


if __name__ == "__main__":
    ppath = os.environ.get("PROJECT_PATH", "")
    split_dataset(
        images_dir=f"{ppath}/datasets/TrayTrack/images",
        labels_dir=f"{ppath}/datasets/TrayTrack/labels",
        output_dir=f"{ppath}/datasets/TrayTrack-splits"
    )