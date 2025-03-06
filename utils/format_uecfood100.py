import os
import shutil
import random
from PIL import Image


root_dir = ""
source_dir = f"{root_dir}/datasets/UECFOOD100/"
target_dir = f"{root_dir}/datasets/uecfood100-yolo"

train_dir = os.path.join(target_dir, 'train')
val_dir = os.path.join(target_dir, 'val')

os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)

category_map = {}
with open(os.path.join(source_dir, 'category.txt'), 'r') as f:
    for line in f.readlines()[1:]:
        parts = line.strip().split()
        if len(parts) == 2:
            category_map[int(parts[0])] = parts[1]


def split_dataset(ratio=0.8):
    class_image_map = {}

    for i in range(1, 101):
        img_dir = os.path.join(source_dir, str(i))
        bb_info_path = os.path.join(img_dir, 'bb_info.txt')

        if not os.path.exists(bb_info_path):
            continue

        with open(bb_info_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                
                if len(parts) < 5 or not parts[1].isdigit():
                    continue
                
                img_name = parts[0]
                img_path = os.path.join(img_dir, f"{img_name}.jpg")

                if img_name not in class_image_map:
                    class_image_map[img_name] = []
                class_image_map[img_name].append(i - 1)

    train_images = set()
    val_images = set()
    
    for img_name in class_image_map.keys():
        if random.random() < ratio:
            train_images.add(img_name)
        else:
            val_images.add(img_name)

    return train_images, val_images

train_images, val_images = split_dataset(ratio=0.8)

for i in range(1, 101):
    img_dir = os.path.join(source_dir, str(i))

    bb_info_path = os.path.join(img_dir, 'bb_info.txt')

    if not os.path.exists(bb_info_path):
        continue

    with open(bb_info_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            
            if len(parts) < 5 or not parts[1].isdigit():
                continue
            
            img_name = parts[0]
            img_path = os.path.join(img_dir, f"{img_name}.jpg")

            class_id = i - 1
            x_min, y_min, x_max, y_max = map(int, parts[1:5])

            if not os.path.exists(img_path):
                continue
            
            with Image.open(img_path) as img:
                img_width, img_height = img.size

            box_center_x = (x_min + x_max) / 2 / img_width
            box_center_y = (y_min + y_max) / 2 / img_height
            box_width = (x_max - x_min) / img_width
            box_height = (y_max - y_min) / img_height
            
            dest_img_path = os.path.join(train_dir if img_name in train_images else val_dir, 'images', f"{img_name}.jpg")
            if not os.path.exists(dest_img_path):
                shutil.copy(img_path, dest_img_path)
            
            label_filename = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(train_dir if img_name in train_images else val_dir, 'labels', label_filename)
            
            write_mode = 'a' if os.path.exists(label_path) else 'w'
            with open(label_path, write_mode) as label_file:
                label_file.write(f"{class_id} {box_center_x:.6f} {box_center_y:.6f} {box_width:.6f} {box_height:.6f}\n")

print("Conversion to YOLO format and dataset splitting completed.")
