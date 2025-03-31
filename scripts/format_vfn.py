import os
import shutil
from PIL import Image, ImageOps


def load_img_ids(file_path):
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f)


def conv2yolo(x1, y1, x2, y2, img_width, img_height):
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height


def process_annotations(annotation_file, split_ids, split, images_dir, base_dir):
    print(f"Processing {split}")
    with open(annotation_file, 'r') as f:
        for line_count, line in enumerate(f, 1):
            image_id, x1, y1, x2, y2, category_id = line.strip().split()
            x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

            x1, y1 = y1, x1
            x2, y2 = y2, x2

            if image_id not in split_ids:
                continue

            src_image_path = os.path.join(images_dir, category_id, image_id)
            dest_image_path = os.path.join(base_dir, split, 'images', image_id)
            label_path = os.path.join(base_dir, split, 'labels', f"{os.path.splitext(image_id)[0]}.txt")

            shutil.copyfile(src_image_path, dest_image_path)

            with open(dest_image_path, 'rb') as img_file:
                with Image.open(img_file) as img:
                    img = ImageOps.exif_transpose(img)
                    img_width, img_height = img.size

            x_center, y_center, width, height = conv2yolo(x1, y1, x2, y2, img_width, img_height)

            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            width = min(max(width, 0), 1)
            height = min(max(height, 0), 1)

            with open(label_path, 'a') as label_file:
                label_file.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

            if line_count % 100 == 0:
                print(f"Processed {line_count} annot. for {split}")

    print(f"Completed processing {split}")


if __name__ == "__main__":
    ppath = os.environ.get("PROJECT_PATH", "")
    base_dir = f'{ppath}/datasets/vfn-yolo'
    vfn_dir = f'{ppath}/datasets/VFN'

    images_dir = os.path.join(vfn_dir, 'Images')
    meta_dir = os.path.join(vfn_dir, 'Meta')

    os.makedirs(os.path.join(base_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test', 'labels'), exist_ok=True)

    print("dir structure created.")

    train_ids = load_img_ids(os.path.join(meta_dir, 'training.txt'))
    val_ids = load_img_ids(os.path.join(meta_dir, 'validation.txt'))
    test_ids = load_img_ids(os.path.join(meta_dir, 'testing.txt'))

    print("img ids loaded")

    category_mapping = {}
    with open(os.path.join(meta_dir, 'category_ids.txt'), 'r') as f:
        for line in f:
            category_id, category_name = line.strip().split()
            category_mapping[category_id] = category_name

    annotation_file = os.path.join(meta_dir, 'annotations.txt')
    process_annotations(annotation_file, train_ids, 'train', images_dir, base_dir)
    process_annotations(annotation_file, val_ids, 'val', images_dir, base_dir)
    process_annotations(annotation_file, test_ids, 'test', images_dir, base_dir)

    print("VFN to YOLO formatting completed")
