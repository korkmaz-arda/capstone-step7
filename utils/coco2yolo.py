import json
import os


def convert_coco_to_yolo(coco_json_path, images_dir, output_dir):
    with open(coco_json_path) as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)

    categories = {category['id']: idx for idx, category in enumerate(data['categories'])}

    for annotation in data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']

        x, y, w, h = bbox
        x_center = x + w / 2
        y_center = y + h / 2

        image_info = next(item for item in data['images'] if item["id"] == image_id)
        image_w, image_h = image_info['width'], image_info['height']
        x_center /= image_w
        y_center /= image_h
        w /= image_w
        h /= image_h

        label_path = os.path.join(output_dir, f"{image_info['file_name'].split('.')[0]}.txt")

        with open(label_path, 'a') as f:
            f.write(f"{categories[category_id]} {x_center} {y_center} {w} {h}\n")


if __name__ == "__main__":
    root_dir = ""
    coco_json_path = f'{root_dir}/datasets/tray-dataset/annotations.json'
    images_dir = f'{root_dir}/datasets/tray-dataset/images'
    output_dir = f'{root_dir}/datasets/tray-dataset/labels'

    convert_coco_to_yolo(coco_json_path, images_dir, output_dir)
