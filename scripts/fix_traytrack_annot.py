import os
import json
from PIL import Image


def get_img_size(image_path):
    with Image.open(image_path) as img:
        return img.size # (width, height)


def verify_img_size(annotations, images_dir):
    for img_info in annotations['images']:
        img_path = os.path.join(images_dir, img_info['file_name'])
        if os.path.exists(img_path):
            actual_width, actual_height = get_img_size(img_path)
            img_info['width'], img_info['height'] = actual_width, actual_height


if __name__ == '__main__':
    ppath = os.environ.get("PROJECT_PATH", "")
    dataset_root = f'{ppath}/datasets/TrayTrack'
    with open(f'{dataset_root}/annotations.json') as f:
        data = json.load(f)
    
    verify_img_size(data, f'{dataset_root}/images')
    
    with open(f'{dataset_root}/fixed_annotations.json', 'w') as f:
        json.dump(data, f)
