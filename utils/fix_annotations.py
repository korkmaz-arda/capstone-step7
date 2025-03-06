import json
import os
import cv2


def correct_image_sizes(images_dir, annotations_path):
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    image_annotations = {img['file_name']: img for img in data['images']}

    corrections = 0

    for image_name in os.listdir(images_dir):
        if 'cropped' in image_name:
            image_path = os.path.join(images_dir, image_name)

            image = cv2.imread(image_path)
            if image is None:
                continue

            actual_height, actual_width = image.shape[:2]

            if image_name in image_annotations:
                annotation = image_annotations[image_name]

                if annotation['width'] != actual_width or annotation['height'] != actual_height:
                    annotation['width'] = actual_width
                    annotation['height'] = actual_height
                    corrections += 1
            else:
                continue

    if corrections > 0:
        with open(annotations_path, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        print("No corrections were necessary.")


if __name__ == "__main__":
    root_dir = ""
    images_dir = f'{root_dir}/datasets/tray-dataset/images'
    annotations_path = f'{root_dir}/datasets/tray-dataset/annotations.json'

    correct_image_sizes(images_dir, annotations_path)
