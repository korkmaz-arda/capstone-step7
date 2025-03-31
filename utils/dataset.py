import os
import yaml
import json
import shutil

from pathlib import Path
from utils.bbox import yolo2poly


def load_dataset_labels(dataset_path):
    yaml_filename = next(f for f in os.listdir(dataset_path) if f.endswith('.yaml'))
    yaml_file = os.path.join(dataset_path, yaml_filename)

    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    
    dataset_classes = config.get('names', [])
    print(f"dataset_classes: {dataset_classes}")
    return {idx: cls for idx, cls in dataset_classes.items()}


def load_ground_truth(
        dataset_path, 
        split_dir: Path
    ):
    """
    OUTPUT FORMAT:
    [
        {"img_path1": [{"class": ":cls_name1", "bbox": [x1, y1, w1, h1]}, {"class": ":cls_name2", "bbox": [x2, y2, w2, h2]}]},
        {"img_path2": [{"class": ":cls_name3", "bbox": [x3, y3, w3, h3]}]}, ...
        ...
    ]
    """

    dataset_label_map = load_dataset_labels(dataset_path)
    labels_path = split_dir / "labels"
    ground_truth = []
    
    label_files = sorted(labels_path.glob("*.txt"))
    for label_file in label_files:
        image_ground_truth = []
        
        is_empty = os.stat(label_file).st_size == 0
        if not is_empty:
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    
                    image_ground_truth.append({
                        "class": dataset_label_map[cls_id],
                        "bbox": yolo2poly([cx, cy, w, h])
                    })

        ground_truth.append({f"{_get_img_file(label_file)}": image_ground_truth})
    
    return ground_truth

def _get_img_file(label_file):
    return Path(str(label_file).replace('/labels/', '/images/').replace('.txt', '.jpg'))


def filter_yolo_dataset(
        input_path, 
        output_path, 
        config_file,
        cls_filer=[],
        cls_name_map={}, # key (dataset-class) -> value (yolo-class)
    ):
    if not os.path.exists(output_path):
        print("input:", input_path)
        shutil.copytree(input_path, output_path)

    config_dest = os.path.join(output_path, os.path.basename(config_file))
    shutil.copy2(config_file, config_dest)

    with open(config_dest, 'r') as file:
        config_content = file.read()

    updated_config_content = config_content.replace(input_path, output_path)

    with open(config_dest, 'w') as file:
        file.write(updated_config_content)

    with open(config_dest, 'r') as file:
        config = yaml.safe_load(file)

    cls_names = config.get('names', [])
    print("cls_names:", cls_names)

    filtered_ids = [
        id
        for id, cls_name in cls_names.items() 
        if cls_name_map.get(cls_name, cls_name) in cls_filer
    ]

    filtered_classes = [cls_names[id] for id in filtered_ids]
    print(filtered_classes)
    
    splits = ['train', 'val', 'test']
    for split in splits:
        labels_path = os.path.join(output_path, split, 'labels')
        if not os.path.exists(labels_path):
            continue

        for label_file in os.listdir(labels_path):
            if label_file.endswith('.txt'):
                file_path = os.path.join(labels_path, label_file)
                
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                filtered_lines = [
                    line for line in lines
                    if int(line.split()[0]) in filtered_ids
                ]

                with open(file_path, 'w') as file:
                    file.writelines(filtered_lines)


def detect_format(dataset_path):
    output_msg = ""
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.json', '.xml', '.txt')):
                ann_file = os.path.join(root, file)
                print(f"Analyzing: {ann_file}")
                
                if file.endswith('.json'):
                    with open(ann_file) as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        if 'images' in data and 'annotations' in data:
                            output_msg = "COCO format"
                        elif 'boxes' in data or 'bbox' in data:
                            output_msg = "Custom JSON with bbox format"
                        output_msg = "Unknown JSON format"
                
                elif file.endswith('.xml'):
                    if "<annotation>" in open(ann_file).read(500):
                        output_msg = "Pascal VOC XML format"
                    output_msg = "Unknown XML format"
                
                elif file.endswith('.txt'):
                    first_line = open(ann_file).readline()
                    parts = first_line.strip().split()
                    if len(parts) == 5:  # class x y w h
                        output_msg = "YOLO-like format (but you said it's not YOLO)"
                    elif len(parts) == 4:  # x1 y1 x2 y2
                        output_msg = "Simple coordinates format"
                
                output_msg = f"Unknown format (file: {file})"
    
    output_msg = "No annotation files found"
    print(f"Detected format: {output_msg}")