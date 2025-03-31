import os
import shutil
import yaml


YOLO_FOOD_CLS = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]


def filter_yolo_dataset(
        input_path, 
        output_path, 
        config_file,
        yolo_cls_map={}, # key (dataset-class) -> value (yolo-class)
        class_filter=YOLO_FOOD_CLS
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


    class_names = config.get('names', [])
    print("class_names:", class_names)


    filtered_ids = [
        id
        for id, cls_name in class_names.items() 
        if yolo_cls_map.get(cls_name, cls_name) in class_filter
    ]


    filtered_classes = [class_names[id] for id in filtered_ids]
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




if __name__ == "__main__":
    uecfood100_data = [
        "/root/workspace/s7/datasets/uecfood100-yolo",
        "/root/workspace/s7/datasets/uecfood100-yolo-filtered",
        "//root/workspace/s7/configs/uecfood100_yolo.yaml"
    ]
    UECFOOD_2YOLO_MAP = {
        "sandwiches": "sandwich"
    }
    filter_yolo_dataset(*uecfood100_data, UECFOOD_2YOLO_MAP)
    print()

    vfn_data = [
        "/root/workspace/s7/datasets/vfn-yolo",
        "/root/workspace/s7/datasets/vfn-yolo-filtered",
        "//root/workspace/s7/configs/vfn_yolo.yaml"
    ]
    VFN_2YOLO_MAP = {
        "bananas": "banana",
        "hot_dog": "hot dog",
        "cheese_sandwiches": "sandwich",
    }
    filter_yolo_dataset(*vfn_data, VFN_2YOLO_MAP)
    print()