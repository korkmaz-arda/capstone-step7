#!/bin/bash
export PROJECT_PATH="$(pwd)"
echo "PROJECT_PATH: $PROJECT_PATH"

# sudo apt-get update
# sudo apt-get install ffmpeg
# sudo apt-get install libgl1-mesa-glx
# python3 -m pip install -r requirements.txt

# mkdir -p "$PROJECT_PATH/datasets"
# mkdir -p "$PROJECT_PATH/.api-keys"
# chmod 600 "$PROJECT_PATH/.api-keys/"

# echo "Downloading: UECFOOD100 Dataset"
# wget --progress=bar:force "http://foodcam.mobi/dataset100.zip" -O "$PROJECT_PATH/datasets/uecfood100.zip"
# unzip "$PROJECT_PATH/datasets/uecfood100.zip" -d "$PROJECT_PATH/datasets/"
# rm "$PROJECT_PATH/datasets/uecfood100.zip"
# echo "Converting UECFOOD100 to YOLO format"
# python3 scripts/format_uecfood100.py

# OR:
# export KAGGLE_CONFIG_DIR="$PROJECT_PATH/api_keys/"
# echo "Downloading: UECFOOD100 (Kaggle)"
# kaggle datasets download -d rkuo2000/uecfood100 -p "$PROJECT_PATH/datasets/"
# unzip "$PROJECT_PATH/datasets/uecfood100.zip" -d "$PROJECT_PATH/datasets/"
# rm "$PROJECT_PATH/datasets/uecfood100.zip"

# echo "Downloading: VIPER-FoodNet (VFN) Dataset"
# wget --no-check-certificate --progress=bar:force "https://lorenz.ecn.purdue.edu/~vfn/vfn_1_0.zip" -O "$PROJECT_PATH/datasets/vfn.zip"
# unzip "$PROJECT_PATH/datasets/vfn.zip" -d "$PROJECT_PATH/datasets/"
# rm "$PROJECT_PATH/datasets/vfn.zip"
# mv "$PROJECT_PATH/datasets/vfn_1_0" "$PROJECT_PATH/datasets/VFN"
# echo "Converting VFN to YOLO format"
# python3 scripts/format_vfn.py

# export ROBOFLOW_API_KEY=$(cat "$PROJECT_PATH/api-keys/roboflow.txt")
# echo "Downloading Food Detection Image Dataset"
# python3 scripts/dl_roboflow_dataset.py
# mv "$PROJECT_PATH/Food-Detection-8" "$PROJECT_PATH/datasets/FoodDetection"

# echo "Downloading TrayTrack Dataset"
# wget --progress=bar:force https://github.com/korkmaz-arda/UCSD-MLE-Bootcamp-Captstone-Datasets/raw/refs/heads/main/tray-track/tray-track.zip -O "$PROJECT_PATH/datasets/tray-track.zip"
# mkdir "$PROJECT_PATH/datasets/TrayTrack"
# unzip "$PROJECT_PATH/datasets/tray-track.zip" -d "$PROJECT_PATH/datasets/TrayTrack"
# rm "$PROJECT_PATH/datasets/tray-track.zip"
# cp "$PROJECT_PATH/configs/tray_track.yaml" "$PROJECT_PATH/datasets/TrayTrack/tray_track.yaml"
# echo "Converting TrayTrack to YOLO format"
# python3 "$PROJECT_PATH/scripts/fix_traytrack_annot.py"
# mv -f "$PROJECT_PATH/datasets/TrayTrack/fixed_annotations.json" "$PROJECT_PATH/datasets/TrayTrack/annotations.json"
# python3 "$PROJECT_PATH/scripts/format_traytrack.py"
# for f in $PROJECT_PATH/datasets/TrayTrack/images/*.jpeg; do mv -- "$f" "${f%.jpeg}.jpg"; done