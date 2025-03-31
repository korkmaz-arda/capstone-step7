import os
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon

from utils.bbox import bbox2poly


def detect_trays(frame, tray_model, tray_min_conf):
    tray_results = tray_model.predict(frame, verbose=False)
    tray_polygons = []
    for result in tray_results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf < tray_min_conf:
                continue
            xyxy = box.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2)
            tray_polygon = bbox2poly(xyxy)
            tray_polygons.append((tray_polygon, xyxy, conf))
    return tray_polygons


def draw_tray_boxes(frame, tray_polygons):
    for tray_polygon, xyxy, conf in tray_polygons:
        x1, y1, x2, y2 = [int(coord) for coord in xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        label = f"Tray {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame


def detect_food(frame, food_model, food_min_conf, class_filter):
    food_results = food_model.predict(frame, verbose=False)
    food_detections = []
    for result in food_results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2)
            conf = float(box.conf[0].cpu().numpy())
            if conf < food_min_conf:
                continue
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = result.names.get(cls_id, "unknown")
            if class_filter and class_name not in class_filter:
                continue
            food_detections.append((xyxy, conf, class_name))
    return food_detections


def draw_food_boxes(frame, food_detections, tray_polygons, intersection_threshold):
    for xyxy, conf, class_name in food_detections:
        food_polygon = bbox2poly(xyxy)
        for tray_polygon, _, _ in tray_polygons:
            if tray_polygon.is_valid and food_polygon.is_valid:
                intersection_area = tray_polygon.intersection(food_polygon).area
                food_area = food_polygon.area
                if food_area > 0 and (intersection_area / food_area) >= intersection_threshold:
                    x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  color=(0, 255, 0), thickness=2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break  # annotate only once per food detection if it matches any tray.
    return frame


def process_video(
        tray_model, 
        food_model, 
        input_path, 
        output_path, 
        only_tray=False,
        class_filter=None, 
        tray_min_conf=0.72, 
        food_min_conf=0.3, 
        intersection_threshold=0.5
    ):

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Tray detection and annotation
        tray_polygons = detect_trays(frame, tray_model, tray_min_conf)
        frame = draw_tray_boxes(frame, tray_polygons)

        if not only_tray:
            # Food detection and annotation (only if not in tray-only mode)
            food_detections = detect_food(frame, food_model, food_min_conf, class_filter)
            frame = draw_food_boxes(frame, food_detections, tray_polygons, intersection_threshold)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved to: {output_path}")


def process_videos_dir(
        tray_model, 
        food_model, 
        source_dir, 
        output_dir, 
        only_tray=False,
        class_filter=None, 
        tray_min_conf=0.72, 
        food_min_conf=0.3, 
        intersection_threshold=0.5
    ):
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"] # supported formats
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(Path(source_dir).glob(ext)))
    if not video_files:
        print(f"No video files found in {source_dir}. Supported extensions: {', '.join(video_extensions)}")
        return

    for video_file in video_files:
        output_file = Path(output_dir) / f"{video_file.stem}_annotated{video_file.suffix}"
        print(f"Processing {video_file}")
        process_video(tray_model, food_model, video_file, output_file, only_tray,
                           class_filter, tray_min_conf, food_min_conf, intersection_threshold)


def load_models(tray_model_path, food_model_path):
    print(f"Loading tray detection model from: {tray_model_path}")
    tray_model = YOLO(str(tray_model_path))
    print(f"Loading food detection model from: {food_model_path}")
    food_model = YOLO(str(food_model_path))
    return tray_model, food_model


def detection_test(
        only_tray=False, 
        archive=True,
        tray_model_path=None,
        food_model_path=None,
        source_dir=None,
        output_dir=None
    ):
    ppath = os.environ.get("ppath", "")
    if tray_model_path == None: 
        tray_model_path = Path(ppath) / "models" / "tray_detector.pt"
    if food_model_path == None:
        food_model_path = Path(ppath) / "models" / "yolo11n.pt"
    if source_dir == None:
        source_dir = Path(ppath) / "input-vids"
    if output_dir == None:
        output_dir = Path(ppath) / "output-vids"

    tray_min_conf = 0.77
    food_min_conf = 0.3
    intersection_threshold = 0.5

    food_classes = [
        "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake"
    ]

    tray_model, food_model = load_models(tray_model_path, food_model_path)

    process_videos_dir(tray_model, food_model, source_dir, output_dir, only_tray,
                            food_classes, tray_min_conf, food_min_conf, intersection_threshold)

    if archive:
        shutil.make_archive(f"{output_dir}/detected-videos", 'gztar', output_dir)

