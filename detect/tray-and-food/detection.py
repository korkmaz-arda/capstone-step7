import sys
import shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon


def convert_bbox_to_polygon(bbox):
    x1, y1, x2, y2 = bbox
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def process_video(tray_model, food_model, input_path, output_path,
                  class_filter=None, tray_min_conf=0.72, food_min_conf=0.3, intersection_threshold=0.5):

    # video capture
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', etc.

    # video writer
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # no more frames

        frame_count += 1

        # ============= Tray Detection =============
        tray_results = tray_model.predict(frame, verbose=False)
        tray_polygons = []
        for result in tray_results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                # YOLO box structure: [xyxy, conf, cls]
                conf = float(box.conf[0].cpu().numpy())
                if conf < tray_min_conf:
                    continue  # skip low conf tray detections

                xyxy = box.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2)
                tray_polygon = convert_bbox_to_polygon(xyxy)
                tray_polygons.append(tray_polygon)

                # draw tray box in blue
                x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                tray_label = f"Tray {conf:.2f}"
                cv2.putText(frame, tray_label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # ============= Food Detection =============
        food_results = food_model.predict(frame, verbose=False)
        for result in food_results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2)
                conf = float(box.conf[0].cpu().numpy())
                if conf < food_min_conf:
                    continue  # skip low conf food detections

                cls_id = int(box.cls[0].cpu().numpy())
                class_name = result.names.get(cls_id, "unknown")

                # filter by food classes
                if class_filter and class_name not in class_filter:
                    continue

                food_polygon = convert_bbox_to_polygon(xyxy)

                # check intersection with any tray polygon
                for tray_polygon in tray_polygons:
                    if tray_polygon.is_valid and food_polygon.is_valid:
                        intersection_area = tray_polygon.intersection(food_polygon).area
                        food_area = food_polygon.area
                        
                        if food_area > 0:
                            ratio = intersection_area / food_area
                            
                            if ratio >= intersection_threshold:
                                x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                                
                                # draw food box in green
                                cv2.rectangle(frame, (x1, y1), (x2, y2),
                                              color=(0, 255, 0), thickness=2)
                                
                                # label: class name + confidence
                                label = f"{class_name} {conf:.2f}"
                                cv2.putText(frame, label, (x1, y1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved to: {output_path}\n")


if __name__ == "__main__":
    TRAY_MODEL_PATH = "../../models/tray_detector.pt"
    FOOD_MODEL_PATH = "../../models/yolo11n.pt"
    SOURCE_PATH = "../../videos/tray-recordings-1"
    OUTPUT_PATH = "./outputs"

    TRAY_MIN_CONF = 0.72
    FOOD_MIN_CONF = 0.3
    INTERSECTION_THRESHOLD = 0.5

    food_classes = [
        "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake"
    ]

    tray_model_path = Path(TRAY_MODEL_PATH)
    food_model_path = Path(FOOD_MODEL_PATH)
    source_dir = Path(SOURCE_PATH)
    output_dir = Path(OUTPUT_PATH)

    print(f"Loading tray detection model from: {tray_model_path}")
    tray_model = YOLO(str(tray_model_path))

    print(f"Loading food detection model from: {food_model_path}")
    food_model = YOLO(str(food_model_path))

    # get video files
    video_files = list(source_dir.glob("*.mp4")) + \
                  list(source_dir.glob("*.avi")) + \
                  list(source_dir.glob("*.mov")) + \
                  list(source_dir.glob("*.mkv"))

    if not video_files:
        print(f"No video files found in {source_dir}. Supported extensions: .mp4, .avi, .mov, .mkv.")
        sys.exit(0)

    # process each video file
    for video_file in video_files:
        output_file = output_dir / f"{video_file.stem}_annotated{video_file.suffix}"
        print(f"Processing {video_file}")
        process_video(
            tray_model=tray_model,
            food_model=food_model,
            input_path=video_file,
            output_path=output_file,
            class_filter=food_classes,
            tray_min_conf=TRAY_MIN_CONF,
            food_min_conf=FOOD_MIN_CONF,
            intersection_threshold=INTERSECTION_THRESHOLD
        )
    
    shutil.make_archive(f"{OUTPUT_PATH}/detected-videos", 'gztar', OUTPUT_PATH)
