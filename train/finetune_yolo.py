from ultralytics import YOLO

model = YOLO('../models/yolo11n.pt')

model.train(
    data='../datasets/configs/simple_food_det.yaml',
    epochs=120,            
    imgsz=640
    batch=64, # 16 for RTX 3090
    device=0, # using 1 GPU
)

print("Training complete. Model and weights saved in the 'runs/train/' directory.")
