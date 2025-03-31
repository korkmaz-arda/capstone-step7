import os
from roboflow import Roboflow

api_key = os.getenv("ROBOFLOW_API_KEY")

if api_key is None:
    print("Error: API key is not set.")
    exit(1)

rf = Roboflow(api_key=api_key)
project = rf.workspace("food-hofna").project("food-detection-fme3o")
version = project.version(8)

dataset = version.download("yolov8")