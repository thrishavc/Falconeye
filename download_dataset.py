import os
os.environ["YOLO_DATASETS_DIR"] = r"C:\Users\Thrisha V C\Desktop\FalconEye\data"

from ultralytics import YOLO, settings

# Tell Ultralytics to store datasets inside our project folder
settings.update({"datasets_dir": r"C:\Users\Thrisha V C\Desktop\FalconEye\data"})

print("Downloading COCO128 sample dataset...")

model = YOLO("yolov8n.pt")
results = model.train(
    data="coco128.yaml",
    epochs=1,
    imgsz=640,
    batch=4,
    device=0,
    project=r"C:\Users\Thrisha V C\Desktop\FalconEye\runs",
    name="test_run"
)

print("Done! COCO128 dataset is ready and 1 epoch test run completed.")
