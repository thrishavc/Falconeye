from ultralytics import YOLO
import os

if __name__ == '__main__':
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)

    base_model = os.path.join(this_dir, 'models', 'best.pt')

    if not os.path.exists(base_model):
        print("ERROR: best.pt not found at:", base_model)
        exit()

    print(f"Loading base model: {base_model}")
    model = YOLO(base_model)

    print("Starting optimization run...")
    results = model.train(
        data=os.path.join(this_dir, 'yolo_params.yaml'),
        epochs=100,
        device=0,
        imgsz=640,
        batch=8,
        optimizer='AdamW',
        lr0=0.00005,
        lrf=0.00001,
        momentum=0.9,
        patience=30,
        mosaic=0.8,
        augment=True,
        fliplr=0.5,
        flipud=0.1,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mixup=0.1,
        copy_paste=0.1,
        project='runs/detect',
        name='optimized'
    )

    print("Optimization complete!")
    print(f"Best mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print("Model saved to: runs/detect/optimized/weights/best.pt")
