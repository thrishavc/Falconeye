from ultralytics import YOLO
import torch

def main():
    # Confirm GPU
    print(f"Using device: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Load pretrained YOLOv8m model
    model = YOLO('yolov8m.pt')

    # Train
    results = model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        patience=20,
        save_period=10,
        device=0,
        workers=4,
        project='runs/train',
        name='falconeye_v1',
        exist_ok=True,
        # Augmentations
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        degrees=45.0,
        perspective=0.001,
        erasing=0.4,
    )

    print(f"Training complete! Best mAP@0.5: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print("Best model saved at: runs/train/falconeye_v1/weights/best.pt")

if __name__ == '__main__':
    main()