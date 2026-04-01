from ultralytics import YOLO
import os
import yaml
import argparse

# Optimization config - tweak these based on first training results
EPOCHS = 100
MOSAIC = 0.5
OPTIMIZER = 'AdamW'
MOMENTUM = 0.9
LR0 = 0.0001
LRF = 0.00001
PATIENCE = 20  # early stopping

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--mosaic', type=float, default=MOSAIC)
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER)
    parser.add_argument('--lr0', type=float, default=LR0)
    parser.add_argument('--lrf', type=float, default=LRF)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    args = parser.parse_args()

    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)

    # Load best model from first training run
    base_model = os.path.join(this_dir, 'runs', 'detect', 'train', 'weights', 'best.pt')
    
    if not os.path.exists(base_model):
        print("ERROR: best.pt not found at:", base_model)
        print("Make sure first training run is complete before optimizing.")
        exit()

    print(f"Loading base model from: {base_model}")
    model = YOLO(base_model)

    print("Starting optimization training run...")
    results = model.train(
        data=os.path.join(this_dir, 'yolo_params.yaml'),
        epochs=args.epochs,
        device=0,
        mosaic=args.mosaic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        patience=args.patience,
        augment=True,
        fliplr=0.5,
        flipud=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        project='runs/detect',
        name='optimized'
    )

    print("\n===== OPTIMIZATION COMPLETE =====")
    print(f"Best mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print("Optimized model saved to: runs/detect/optimized/weights/best.pt")
