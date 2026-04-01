from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import yaml

def evaluate(model_path='runs/train/falconeye_v1/weights/best.pt', 
             data_path='dataset.yaml'):
    
    # Load class names
    with open(data_path) as f:
        data = yaml.safe_load(f)
    class_names = data['names']
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # Run evaluation on test set
    print("Running evaluation on test set...")
    metrics = model.val(
        data=data_path,
        split='test',
        imgsz=640,
        batch=16,
        device=0,
        save_json=True,
        plots=True,
    )

    # Print results
    print("\n===== EVALUATION RESULTS =====")
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")

    # Per-class results
    print("\n===== PER-CLASS RESULTS =====")
    for i, name in enumerate(class_names.values()):
        ap = metrics.box.ap50[i]
        print(f"  {name:<25} AP@0.5: {ap:.4f}")

    # Confusion matrix
    print("\nGenerating confusion matrix...")
    os.makedirs('results', exist_ok=True)
    conf_matrix = metrics.confusion_matrix.matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='.0f',
        cmap='Blues',
        xticklabels=list(class_names.values()) + ['background'],
        yticklabels=list(class_names.values()) + ['background'],
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=150)
    print("Confusion matrix saved to results/confusion_matrix.png")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")

if __name__ == '__main__':
    evaluate()