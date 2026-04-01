from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import yaml


def evaluate(model_path='models/best.pt',
             data_path='yolo_params.yaml'):

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
        batch=8,
        device=0,
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
    for i, name in enumerate(class_names):
        ap = metrics.box.ap50[i]
        print(f"  {name:<25} AP@0.5: {ap:.4f}")

    # Save confusion matrix
    print("\nGenerating confusion matrix...")
    os.makedirs('results', exist_ok=True)
    conf_matrix = metrics.confusion_matrix.matrix

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='.0f',
        cmap='Blues',
        xticklabels=list(class_names) + ['background'],
        yticklabels=list(class_names) + ['background'],
    )
    plt.title('FalconEye - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=150)
    print("Saved: results/confusion_matrix.png")

    # Save mAP summary
    with open('results/performance_report.txt', 'w') as f:
        f.write("===== FALCONEYE PERFORMANCE REPORT =====\n\n")
        f.write(f"mAP@0.5:      {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
        f.write(f"Precision:    {metrics.box.mp:.4f}\n")
        f.write(f"Recall:       {metrics.box.mr:.4f}\n\n")
        f.write("PER-CLASS RESULTS:\n")
        for i, name in enumerate(class_names):
            ap = metrics.box.ap50[i]
            f.write(f"  {name:<25} AP@0.5: {ap:.4f}\n")
    print("Saved: results/performance_report.txt")
    print(f"\nFinal mAP@0.5: {metrics.box.map50:.4f}")


if __name__ == '__main__':
    evaluate()