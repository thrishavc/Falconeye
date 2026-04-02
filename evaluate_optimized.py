from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import yaml

def evaluate(model_path='models/optimized/best.pt',
             data_path='yolo_params.yaml'):

    with open(data_path) as f:
        data = yaml.safe_load(f)
    class_names = data['names']

    print(f"Loading optimized model from: {model_path}")
    model = YOLO(model_path)

    print("Running evaluation on test set...")
    metrics = model.val(
        data=data_path,
        split='test',
        imgsz=640,
        batch=8,
        device=0,
        plots=True,
    )

    print("\n===== OPTIMIZED MODEL RESULTS =====")
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")

    print("\n===== PER-CLASS RESULTS =====")
    for i, name in enumerate(class_names):
        ap = metrics.box.ap50[i]
        print(f"  {name:<25} AP@0.5: {ap:.4f}")

    print("\n===== COMPARISON WITH BASELINE =====")
    print(f"Baseline mAP@0.5:  0.7209")
    print(f"Optimized mAP@0.5: {metrics.box.map50:.4f}")
    diff = metrics.box.map50 - 0.7209
    if diff > 0:
        print(f"Improvement:      +{diff:.4f} BETTER")
    else:
        print(f"Change:           {diff:.4f} WORSE")

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
    plt.title('FalconEye - Optimized Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix_optimized.png', dpi=150)
    print("Saved: results/confusion_matrix_optimized.png")

    with open('results/performance_report_optimized.txt', 'w') as f:
        f.write("===== FALCONEYE OPTIMIZED MODEL REPORT =====\n\n")
        f.write(f"mAP@0.5:      {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
        f.write(f"Precision:    {metrics.box.mp:.4f}\n")
        f.write(f"Recall:       {metrics.box.mr:.4f}\n\n")
        f.write("BASELINE COMPARISON:\n")
        f.write(f"Baseline mAP@0.5:  0.7209\n")
        f.write(f"Optimized mAP@0.5: {metrics.box.map50:.4f}\n")
        diff = metrics.box.map50 - 0.7209
        f.write(f"Improvement:       {diff:+.4f}\n\n")
        f.write("PER-CLASS RESULTS:\n")
        for i, name in enumerate(class_names):
            ap = metrics.box.ap50[i]
            f.write(f"  {name:<25} AP@0.5: {ap:.4f}\n")
    print("Saved: results/performance_report_optimized.txt")

if __name__ == '__main__':
    evaluate()
