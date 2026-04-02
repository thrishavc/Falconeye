import os
from pathlib import Path
from collections import defaultdict

# Class names matching our dataset
CLASS_NAMES = [
    'OxygenTank',
    'NitrogenTank', 
    'FirstAidBox',
    'FireAlarm',
    'SafetySwitchPanel',
    'EmergencyPhone',
    'FireExtinguisher'
]

def analyze_dataset(labels_path):
    labels_dir = Path(labels_path)
    
    if not labels_dir.exists():
        print(f"ERROR: Path not found: {labels_path}")
        return
    
    # Count images per class and total annotations per class
    class_image_count = defaultdict(int)
    class_annotation_count = defaultdict(int)
    total_images = 0
    empty_images = 0

    for label_file in labels_dir.glob('*.txt'):
        total_images += 1
        classes_in_image = set()
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            empty_images += 1
            continue
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 1:
                cls_id = int(parts[0])
                class_annotation_count[cls_id] += 1
                classes_in_image.add(cls_id)
        
        for cls_id in classes_in_image:
            class_image_count[cls_id] += 1

    # Print results
    print("=" * 60)
    print("FALCONEYE - TRAINING DATASET CLASS ANALYSIS")
    print("=" * 60)
    print(f"\nTotal training images analyzed: {total_images}")
    print(f"Empty label files: {empty_images}")
    print()
    print(f"{'Class':<25} {'Images':>10} {'Annotations':>12} {'% of Total':>12}")
    print("-" * 60)
    
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        img_count = class_image_count[cls_id]
        ann_count = class_annotation_count[cls_id]
        percentage = (img_count / total_images * 100) if total_images > 0 else 0
        print(f"{cls_name:<25} {img_count:>10} {ann_count:>12} {percentage:>11.1f}%")
    
    print("-" * 60)
    total_annotations = sum(class_annotation_count.values())
    print(f"{'TOTAL':<25} {total_images:>10} {total_annotations:>12} {'100.0%':>12}")
    print()
    
    # Save to file
    output_path = Path('results/dataset_analysis.txt')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("FALCONEYE - TRAINING DATASET CLASS ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total training images analyzed: {total_images}\n")
        f.write(f"Empty label files: {empty_images}\n\n")
        f.write(f"{'Class':<25} {'Images':>10} {'Annotations':>12} {'% of Total':>12}\n")
        f.write("-" * 60 + "\n")
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            img_count = class_image_count[cls_id]
            ann_count = class_annotation_count[cls_id]
            percentage = (img_count / total_images * 100) if total_images > 0 else 0
            f.write(f"{cls_name:<25} {img_count:>10} {ann_count:>12} {percentage:>11.1f}%\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'TOTAL':<25} {total_images:>10} {total_annotations:>12} {'100.0%':>12}\n")
    
    print(f"Report saved to: results/dataset_analysis.txt")
    print("=" * 60)

if __name__ == '__main__':
    # Train labels only - excluding validation set as requested
    train_labels = r"C:\Users\Thrisha V C\Desktop\FalconEye\data\train_3\train3\labels"
    
    print("\nAnalyzing TRAINING set only (excluding validation)...")
    analyze_dataset(train_labels)
