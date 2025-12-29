"""
Chocolate Detection - Training Script
Uses your locally downloaded Roboflow dataset
"""

from ultralytics import YOLO
import torch
from pathlib import Path

def train_chocolate_detector():
    """Train YOLOv8 on local dataset."""

    # ============ CONFIGURATION ============

    # Dataset path (update this to your dataset location)
    DATASET_FOLDER = "dataset"  # ← Change if your folder has different name
    DATA_YAML = f"{DATASET_FOLDER}/data.yaml"

    # Training settings
    MODEL_SIZE = 'n'        # 'n' (nano) for speed, 's' (small) for accuracy
    EPOCHS = 100            # Can increase to 150-200 if needed
    BATCH_SIZE = 16         # Reduce to 8 if GPU memory error
    IMAGE_SIZE = 640        # Standard YOLO size

    # Save location
    PROJECT_NAME = 'runs/detect'
    RUN_NAME = 'chocolate_detector'

    # =======================================

    # Check if data.yaml exists
    if not Path(DATA_YAML).exists():
        print(f"❌ ERROR: {DATA_YAML} not found!")
        print(f"Make sure your dataset folder is in the same directory as this script.")
        return

    # Check GPU availability
    device = 0 if torch.cuda.is_available() else 'cpu'
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'

    print(f"\n{'='*70}")
    print(f"🎯 CHOCOLATE DETECTOR TRAINING")
    print(f"{'='*70}")
    print(f"Model: YOLOv8{MODEL_SIZE}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Device: {'GPU - ' + gpu_name if device == 0 else 'CPU'}")
    print(f"{'='*70}\n")

    # Load pretrained YOLOv8 model
    model = YOLO(f'yolov8{MODEL_SIZE}.pt')

    # Train the model
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        project=PROJECT_NAME,
        name=RUN_NAME,

        # Optimization
        optimizer='AdamW',
        lr0=0.01,
        patience=20,        # Early stopping

        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,

        # Saving
        save=True,
        save_period=-1,
        plots=True,

        # Performance
        workers=8,
        verbose=True
    )

    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {results.save_dir}")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    train_chocolate_detector()