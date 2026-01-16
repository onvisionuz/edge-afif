"""
Chocolate Detection - Training Script (OPTIMIZED FOR 95%+ ACCURACY)
Uses your locally downloaded Roboflow dataset from dataset/ folder

KEY OPTIMIZATIONS:
- YOLOv8s (Small) model for better accuracy vs YOLOv8n (Nano)
- 200 epochs with patience=50 for thorough convergence
- Cosine learning rate scheduler for better optimization
- Enhanced augmentation (rotation, scale, flip, mosaic)
- Early stopping prevents overfitting when model stops improving
- AMP (Automatic Mixed Precision) for faster GPU training
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

    # Training settings - Optimized for 95%+ accuracy
    MODEL_SIZE = 'n'        # 's' (small) for higher accuracy - best balance
    EPOCHS = 200            # Increased for better convergence with more data
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
    print(f"🎯 CHOCOLATE DETECTOR TRAINING - OPTIMIZED FOR 95%+ ACCURACY")
    print(f"{'='*70}")
    print(f"Model: YOLOv8{MODEL_SIZE} (Small - Better accuracy)")
    print(f"Dataset: {DATA_YAML}")
    print(f"Epochs: {EPOCHS} (with early stopping patience={20 if MODEL_SIZE == 'n' else 50})")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Device: {'GPU - ' + gpu_name if device == 0 else 'CPU'}")
    print(f"Optimizer: AdamW with cosine LR scheduler")
    print(f"Augmentation: Enhanced (rotation, scale, flip, mosaic)")
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

        # Optimization - Tuned for accuracy
        optimizer='AdamW',
        lr0=0.01,               # Initial learning rate
        lrf=0.01,               # Final learning rate (1% of lr0)
        momentum=0.937,         # SGD momentum/Adam beta1
        weight_decay=0.0005,    # Optimizer weight decay
        warmup_epochs=3.0,      # Warmup epochs
        warmup_momentum=0.8,    # Warmup initial momentum
        patience=50,            # Early stopping patience (increased for better convergence)
        cos_lr=True,            # Use cosine learning rate scheduler
        close_mosaic=10,        # Disable mosaic augmentation last N epochs for stability

        # Augmentation - Enhanced for better generalization
        hsv_h=0.015,            # HSV-Hue augmentation (image HSV-Hue range)
        hsv_s=0.7,              # HSV-Saturation augmentation
        hsv_v=0.4,              # HSV-Value augmentation
        degrees=10.0,           # Rotation (+/- deg)
        translate=0.1,          # Translation (+/- fraction)
        scale=0.5,              # Image scale (+/- gain)
        shear=0.0,              # Shear (+/- deg)
        perspective=0.0,        # Perspective (+/- fraction), range 0-0.001
        flipud=0.0,             # Vertical flip (probability)
        fliplr=0.5,             # Horizontal flip (probability)
        mosaic=1.0,             # Mosaic augmentation (probability)
        mixup=0.0,              # MixUp augmentation (probability)
        copy_paste=0.0,         # Copy-paste augmentation (probability)

        # Validation & Saving
        save=True,              # Save checkpoints
        save_period=-1,         # Save checkpoint every x epochs (-1 = disabled)
        val=True,               # Validate/test during training
        plots=True,             # Save plots and images

        # Performance
        workers=8,              # Number of worker threads for data loading
        verbose=True,           # Verbose output
        seed=0,                 # Random seed for reproducibility
        deterministic=True,     # Deterministic mode for reproducible results
        amp=True                # Automatic Mixed Precision training (faster on modern GPUs)
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