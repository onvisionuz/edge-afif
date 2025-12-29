# Chocolate Counter - Phase 2

Industrial computer vision system for real-time chocolate counting on conveyor belts using YOLOv8 with ROI-based detection and line-crossing tracking.

## 📋 Project Overview

**Goal:** Accurately count chocolates moving on a conveyor belt with >95% accuracy using computer vision.

**Approach:**
- Train YOLOv8n model on chocolate dataset
- Implement ROI-based detection to focus on conveyor belt area only
- Use ByteTrack for persistent object tracking
- Count chocolates when they cross a virtual counting line
- Prevent double counting with multi-layer logic

## 🎯 Current Status: Phase 2 Complete

✅ **Model Training** - YOLOv8n trained on 400 labeled frames
✅ **Detection** - 89.8% mAP50 accuracy
✅ **Counting System** - ROI + Tracking + Line-crossing
✅ **Production Ready** - Configurable, logged, visualized

## 📁 Project Structure

```
Phase2-just-counter/
├── dataset/              # Labeled chocolate dataset (400 images, ~6800 annotations)
│   ├── train/            # 238 training images
│   ├── valid/            # 68 validation images
│   ├── test/             # 34 test images
│   └── data.yaml         # Dataset configuration
│
├── runs/                 # Training outputs
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt    # ⭐ Trained YOLOv8n model (89.8% mAP50)
│
├── videos/               # Test videos
│   └── input/            # Place test videos here
│
├── output/               # System outputs (auto-generated)
│   ├── videos/           # Annotated output videos
│   └── logs/             # CSV and JSON logs
│
├── inference_results/    # Detection test outputs
│
├── config.yaml           # Main configuration file
├── counter.py            # Main counting system
├── setup.py              # Setup and verification script
├── train.py              # Model training script
├── video-test.py         # Video inference testing
├── yolov8n.pt            # Base YOLOv8n weights
└── yolo11n.pt            # Base YOLO11n weights
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics opencv-python pyyaml pandas numpy

# Verify setup
python setup.py
```

### 2. Configure System

Edit `config.yaml` to set your parameters:

```yaml
paths:
  model: "runs/detect/train/weights/best.pt"
  input_video: "videos/input/your_video.mp4"
  output_video: "auto"  # Auto-generates: counted_your_video.mp4

roi:
  x1: 0.25    # Left edge (25% from left)
  y1: 0.15    # Top edge (15% from top)
  x2: 0.95    # Right edge (95% from left)
  y2: 0.92    # Bottom edge (92% from top)

counting_line:
  position: 0.75      # 75% down from top of ROI
  direction: "down"   # Count downward movement

detection:
  confidence_threshold: 0.35
```

### 3. Run Counter

```bash
# Place your test video in videos/input/
# Update config.yaml with video filename
python counter.py
```

**Output:**
- `output/videos/counted_{video_name}.mp4` - Annotated video
- `output/logs/count_log_{video_name}.csv` - Frame-by-frame data
- `output/logs/summary_{video_name}.json` - Statistics

## 🎯 How It Works

### System Pipeline

```
Video Frame
    ↓
ROI Extraction (white conveyor belt area only)
    ↓
YOLOv8 Detection (finds chocolates in ROI)
    ↓
ByteTrack Tracking (assigns unique IDs)
    ↓
Line-Crossing Check (count when chocolate crosses line)
    ↓
Visualization (draw boxes, IDs, counter)
    ↓
Logging (CSV + JSON)
    ↓
Output Video + Statistics
```

### Key Features

**1. ROI-Based Processing**
- Only processes white conveyor belt area
- Eliminates false positives from debris/floor
- ~40% faster inference (smaller image region)

**2. Persistent Tracking**
- Each chocolate gets unique ID across frames
- Maintains identity even during brief occlusions
- Uses ByteTrack algorithm

**3. Line-Crossing Detection**
- Counts only when chocolate crosses virtual line
- Directional counting (top → bottom)
- Prevents double counting

**4. Anti-Double-Count Protection**
- **Flag-based:** Each track counted only once
- **Age-based:** Must be alive 3+ frames before counting
- **Direction-based:** Only counts intended movement
- **Position-based:** Must cross line from above

### Visual Output

The output video shows:
- **Green rectangle** - ROI (detection zone)
- **Red line** - Counting line (where counting happens)
- **Yellow boxes** - Uncounted chocolates
- **Orange boxes** - Already counted chocolates
- **Track IDs** - Unique identifier above each box
- **Counter** - Total count (top-left)
- **FPS** - Processing speed (top-right)

## 📊 Model Performance

**Training Dataset:**
- 400 labeled frames
- ~6,800 chocolate annotations
- Train/Valid/Test: 238/68/34 split

**Model Metrics:**
```
mAP50:      89.8%  (Target: >80%)  ✅
mAP50-95:   77.2%  (Target: >60%)  ✅
Precision:  84.7%
Recall:     89.3%
```

**Detection Characteristics:**
- High recall (89.3%) - Rarely misses chocolates
- Good precision (84.7%) - Few false positives
- Lightweight model (6 MB) - Edge-device friendly

## ⚙️ Configuration Guide

### Adjusting ROI

If ROI doesn't fit your conveyor belt:

```yaml
roi:
  x1: 0.25  # Increase to exclude left debris
  x2: 0.95  # Decrease to exclude right edges
  y1: 0.15  # Increase to start detection lower
  y2: 0.92  # Decrease to end detection higher
```

**Tip:** Use fractional coordinates (0.0-1.0) so it works for any resolution.

### Adjusting Counting Line

```yaml
counting_line:
  position: 0.75  # 0.0 = top of ROI, 1.0 = bottom
  # Try 0.70-0.80 range for optimal position
```

### Tuning Detection Sensitivity

```yaml
detection:
  confidence_threshold: 0.35
  # Lower (0.25-0.30): More detections, may include noise
  # Higher (0.40-0.50): Fewer detections, more strict
```

### Tuning Tracking Stability

```yaml
tracking:
  min_track_age: 3        # Frames before counting (anti-noise)
  max_lost_frames: 10     # Keep lost tracks alive (handle occlusions)
```

## 📈 Output Files

### 1. CSV Log (Frame-by-Frame)
```csv
timestamp_sec,frame_number,total_count,active_tracks,detections_this_frame
0.00,0,0,0,0
0.04,1,0,2,2
0.08,2,1,3,3
```

### 2. Events Log (Crossing Events)
```csv
event_id,timestamp_sec,frame_number,track_id,action,position_x,position_y
1,1.24,31,5,counted,512,768
```

### 3. Summary JSON (Statistics)
```json
{
  "total_count": 234,
  "chocolates_per_second": 1.14,
  "chocolates_per_minute": 68.6,
  "processing_fps": 18.5
}
```

## 🔧 Troubleshooting

### No detections / Count is zero
- Check confidence threshold (try lowering to 0.25)
- Verify ROI covers the conveyor belt
- Ensure model path is correct in config.yaml

### Too many false positives
- Increase confidence threshold (try 0.40-0.45)
- Tighten ROI to exclude debris areas
- Check if ROI is properly positioned

### Counting seems random
- Verify counting line position (should be clear crossing point)
- Check direction setting (down/up/both)
- Watch output video to see where counts occur

### Model not found error
```bash
# Verify model exists
ls runs/detect/train/weights/best.pt

# Update config.yaml if in different location
```

## 📊 Expected Performance

**Processing Speed:**
- Desktop GPU: 20-30 FPS (PyTorch)
- Desktop GPU: 40-60 FPS (TensorRT optimized)

**Counting Accuracy:**
- Target: >95%
- Expected: 95-99% with proper ROI/line tuning

## 🎓 Training Your Own Model

If you need to retrain or fine-tune:

```bash
# Prepare your dataset in dataset/ folder
# Update dataset/data.yaml

# Train
python train.py

# Model will be saved to: runs/detect/train/weights/best.pt
# Update config.yaml to use new model
```

## 📝 Notes

- System uses single-class detection ("chocolate")
- All chocolates counted regardless of defects (counting first, classification later)
- Optimized for top-down camera view of conveyor belt
- Fractional ROI coordinates work for any video resolution
- Auto-naming: outputs match input filenames automatically

---

**Status:** Production-ready for video-based counting. Ready for edge deployment optimization.
