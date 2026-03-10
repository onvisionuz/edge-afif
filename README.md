# Chocolate Counter - Production Deployment

Real-time chocolate counting system using YOLOv8 detection with line-crossing tracking for conveyor belts.

**Features:** Video files & RTSP camera streams | 94.2% accuracy | Event delivery to backend API

---

## Quick Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit with your values
nano .env
```

**Required Configuration in `.env`:**

```bash
# Backend Server
SERVER_URL=http://api.onvision.uz/manufacturing/detect
API_KEY=mk_your_key_here  # Get from admin dashboard

# Device Registration (obtain from backend)
MANUFACTURE_ID=1
CAMERA_ID=3
PRODUCT_ID=2
DEVICE_ID=orangepi-001
LINE_ID=Line 1

# Optional: RTSP Camera
RTSP_URL=rtsp://user:password@192.168.1.100:554/stream
```

**Get backend IDs:** Register your manufacturer/camera/product in the admin dashboard → copy IDs to `.env`

### 3. Adjust Detection Settings (Optional)

Edit `config.yaml` for your setup:

```yaml
# ROI (Region of Interest) - adjust to fit your conveyor
roi:
  x1: 0.25  # Left edge (0.0-1.0)
  y1: 0.15  # Top edge
  x2: 0.95  # Right edge
  y2: 0.92  # Bottom edge

# Counting line position within ROI
counting_line:
  position: 0.75  # 0.0=top, 1.0=bottom
  direction: "down"  # down, up, or both

# Detection sensitivity
detection:
  confidence_threshold: 0.35  # Lower=more sensitive, Higher=stricter
  device: "cuda"  # cuda, cpu, or 0 (GPU index)

# Tracking
tracking:
  tracker_type: "bytetrack"  # or "botsort"
  min_track_age: 3  # Frames before counting (anti-noise)
```

**Models Available:**
- `chocolate_detector3` (94.2% mAP50) - **Active** ✅
- `chocolate_detector` - Legacy

Model path is set in `config.yaml` → `paths.model`

---

## Usage

### Video File Processing

```bash
# Production (headless, logs only)
python counter.py --video input.mp4

# Debug (show display)
python counter.py --video input.mp4 --show

# Save annotated video
python counter.py --video input.mp4 --save-video
```

### RTSP Camera Stream

```bash
# Production (headless)
python counter.py --rtsp

# Debug (live view)
python counter.py --rtsp --show

# URL override
python counter.py --rtsp-url rtsp://user:pass@192.168.1.50:554/stream --show
```

**Output:** All modes generate logs in `output/logs/`
- `count_log_*.csv` - Frame-by-frame data
- `events_log_*.csv` - Crossing events
- `summary_*.json` - Statistics

---

## Event Delivery System

**How it works:**
1. Detections are counted when crossing the virtual line
2. Events buffered in local SQLite database (`events.db`)
3. Batched and sent to backend every 5 seconds via HTTP POST
4. Only deleted after successful 201 response (zero data loss during outages)
5. Automatic retry on network failures

**Authentication:** API key sent via `X-Manufacture-API-Key` header

**Batch Format:**
```json
{
  "manufacture_id": 1,
  "detections": [
    {
      "camera_id": 3,
      "product_id": 2,
      "device_id": "orangepi-001",
      "line_id": "Line 1",
      "detection_class": null,
      "timestamp": "2026-03-10T07:00:00+00:00"
    }
  ]
}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No detections / zero count | Lower `confidence_threshold` to 0.25-0.30 in config.yaml |
| Too many false positives | Increase threshold to 0.40-0.45, tighten ROI area |
| Random counting | Adjust `counting_line.position`, verify direction setting |
| API authentication error (401) | Check `API_KEY` in .env matches admin dashboard |
| Authorization error (403) | Verify `MANUFACTURE_ID` matches API key registration |
| Model not found | Check `paths.model` in config.yaml points to existing .pt file |
| RTSP connection failed | Verify `RTSP_URL` credentials and camera IP |

**Check logs:** Errors appear in console output and `output/logs/*.log`

---

## Configuration Architecture

**`.env`** (gitignored - never commit):
- Backend server URL and API key
- Device IDs and registration info
- RTSP credentials
- Deployment-specific settings

**`config.yaml`** (safe to commit):
- ROI coordinates and counting line
- Detection thresholds and model path
- Tracking and performance settings
- Shareable across similar deployments

---

## CLI Reference

```bash
python counter.py [INPUT] [OPTIONS]

Input (choose one):
  --video PATH        Video file path
  --rtsp              Use RTSP from .env
  --rtsp-url URL      RTSP URL override

Options:
  --show              Show live display
  --save-video        Save annotated video
  --config PATH       Custom config (default: config.yaml)
```

---

## Production Recommendations

**Edge Device Deployment:**
```bash
# Headless mode (minimal resources)
python counter.py --rtsp

# Logs provide all data, no UI overhead
# Events auto-delivered to backend
# Resilient to network interruptions
```

**Performance:**
- Desktop GPU: 20-30 FPS
- Edge device (Jetson, Orange Pi): 10-15 FPS
- Counting accuracy: 95-99% with proper tuning

---

**Status:** Production-ready | Supports video files and RTSP camera streams | Backend event delivery with offline buffering
