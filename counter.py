#!/usr/bin/env python3
"""
Chocolate Counter - Industrial Computer Vision System
Real-time chocolate counting with ROI detection and line-crossing tracking

Supports:
- Video file input (offline/batch processing)
- RTSP/RTCP camera streams (real-time processing)

Author: Adapted for production deployment
Version: 3.0
"""

import cv2
import yaml
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, deque
from ultralytics import YOLO
import logging
import sqlite3
import requests
import threading
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars only


class EventSender:
    """
    Handles buffering and batched sending of detection events to a server.
    Uses a local SQLite database for resilient queueing during network outages.
    """
    def __init__(self, config: dict, db_path: str = "events.db"):
        self.config = config
        self.db_path = db_path
        self._init_db()
        self.running = True

        # Read server configuration from environment variables
        self.flush_interval = int(os.getenv("FLUSH_INTERVAL_SECONDS", "5"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "50"))
        self.endpoint_url = os.getenv("SERVER_URL", "")

        self.logger = logging.getLogger(__name__ + ".EventSender")

        # Start background worker
        if self.endpoint_url:
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()
        else:
            self.logger.warning("No endpoint_url configured. Events will be saved locally but not sent.")

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def add_event(self, payload: dict):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO events (payload) VALUES (?)', (json.dumps(payload),))
            conn.commit()

    def _worker_loop(self):
        while self.running:
            try:
                self._flush_events()
            except Exception as e:
                self.logger.error(f"Error in EventSender worker: {e}")
            time.sleep(self.flush_interval)

        # Final flush on stop
        try:
            self._flush_events()
        except:
            pass

    def _flush_events(self):
        if not self.endpoint_url:
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Fetch batch of unsent events
            cursor.execute('SELECT id, payload FROM events ORDER BY id ASC LIMIT ?', (self.batch_size,))
            rows = cursor.fetchall()

            if not rows:
                return

            event_ids = [row['id'] for row in rows]
            payloads = [json.loads(row['payload']) for row in rows]

            # Send HTTP POST with batch wrapper
            try:
                # Construct batch payload with manufacture_id wrapper
                manufacture_id = int(os.getenv("MANUFACTURE_ID", "0"))
                batch_payload = {
                    "manufacture_id": manufacture_id,
                    "detections": payloads  # Events nested under "detections" key
                }

                # Get API key from environment
                api_key = os.getenv("API_KEY", "")

                response = requests.post(
                    self.endpoint_url,
                    json=batch_payload,
                    headers={
                        'Content-Type': 'application/json',
                        'X-Manufacture-API-Key': api_key
                    },
                    timeout=10  # Increased from 5 to 10 seconds
                )

                # Handle specific response codes
                if response.status_code == 201:
                    self.logger.info(f"Successfully sent {len(payloads)} batched events to server.")
                    # Delete sent events from DB
                    placeholders = ','.join(['?'] * len(event_ids))
                    cursor.execute(f'DELETE FROM events WHERE id IN ({placeholders})', event_ids)
                    conn.commit()
                elif response.status_code == 401:
                    self.logger.error(f"Authentication failed: Missing or invalid API key. Check API_KEY in .env")
                elif response.status_code == 403:
                    self.logger.error(f"Authorization failed: API key doesn't match manufacture_id. Check MANUFACTURE_ID in .env")
                elif response.status_code == 422:
                    self.logger.error(f"Payload validation error: {response.text}")
                else:
                    self.logger.warning(f"Failed to send events. Server returned {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Connection error sending events. Will retry next cycle. {e}")

    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)


class ChocolateCounter:
    """
    Main counting system with ROI-based detection and line-crossing tracking.

    Features:
    - ROI-only processing (white conveyor belt)
    - Persistent tracking with unique IDs
    - Line-crossing detection with anti-double-count
    - CSV and JSON logging
    - RTSP/RTCP camera stream support
    - Production-ready error handling
    """

    def __init__(self, config_path: str = "config.yaml",
                 show_video: Optional[bool] = None,
                 save_video: Optional[bool] = None):
        """
        Initialize the counter with configuration.

        Args:
            config_path: Path to YAML configuration file
            show_video: Override config for real-time display (None = use config)
            save_video: Override config for video saving (None = use config)
        """

        # Load configuration
        self.config = self._load_config(config_path)

        # Runtime flags (CLI overrides config)
        self._show_video = show_video
        self._save_video = save_video

        # Track input mode
        self.is_stream = False
        self.source_name = "unknown"

        # Setup logging
        self._setup_logging()

        # Initialize state
        self.total_count = 0
        self.tracks: Dict[int, dict] = {}  # track_id -> track state
        self.counted_tracks = set()  # Set of track IDs that crossed line

        # Initialize EventSender
        self.event_sender = EventSender(self.config)

        # Statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'total_detections': 0,
            'avg_detections_per_frame': 0,
            'avg_confidence': 0,
            'processing_fps': 0,
            'errors': 0,
            'input_mode': 'unknown'
        }

        # Logging data
        self.log_data = []
        self.event_log = []

        # Load model
        self._load_model()

        self.logger.info("ChocolateCounter initialized successfully")

    @property
    def show_video(self) -> bool:
        """Determine if real-time display is enabled."""
        if self._show_video is not None:
            return self._show_video
        # Check new config structure first, fall back to legacy
        if 'output' in self.config and 'show_video' in self.config['output']:
            return self.config['output']['show_video']
        return self.config.get('debug', {}).get('display_realtime', False)

    @property
    def save_video(self) -> bool:
        """Determine if video saving is enabled."""
        if self._save_video is not None:
            return self._save_video
        # Check new config structure first
        if 'output' in self.config and 'save_video' in self.config['output']:
            return self.config['output']['save_video']
        # Legacy: always saved by default
        return True

    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _resolve_input_source(self, video_path: Optional[str] = None,
                               rtsp_url: Optional[str] = None,
                               use_rtsp: bool = False) -> Tuple[cv2.VideoCapture, bool, str]:
        """
        Resolve input source and return VideoCapture object.

        Args:
            video_path: Path to video file (takes priority if provided)
            rtsp_url: RTSP URL (CLI override)
            use_rtsp: If True, use RTSP mode (from .env if rtsp_url not provided)

        Returns:
            Tuple of (VideoCapture, is_stream, source_name)
        """
        # Priority 1: Explicit video file path
        if video_path:
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            self.logger.info(f"Input mode: VIDEO FILE - {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            return cap, False, Path(video_path).stem

        # Priority 2: RTSP mode
        if use_rtsp or rtsp_url:
            # Get RTSP URL: CLI override > .env > error
            url = rtsp_url or os.getenv('RTSP_URL')
            if not url:
                raise ValueError(
                    "RTSP mode enabled but no URL provided. "
                    "Set RTSP_URL in .env or use --rtsp-url flag."
                )
            self.logger.info(f"Input mode: RTSP STREAM")
            self.logger.info(f"  URL: {self._mask_rtsp_url(url)}")

            cap = cv2.VideoCapture(url)

            # Optimize for streaming
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

            if not cap.isOpened():
                raise RuntimeError(f"Failed to connect to RTSP stream: {self._mask_rtsp_url(url)}")

            return cap, True, "rtsp_stream"

        # Priority 3: Config file settings
        input_config = self.config.get('input', {})
        mode = input_config.get('mode', 'video')

        if mode == 'rtsp':
            url = os.getenv('RTSP_URL')
            if not url:
                raise ValueError("Config mode is 'rtsp' but RTSP_URL not set in .env")
            self.logger.info(f"Input mode: RTSP STREAM (from config)")
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to connect to RTSP stream")
            return cap, True, "rtsp_stream"

        # Default: video file from config
        video_path = input_config.get('video_path') or self.config['paths'].get('input_video')
        if not video_path:
            raise ValueError("No input source specified. Use --video or --rtsp flag.")
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.logger.info(f"Input mode: VIDEO FILE (from config) - {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        return cap, False, Path(video_path).stem

    def _mask_rtsp_url(self, url: str) -> str:
        """Mask password in RTSP URL for logging."""
        import re
        # Match rtsp://user:password@host pattern
        masked = re.sub(r'(rtsp://[^:]+:)[^@]+(@)', r'\1****\2', url)
        return masked

    def _auto_generate_output_paths(self, source_name: str):
        """Auto-generate output paths based on input source name."""
        # For streams, add timestamp to make unique filenames
        if self.is_stream:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{source_name}_{timestamp}"
        else:
            base_name = source_name

        # Auto-generate output video path
        if self.config['paths'].get('output_video') == 'auto':
            self.config['paths']['output_video'] = f"output/videos/counted_{base_name}.mp4"

        # Auto-generate CSV log path
        if self.config['paths'].get('output_csv') == 'auto':
            self.config['paths']['output_csv'] = f"output/logs/count_log_{base_name}.csv"

        # Auto-generate JSON summary path
        if self.config['paths'].get('output_json') == 'auto':
            self.config['paths']['output_json'] = f"output/logs/summary_{base_name}.json"

        self.logger.info(f"Output paths:")
        self.logger.info(f"  Video: {self.config['paths']['output_video']}")
        self.logger.info(f"  CSV:   {self.config['paths']['output_csv']}")
        self.logger.info(f"  JSON:  {self.config['paths']['output_json']}")

    def _setup_logging(self):
        """Setup logging system."""
        log_level = logging.DEBUG if self.config['debug'].get('verbose', True) else logging.INFO

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config['paths'].get('error_log', 'errors.log'))
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_model(self):
        """Load YOLO model for chocolate detection."""
        model_path = self.config['paths']['model']

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.logger.info(f"Loading model from: {model_path}")

        # Load model
        self.model = YOLO(model_path)

        # Set device
        device = self.config['detection']['device']
        if device == 'cuda' and not cv2.cuda.getCudaEnabledDeviceCount():
            self.logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'

        # Warm up model
        self.logger.info("Warming up model...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model.predict(dummy_img, verbose=False)

        self.logger.info(f"Model loaded successfully on device: {device}")

    def _calculate_roi_coords(self, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """Convert fractional ROI coordinates to absolute pixel coordinates."""
        roi = self.config['roi']

        x1 = int(roi['x1'] * frame_width)
        y1 = int(roi['y1'] * frame_height)
        x2 = int(roi['x2'] * frame_width)
        y2 = int(roi['y2'] * frame_height)

        return x1, y1, x2, y2

    def _calculate_line_position(self, roi_y1: int, roi_y2: int) -> int:
        """Calculate counting line Y position within ROI."""
        line_pos = self.config['counting_line']['position']
        roi_height = roi_y2 - roi_y1
        line_y = roi_y1 + int(line_pos * roi_height)
        return line_y

    def _extract_roi(self, frame: np.ndarray, roi_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract ROI region from frame."""
        x1, y1, x2, y2 = roi_coords
        return frame[y1:y2, x1:x2]

    def _convert_roi_to_frame_coords(self, bbox: List[float], roi_coords: Tuple[int, int, int, int]) -> List[float]:
        """Convert bounding box from ROI coordinates to full frame coordinates."""
        x1, y1, x2, y2 = bbox
        roi_x1, roi_y1, _, _ = roi_coords

        # Add ROI offset
        return [
            x1 + roi_x1,
            y1 + roi_y1,
            x2 + roi_x1,
            y2 + roi_y1
        ]

    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Calculate center point of bounding box."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return cx, cy

    def _check_line_crossing(self, track_id: int, current_center: Tuple[float, float],
                            line_y: int, frame_number: int) -> bool:
        """
        Check if chocolate crossed the counting line.

        Returns True if crossed and should be counted.
        """
        cx, cy = current_center
        direction = self.config['counting_line']['direction']
        margin = self.config['counting_line'].get('margin', 5)

        # Get track history
        track = self.tracks.get(track_id)
        if not track:
            return False

        # Check if already counted
        if track.get('counted', False):
            return False

        # Check track age (must be alive for min frames before counting)
        min_age = self.config['tracking']['min_track_age']
        if track.get('age', 0) < min_age:
            return False

        # Get previous center
        prev_center = track.get('prev_center')
        if prev_center is None:
            return False

        prev_cx, prev_cy = prev_center

        # Check crossing based on direction
        crossed = False

        if direction == "down":
            # Moving downward: prev_cy < line_y and cy >= line_y
            crossed = (prev_cy < line_y - margin) and (cy >= line_y - margin)

        elif direction == "up":
            # Moving upward: prev_cy > line_y and cy <= line_y
            crossed = (prev_cy > line_y + margin) and (cy <= line_y + margin)

        elif direction == "both":
            # Either direction
            crossed = ((prev_cy < line_y - margin) and (cy >= line_y - margin)) or \
                     ((prev_cy > line_y + margin) and (cy <= line_y + margin))

        if crossed:
            self.logger.debug(f"Track {track_id} crossed line at frame {frame_number}")

        return crossed

    def _update_tracks(self, results, roi_coords: Tuple[int, int, int, int],
                      line_y: int, frame_number: int, timestamp: float):
        """Update track states and check for line crossings."""

        current_track_ids = set()

        # Process each detection with tracking
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                # Get track ID (if tracking is enabled)
                if hasattr(boxes, 'id') and boxes.id is not None:
                    track_id = int(boxes.id[i])
                else:
                    # No tracking - use sequential ID (fallback)
                    track_id = frame_number * 1000 + i

                current_track_ids.add(track_id)

                # Get bbox in ROI coordinates
                bbox_roi = boxes.xyxy[i].cpu().numpy().tolist()

                # Convert to frame coordinates
                bbox_frame = self._convert_roi_to_frame_coords(bbox_roi, roi_coords)

                # Get center point
                center = self._get_bbox_center(bbox_frame)

                # Get confidence
                conf = float(boxes.conf[i])

                # Initialize or update track
                if track_id not in self.tracks:
                    # New track
                    self.tracks[track_id] = {
                        'track_id': track_id,
                        'bbox': bbox_frame,
                        'center': center,
                        'prev_center': None,
                        'confidence': conf,
                        'first_seen': frame_number,
                        'last_seen': frame_number,
                        'age': 1,
                        'counted': False,
                        'lost_frames': 0
                    }
                else:
                    # Update existing track
                    track = self.tracks[track_id]
                    track['prev_center'] = track['center']  # Store previous position
                    track['center'] = center
                    track['bbox'] = bbox_frame
                    track['confidence'] = conf
                    track['last_seen'] = frame_number
                    track['age'] += 1
                    track['lost_frames'] = 0

                # Check for line crossing
                if self._check_line_crossing(track_id, center, line_y, frame_number):
                    # Count this chocolate!
                    self.total_count += 1
                    self.tracks[track_id]['counted'] = True
                    self.counted_tracks.add(track_id)

                    # Generate payload for device (read from environment variables)
                    event_payload = {
                        "camera_id": int(os.getenv("CAMERA_ID", "0")),
                        "product_id": int(os.getenv("PRODUCT_ID", "0")),
                        "device_id": os.getenv("DEVICE_ID", ""),
                        "line_id": os.getenv("LINE_ID"),  # Optional, can be None
                        "detection_class": None,  # For future classification models
                        "timestamp": datetime.now(timezone.utc).isoformat()  # UTC timezone
                    }

                    # Queue event to SQLite and to send to server
                    self.event_sender.add_event(event_payload)

                    # Log crossing event
                    self.event_log.append({
                        'event_id': len(self.event_log) + 1,
                        'timestamp_sec': timestamp,
                        'frame_number': frame_number,
                        'track_id': track_id,
                        'action': 'counted',
                        'position_x': center[0],
                        'position_y': center[1]
                    })

                    self.logger.info(f"✓ Counted chocolate #{self.total_count} (Track ID: {track_id})")

        # Update lost tracks (not detected in current frame)
        max_lost = self.config['tracking']['max_lost_frames']
        tracks_to_remove = []

        for track_id, track in self.tracks.items():
            if track_id not in current_track_ids:
                track['lost_frames'] += 1

                # Remove if lost for too long
                if track['lost_frames'] > max_lost:
                    tracks_to_remove.append(track_id)
                    self.logger.debug(f"Track {track_id} removed (lost for {track['lost_frames']} frames)")

        # Clean up dead tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]

    def _draw_visualizations(self, frame: np.ndarray, roi_coords: Tuple[int, int, int, int],
                           line_y: int, frame_number: int, fps: float) -> np.ndarray:
        """Draw all visualizations on frame."""

        viz = self.config['visualization']
        colors = viz['colors']
        thickness = viz['line_thickness']
        roi_thickness = viz.get('roi_thickness', 3)
        line_thickness = viz.get('counting_line_thickness', 4)
        font_scale = viz['font_scale']

        output = frame.copy()

        # Draw ROI rectangle
        if viz['show_roi']:
            x1, y1, x2, y2 = roi_coords
            cv2.rectangle(output, (x1, y1), (x2, y2), colors['roi'], roi_thickness)
            cv2.putText(output, "ROI", (x1 + 10, y1 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors['roi'], 2)

        # Draw counting line
        if viz['show_line']:
            x1, _, x2, _ = roi_coords
            cv2.line(output, (x1, line_y), (x2, line_y), colors['line'], line_thickness)
            cv2.putText(output, "COUNT LINE", (x2 - 200, line_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, colors['line'], 2)

        # Draw bounding boxes and track IDs
        for track_id, track in self.tracks.items():
            bbox = track['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # Choose color based on counted status
            if track['counted']:
                box_color = colors['counted_box']
            else:
                box_color = colors['uncounted_box']

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), box_color, thickness)

            # Draw track ID
            if viz['show_track_ids']:
                label = f"ID: {track_id}"

                # Add confidence if enabled
                if viz['show_confidence']:
                    label += f" ({track['confidence']:.2f})"

                # Draw label with background
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, 2
                )
                cv2.rectangle(output, (x1, y1 - label_height - 10),
                            (x1 + label_width, y1), box_color, -1)
                cv2.putText(output, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6,
                           colors['track_id'], 2)

            # Draw trails (path history)
            if viz['show_trails'] and track.get('prev_center'):
                current = tuple(map(int, track['center']))
                previous = tuple(map(int, track['prev_center']))
                cv2.line(output, previous, current, box_color, 2)

        # Draw counter display
        counter_text = f"COUNT: {self.total_count}"
        cv2.putText(output, counter_text, (30, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.5,
                   colors['counter_text'], 3)

        # Draw statistics
        if viz['show_stats']:
            active_tracks = len(self.tracks)
            stats_text = f"Active: {active_tracks}"
            cv2.putText(output, stats_text, (30, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8,
                       (255, 255, 255), 2)

        # Draw FPS
        if viz['show_fps']:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(output, fps_text, (frame.shape[1] - 150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8,
                       (255, 255, 255), 2)

        return output

    def _save_logs(self, source_name: str, duration: float):
        """Save CSV and JSON logs."""

        # Create output directories
        Path(self.config['paths']['output_csv']).parent.mkdir(parents=True, exist_ok=True)

        # Save CSV log
        if self.config['logging']['enable_csv'] and self.log_data:
            df = pd.DataFrame(self.log_data)
            csv_path = self.config['paths']['output_csv']
            df.to_csv(csv_path, index=False)
            self.logger.info(f"CSV log saved: {csv_path}")

        # Save event log
        if self.config['logging']['enable_events'] and self.event_log:
            events_path = self.config['paths']['output_csv'].replace('count_log', 'events_log')
            df_events = pd.DataFrame(self.event_log)
            df_events.to_csv(events_path, index=False)
            self.logger.info(f"Events log saved: {events_path}")

        # Save summary JSON
        if self.config['logging']['enable_summary']:
            summary = {
                'source': source_name,
                'input_mode': self.stats['input_mode'],
                'processed_at': datetime.now().isoformat(),
                'duration_sec': duration,
                'total_frames': self.stats['total_frames'] if not self.is_stream else 'N/A (stream)',
                'processed_frames': self.stats['processed_frames'],
                'total_count': self.total_count,
                'chocolates_per_second': self.total_count / duration if duration > 0 else 0,
                'chocolates_per_minute': (self.total_count / duration) * 60 if duration > 0 else 0,
                'avg_chocolates_per_frame': self.stats['avg_detections_per_frame'],
                'peak_chocolates_per_frame': max([d['detections_this_frame'] for d in self.log_data]) if self.log_data else 0,
                'processing_fps': self.stats['processing_fps'],
                'model_metrics': {
                    'avg_confidence': self.stats['avg_confidence'],
                    'total_detections': self.stats['total_detections']
                },
                'runtime_flags': {
                    'show_video': self.show_video,
                    'save_video': self.save_video
                }
            }

            json_path = self.config['paths']['output_json']
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)

            self.logger.info(f"Summary JSON saved: {json_path}")

    def process(self, video_path: Optional[str] = None,
                rtsp_url: Optional[str] = None,
                use_rtsp: bool = False):
        """
        Main processing pipeline - handles both video files and RTSP streams.

        Args:
            video_path: Path to input video file
            rtsp_url: RTSP URL (overrides .env)
            use_rtsp: If True, use RTSP mode (URL from .env if rtsp_url not provided)
        """
        # Resolve input source
        cap, self.is_stream, self.source_name = self._resolve_input_source(
            video_path=video_path,
            rtsp_url=rtsp_url,
            use_rtsp=use_rtsp
        )

        # Update stats
        self.stats['input_mode'] = 'rtsp' if self.is_stream else 'video'

        # Auto-generate output paths
        self._auto_generate_output_paths(self.source_name)

        # Log runtime flags
        self.logger.info(f"Runtime flags:")
        self.logger.info(f"  Show video: {self.show_video}")
        self.logger.info(f"  Save video: {self.save_video}")

        # Get video/stream properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default 30 FPS for streams

        if self.is_stream:
            total_frames = float('inf')  # Unknown for streams
            duration = 0
            self.logger.info(f"Stream properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")
            self.logger.info(f"Stream mode: continuous (press 'q' to stop if --show enabled)")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            self.stats['total_frames'] = total_frames
            self.logger.info(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")
            self.logger.info(f"Total frames: {total_frames}, Duration: {duration:.2f}s")

        # Calculate ROI and line coordinates
        roi_coords = self._calculate_roi_coords(frame_width, frame_height)
        line_y = self._calculate_line_position(roi_coords[1], roi_coords[3])

        self.logger.info(f"ROI coordinates: {roi_coords}")
        self.logger.info(f"Counting line Y position: {line_y}")

        # Setup output video writer (only if saving)
        out = None
        output_video_path = None
        if self.save_video and not self.is_stream:
            output_video_path = self.config['paths']['output_video']
            Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self.config['performance']['output_codec'])
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            self.logger.info(f"Video will be saved to: {output_video_path}")
        elif self.save_video and self.is_stream:
            self.logger.warning("Video saving is not supported for RTSP streams (ignored)")

        # Processing loop
        frame_number = 0
        processing_start = datetime.now()
        progress_interval = self.config['logging']['progress_interval']

        # Detection parameters
        conf_thresh = self.config['detection']['confidence_threshold']
        iou_thresh = self.config['detection']['iou_threshold']
        tracker = f"{self.config['tracking']['tracker_type']}.yaml"

        self.logger.info("Starting frame processing loop...")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if self.is_stream:
                        self.logger.warning("Lost stream connection, attempting reconnect...")
                        continue  # Try to read again for streams
                    break  # End of video file

                # Check frame limits (for debugging, not applicable to streams)
                max_frames = self.config['debug'].get('max_frames')
                if max_frames and frame_number >= max_frames:
                    self.logger.info(f"Reached max frame limit: {max_frames}")
                    break

                timestamp = frame_number / fps if fps > 0 else 0

                # Skip corrupted frames
                if self.config['advanced']['skip_corrupted_frames']:
                    if frame is None or frame.size == 0:
                        self.logger.warning(f"Skipping corrupted frame {frame_number}")
                        frame_number += 1
                        continue

                # Extract ROI
                roi_frame = self._extract_roi(frame, roi_coords)

                # Run detection with tracking
                results = self.model.track(
                    source=roi_frame,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    persist=self.config['tracking']['persist'],
                    tracker=tracker,
                    verbose=False,
                    device=self.config['detection']['device']
                )

                # Count detections
                num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
                self.stats['total_detections'] += num_detections

                # Update tracks and check crossings
                self._update_tracks(results, roi_coords, line_y, frame_number, timestamp)

                # Calculate processing FPS
                elapsed = (datetime.now() - processing_start).total_seconds()
                current_fps = (frame_number + 1) / elapsed if elapsed > 0 else 0

                # Draw visualizations (needed for both display and save)
                output_frame = self._draw_visualizations(
                    frame, roi_coords, line_y, frame_number, current_fps
                )

                # Write to output video (if enabled and not streaming)
                if out is not None:
                    out.write(output_frame)

                # Log data
                self.log_data.append({
                    'timestamp_sec': timestamp,
                    'frame_number': frame_number,
                    'total_count': self.total_count,
                    'active_tracks': len(self.tracks),
                    'detections_this_frame': num_detections
                })

                # Display progress
                if frame_number % progress_interval == 0:
                    if self.is_stream:
                        self.logger.info(
                            f"Frame: {frame_number} | "
                            f"Count: {self.total_count} | Active: {len(self.tracks)} | "
                            f"FPS: {current_fps:.1f}"
                        )
                    else:
                        progress = (frame_number / total_frames) * 100 if total_frames > 0 else 0
                        self.logger.info(
                            f"Progress: {frame_number}/{int(total_frames)} ({progress:.1f}%) | "
                            f"Count: {self.total_count} | Active: {len(self.tracks)} | "
                            f"FPS: {current_fps:.1f}"
                        )

                # Real-time display (if --show flag enabled)
                if self.show_video:
                    cv2.imshow('Chocolate Counter', output_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("User stopped processing (pressed 'q')")
                        break

                frame_number += 1
                self.stats['processed_frames'] = frame_number

        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user (Ctrl+C)")

        except Exception as e:
            self.logger.error(f"Error during processing: {e}", exc_info=True)
            self.stats['errors'] += 1

        finally:
            # Stop EventSender gracefully
            if hasattr(self, 'event_sender'):
                self.logger.info("Stopping EventSender background thread...")
                self.event_sender.stop()

            # Cleanup
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()

            # Calculate final statistics
            processing_time = (datetime.now() - processing_start).total_seconds()
            self.stats['processing_fps'] = frame_number / processing_time if processing_time > 0 else 0
            self.stats['avg_detections_per_frame'] = (
                self.stats['total_detections'] / frame_number if frame_number > 0 else 0
            )

            # Save logs
            self._save_logs(self.source_name, duration if not self.is_stream else processing_time)

            # Print final summary
            self.logger.info("\n" + "="*70)
            self.logger.info("PROCESSING COMPLETE")
            self.logger.info("="*70)
            self.logger.info(f"Input Mode: {'RTSP Stream' if self.is_stream else 'Video File'}")
            self.logger.info(f"Total Count: {self.total_count} chocolates")
            self.logger.info(f"Frames Processed: {frame_number}")
            self.logger.info(f"Processing Time: {processing_time:.2f}s")
            self.logger.info(f"Processing FPS: {self.stats['processing_fps']:.2f}")
            self.logger.info(f"Average Detections/Frame: {self.stats['avg_detections_per_frame']:.2f}")
            if output_video_path:
                self.logger.info(f"Output Video: {output_video_path}")
            self.logger.info("="*70 + "\n")

    # Legacy method for backward compatibility
    def process_video(self, video_path: Optional[str] = None):
        """Legacy method - calls process() for backward compatibility."""
        return self.process(video_path=video_path)


def main():
    """Main entry point with full CLI support."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Chocolate Counter - Industrial CV System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Video file (headless, logs only)
  python counter.py --video videos/input/test.mp4

  # Video file with display
  python counter.py --video videos/input/test.mp4 --show

  # Video file with output video saved
  python counter.py --video videos/input/test.mp4 --save-video

  # RTSP stream (production mode, no UI)
  python counter.py --rtsp

  # RTSP stream with live display (debug mode)
  python counter.py --rtsp --show

  # RTSP with URL override
  python counter.py --rtsp-url rtsp://admin:pass@192.168.1.100:554/stream --show
        """
    )

    # Configuration
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')

    # Input source (mutually exclusive group)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--video', type=str, default=None,
                            help='Path to input video file')
    input_group.add_argument('--rtsp', action='store_true',
                            help='Use RTSP stream (URL from .env file)')

    # RTSP URL override
    parser.add_argument('--rtsp-url', type=str, default=None,
                       help='RTSP URL (overrides .env, implies --rtsp)')

    # Output control flags
    parser.add_argument('--show', action='store_true',
                       help='Show real-time annotated frames (cv2.imshow)')
    parser.add_argument('--save-video', action='store_true',
                       help='Save annotated output video (video mode only)')

    args = parser.parse_args()

    # Determine RTSP mode
    use_rtsp = args.rtsp or (args.rtsp_url is not None)

    # Initialize counter with runtime flags
    counter = ChocolateCounter(
        config_path=args.config,
        show_video=args.show if args.show else None,
        save_video=args.save_video if args.save_video else None
    )

    # Process
    counter.process(
        video_path=args.video,
        rtsp_url=args.rtsp_url,
        use_rtsp=use_rtsp
    )


if __name__ == "__main__":
    main()
