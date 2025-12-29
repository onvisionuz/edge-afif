"""
Run chocolate detector on new images/videos
"""

from ultralytics import YOLO
import cv2
from pathlib import Path

# ============ CONFIGURATION ============
MODEL = "runs/detect/chocolate_detector/weights/best.pt"
CONFIDENCE = 0.5  # Detection confidence threshold
# =======================================

def detect_image(image_path):
    """Detect chocolates in a single image."""
    model = YOLO(MODEL)
    results = model(image_path, conf=CONFIDENCE)

    num_chocolates = len(results[0].boxes)
    print(f"✅ Detected {num_chocolates} chocolates")

    # Show result
    annotated = results[0].plot()
    cv2.imshow('Detection', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_folder(folder_path, output_folder='detections'):
    """Detect chocolates in all images in a folder."""
    model = YOLO(MODEL)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    images = list(Path(folder_path).glob('*.jpg'))

    for img in images:
        results = model(img, conf=CONFIDENCE)
        num = len(results[0].boxes)
        print(f"{img.name}: {num} chocolates")

        # Save result
        annotated = results[0].plot()
        cv2.imwrite(str(output_path / img.name), annotated)

    print(f"\n✅ Results saved to {output_folder}/")

def detect_video(video_path, output_path='output.mp4'):
    """Detect chocolates in video."""
    model = YOLO(MODEL)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONFIDENCE, verbose=False)
        annotated = results[0].plot()

        num = len(results[0].boxes)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {num} chocolates")

        out.write(annotated)

    cap.release()
    out.release()
    print(f"✅ Video saved to {output_path}")

# Usage examples:
if __name__ == "__main__":

    # Detect in single image
    # detect_image("")

    # OR detect in folder
    # detect_folder("test_images/", "results/")

    # OR detect in video
    detect_video("data_prep/video_dataset/D02_20251215151059.mp4", "output.mp4")