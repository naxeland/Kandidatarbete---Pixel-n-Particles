"""
capture_images.py

Captures photos using a Raspberry Pi camera module on an NVIDIA Jetson Nano
and saves them to the images/ folder.

Requirements:
    - OpenCV (cv2) with GStreamer support (pre-installed on Jetson Nano JetPack)
    - Camera module connected via CSI port

Usage:
    python capture_images.py                   # capture one photo
    python capture_images.py --count 5         # capture 5 photos
    python capture_images.py --interval 2      # capture continuously every 2 seconds
    python capture_images.py --count 10 --interval 1  # 10 photos, 1 second apart
"""

import cv2
import os
import time
import argparse
from datetime import datetime


IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

# GStreamer pipeline for Raspberry Pi Camera on Jetson Nano (CSI camera)
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )


def capture_photo(cap, filename):
    """Capture a single frame and save it to the images folder."""
    ret, frame = cap.read()
    if not ret or frame is None:
        print("ERROR: Failed to capture frame from camera.")
        return False

    filepath = os.path.join(IMAGES_DIR, filename)
    success = cv2.imwrite(filepath, frame)
    if success:
        print(f"Saved: {filepath}")
    else:
        print(f"ERROR: Failed to write image to {filepath}")
    return success


def main():
    parser = argparse.ArgumentParser(description="Capture images from Raspberry Pi Camera on Jetson Nano")
    parser.add_argument("--count", type=int, default=1, help="Number of photos to capture (default: 1)")
    parser.add_argument("--interval", type=float, default=0.0, help="Seconds between captures (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width in pixels (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Capture height in pixels (default: 720)")
    parser.add_argument("--prefix", type=str, default="capture", help="Filename prefix (default: capture)")
    args = parser.parse_args()

    os.makedirs(IMAGES_DIR, exist_ok=True)

    pipeline = gstreamer_pipeline(
        capture_width=args.width,
        capture_height=args.height,
        display_width=args.width,
        display_height=args.height,
    )

    print(f"Opening camera with GStreamer pipeline...")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("ERROR: Could not open camera. Check that:")
        print("  - The Raspberry Pi camera is connected to the CSI port")
        print("  - The camera is enabled (sudo nvpmodel -m 0 / check /boot/extlinux/extlinux.conf)")
        print("  - OpenCV was built with GStreamer support")
        return

    print(f"Camera opened. Capturing {args.count} photo(s) to '{IMAGES_DIR}/'")

    try:
        for i in range(args.count):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
            filename = f"{args.prefix}_{timestamp}.jpg"
            capture_photo(cap, filename)

            if args.interval > 0 and i < args.count - 1:
                time.sleep(args.interval)
    finally:
        cap.release()
        print("Camera released.")


if __name__ == "__main__":
    main()
