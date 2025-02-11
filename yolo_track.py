import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from PIL import Image
from types import SimpleNamespace
from src.core import YAMLConfig  # Your custom config loader
from ultralytics import YOLO

# Paths
VIDEOS_DIR = "/home/veesion/People-detector/videos"
OUTPUT_DIR = "/home/veesion/People-detector/tracks"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists

# Class names from your dataset
CLASS_NAMES = {
    0: "Backpack",
    1: "Handbag",
    2: "Tote Bag",
    3: "Banana Bag / Satchel",
    4: "Shopping Cart Bag",
    5: "Shopping Basket",
    6: "Grocery Bag",
    7: "Fruit and Vegetable Bag (plastic or paper)",
    8: "Sports Bag",
    9: "Cooler Bag",
    10: "Shop's Grocery Cart",
    11: "Travelling Bag",
    12: "Shop's Trolley",
}


def process_video(video_path, output_path):
    """Process a single video and save tracking data as JSON"""
    cap = cv2.VideoCapture(video_path)
    model = YOLO("heavy_best_yolo.pth")
    tracking_data = {}  # {frame_idx: {track_id: (box, conf)}}
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tracked_objects = model.track(frame, persist=True)
        if tracked_objects:
            frame_tracks = {}
            for track in tracked_objects:
                x1, y1, x2, y2, track_id, conf, cls, _ = track
                frame_tracks[int(track_id)] = (
                    [x1, y1, x2, y2],  # Bounding box
                    float(conf),  # Confidence
                    int(cls),
                )
            print(frame_id, list(frame_tracks))
            tracking_data[frame_id] = frame_tracks
        frame_id += 1

    cap.release()

    # Save tracking data
    with open(output_path, "wb") as f:
        pickle.dump(tracking_data, f)

    print(f"Tracking complete. Output saved as {output_path}")


import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Object Tracker")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to YAML config"
    )
    parser.add_argument(
        "-r", "--resume", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device (cpu/cuda)"
    )

    args = parser.parse_args()

    # Process all videos
    for video_file in os.listdir(VIDEOS_DIR):
        if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue  # Skip non-video files
        # if "1f9ff" not in video_file:
        #     continue

        video_path = os.path.join(VIDEOS_DIR, video_file)
        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(video_file)[0]}.pkl")

        print(f"Processing {video_file}...")
        process_video(video_path, output_path, args.device)

    print("All videos processed.")
