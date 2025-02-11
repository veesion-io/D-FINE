import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import json
from PIL import Image
import torchvision.transforms as T
from ultralytics.trackers.byte_tracker import BYTETracker
from types import SimpleNamespace
from src.core import YAMLConfig  # Your custom config loader

# Paths
VIDEOS_DIR = "/home/veesion/Bag-detector/videos"
OUTPUT_DIR = "/home/veesion/Bag-detector/tracks"
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


class Model(nn.Module):
    """Exact model structure from the working script"""

    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


def load_custom_model(config_path, checkpoint_path, device="cpu"):
    """Load model EXACTLY like the working script"""
    cfg = YAMLConfig(config_path, resume=checkpoint_path)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("ema", checkpoint).get("module", checkpoint.get("model"))

    cfg.model.load_state_dict(state)
    return Model(cfg).to(device)


class DetectionResults:
    """Ensure BYTETracker gets correctly formatted detections"""

    def __init__(self, detections):
        xyxy = detections[:, :4]
        self.xywh = self.xyxy_to_xywh(xyxy)
        self.conf = detections[:, 4]
        self.cls = detections[:, 5]

    @staticmethod
    def xyxy_to_xywh(xyxy):
        xywh = np.zeros_like(xyxy)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2  # center_x
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2  # center_y
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]  # width
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]  # height
        return xywh


def process_video(model, video_path, output_path, device="cuda:0", conf_thresh=0.6):
    """Process a single video and save tracking data as JSON"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize BYTETracker
    tracker_args = SimpleNamespace(
        track_buffer=30,
        match_thresh=0.7,
        track_high_thresh=0.25,
        new_track_thresh=0.25,
        track_low_thresh=0.1,
        fuse_score=True,
    )
    tracker = BYTETracker(tracker_args, frame_rate=fps)

    # Input preprocessing
    transforms = T.Compose([T.Resize((960, 960)), T.ToTensor()])

    tracking_data = {}  # {frame_idx: {track_id: (box, conf)}}

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = frame_pil.size

        # Ensure inputs are on the correct device
        orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=device)
        img_tensor = transforms(frame_pil).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            labels, boxes, scores = model(img_tensor, orig_size)

        detections = []
        for i in range(len(labels[0])):
            if scores[0][i].item() >= conf_thresh:
                detections.append(
                    [
                        boxes[0][i][0].item(),
                        boxes[0][i][1].item(),
                        boxes[0][i][2].item(),
                        boxes[0][i][3].item(),
                        scores[0][i].item(),
                        labels[0][i].item(),
                    ]
                )

        detections = np.array(detections, dtype=np.float32)

        if detections.shape[0] > 0:
            structured_detections = DetectionResults(detections)
            tracked_objects = tracker.update(structured_detections)

            # Store tracking results
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

    # Load model
    custom_model = load_custom_model(args.config, args.resume, args.device)

    # Process all videos
    for video_file in os.listdir(VIDEOS_DIR):
        if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue  # Skip non-video files
        # if "1f9ff" not in video_file:
        #     continue

        video_path = os.path.join(VIDEOS_DIR, video_file)
        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(video_file)[0]}.pkl")

        print(f"Processing {video_file}...")
        process_video(custom_model, video_path, output_path, args.device)

    print("All videos processed.")
