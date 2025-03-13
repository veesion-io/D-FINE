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
import time
import contextlib
import collections
from collections import OrderedDict
import tensorrt as trt
import pickle
import boto3
from io import BytesIO

# Paths

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
"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self):
        self.total = 0

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start

    def reset(self):
        self.total = 0

    def time(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()


class TRTInference(object):
    def __init__(
        self,
        engine_path,
        device="cuda:0",
        backend="torch",
        max_batch_size=1,
        verbose=False,
    ):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size

        self.logger = (
            trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
        )

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.bindings = self.get_bindings(
            self.engine, self.context, self.max_batch_size, self.device
        )
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        self.time_profile = TimeProfiler()

    def load_engine(self, path):
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(
        self, engine, context, max_batch_size=32, device=None
    ) -> OrderedDict:
        Binding = collections.namedtuple(
            "Binding", ("name", "dtype", "shape", "data", "ptr")
        )
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_torch(self, blob):
        for n in self.input_names:
            if blob[n].dtype is not self.bindings[n].data.dtype:
                blob[n] = blob[n].to(dtype=self.bindings[n].data.dtype)
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape)
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)

            assert self.bindings[n].data.dtype == blob[n].dtype, (
                "{} dtype mismatch".format(n)
            )

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs

    def __call__(self, blob):
        if self.backend == "torch":
            return self.run_torch(blob)
        else:
            raise NotImplementedError("Only 'torch' backend is implemented.")

    def synchronize(self):
        if self.backend == "torch" and torch.cuda.is_available():
            torch.cuda.synchronize()


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


def load_pytorch_model(config_path, checkpoint_path, device="cpu"):
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
        if not len(detections):
            self.xywh = np.zeros((0, 4))
            self.conf = np.array([])
            self.cls = np.array([])
        else:
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


def process_image(trt_mode, frame, model, transforms, device):
    # Convert frame to PIL image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    w, h = frame_pil.size
    orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=device)
    img_tensor = transforms(frame_pil).unsqueeze(0).to(device)
    if trt_mode:
        blob = {
            "images": img_tensor,
            "orig_target_sizes": orig_size,
        }
        output = model(blob)
        return output["labels"], output["boxes"], output["scores"]
    else:
        with torch.no_grad():
            return model(img_tensor, orig_size)


def process_video(model, video_path, device="cuda:0", conf_thresh=0.6, trt_mode=True):
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

        labels, boxes, scores = process_image(
            trt_mode, frame, model, transforms, device
        )
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
        if len(frame_tracks):
            tracking_data[frame_id] = frame_tracks
        frame_id += 1

    cap.release()

    return tracking_data


def list_s3_videos(s3_client, bucket, prefix):
    """List all video files in the given S3 bucket and prefix."""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in response:
        return []

    return [
        obj["Key"]
        for obj in response["Contents"]
        if obj["Key"].lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]


def download_video_from_s3(s3_client, bucket, video_s3_path, local_path):
    """Download a video file from S3 into a BytesIO buffer."""
    s3_client.download_file(bucket, video_s3_path, local_path)
    return os.path.exists(local_path)


def upload_results_to_s3(s3_client, bucket, output_key, data):
    """Upload tracking results (pickle) to S3."""
    output_buffer = BytesIO()
    pickle.dump(data, output_buffer)
    output_buffer.seek(0)
    s3_client.put_object(Bucket=bucket, Key=output_key, Body=output_buffer.getvalue())


def process_videos(
    s3_client,
    model,
    bucket,
    s3_output_bucket,
    output_prefix,
    rank,
    world_size,
    device,
    trt_mode,
):
    """Process a subset of videos assigned to this worker."""
    with open("val_2025-02-01_to_2025-02-28_100.json", "r") as f:
        dataset = json.load(f)

    videos_names = sorted(list(dataset))

    # Distribute workload across workers
    l = np.arange(len(videos_names))
    np.random.seed(42)
    np.random.shuffle(l)
    videos_names = [
        videos_names[l[i]] for i in range(rank, len(videos_names), world_size)
    ]
    if not videos_names:
        print(f"Worker {rank}: No videos to process.")
        return

    for video_name in videos_names:
        output_key = (
            f"{output_prefix}/{os.path.splitext(os.path.basename(video_name))[0]}.pkl"
        )
        try:
            s3_client.head_object(Bucket=s3_output_bucket, Key=output_key)
            continue  # Skip processing
        except s3_client.exceptions.ClientError:
            pass  # File doesn't exist, proceed with processing

        # Download and process video
        video_s3_path = dataset[video_name]["cloud_url"]
        local_path = f"/tmp/{video_name}"
        result = download_video_from_s3(s3_client, bucket, video_s3_path, local_path)
        if not result:
            continue
        tracking_data = process_video(model, local_path, device, trt_mode=trt_mode)

        # Upload results
        upload_results_to_s3(s3_client, s3_output_bucket, output_key, tracking_data)

    print(f"Worker {rank}: Finished processing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Custom Object Tracker (Distributed Execution)"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/dfine/custom/objects365/dfine_hgnetv2_x_obj2custom.yml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default="dfine_bag_detector.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "-trt", "--tensorrt", action="store_true", help="Tensorrt inference"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device (cpu/cuda)"
    )
    parser.add_argument("--s3-bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument(
        "--s3-output-prefix",
        type=str,
        required=True,
        help="S3 output prefix (directory)",
    )
    parser.add_argument(
        "--s3-output-bucket",
        type=str,
        required=True,
        help="S3 output bucket",
    )
    parser.add_argument(
        "--rank", type=int, required=True, help="Worker rank (0-based index)"
    )
    parser.add_argument(
        "--world-size", type=int, required=True, help="Total number of workers"
    )

    args = parser.parse_args()

    s3_client = boto3.client("s3")

    # Load model
    if args.tensorrt:
        model = TRTInference("dfine_bag_detector.trt")
    else:
        model = load_pytorch_model(args.config, args.resume, args.device)

    process_videos(
        s3_client,
        model,
        args.s3_bucket,
        args.s3_output_bucket,
        args.s3_output_prefix,
        args.rank,
        args.world_size,
        args.device,
        args.tensorrt,
    )
    print("All videos processed.")
