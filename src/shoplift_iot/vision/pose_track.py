"""Steps 2 & 3: YOLOv8-pose person detection (COCO17) + ByteTrack multi-person tracking."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from shoplift_iot.capture.source import load_source_from_config
from shoplift_iot.config import PROJECT_ROOT, AppConfig, load_app_config
from shoplift_iot.contracts import PersonPose


def _resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


class PoseTracker:
    """Wrapper around Ultralytics YOLOv8-pose with integrated ByteTrack tracker."""

    def __init__(
        self,
        model_name: str = "models/pretrained/yolov8n-pose.pt",
        tracker_config: str = "bytetrack.yaml",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda:0",
        yolo_instance: Any | None = None,
    ) -> None:
        """Initialize YOLOv8-pose + ByteTrack tracker."""
        candidate = PROJECT_ROOT / model_name
        self.model_name = str(candidate) if candidate.exists() else model_name
        self.tracker_config = tracker_config
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.device = _resolve_device(device)
        self._model = yolo_instance

    @property
    def model(self) -> Any:
        """Lazy-load Ultralytics YOLO model when needed."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
        return self._model

    def process_frame(self, frame: np.ndarray) -> list[PersonPose]:
        """Run YOLOv8-pose + ByteTrack on a single BGR frame and return tracked poses."""
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        res = results[0]
        if res.boxes is None or res.boxes.id is None:
            return []

        track_ids = res.boxes.id.int().cpu().numpy()
        bboxes = res.boxes.xywh.float().cpu().numpy()
        confs = res.boxes.conf.float().cpu().numpy()
        classes = (
            res.boxes.cls.int().cpu().numpy()
            if getattr(res.boxes, "cls", None) is not None
            else np.zeros(len(track_ids), dtype=np.int32)
        )
        kps_data = (
            res.keypoints.data.float().cpu().numpy()
            if res.keypoints is not None
            else np.zeros((len(track_ids), 17, 3), dtype=np.float32)
        )

        poses: list[PersonPose] = []
        for idx, tid in enumerate(track_ids):
            if int(classes[idx]) != 0:
                continue
            kp = (
                kps_data[idx]
                if idx < len(kps_data)
                else np.zeros((17, 3), dtype=np.float32)
            )
            if kp.shape == (17, 2):
                kp = np.concatenate([kp, np.ones((17, 1), dtype=np.float32)], axis=-1)
            poses.append(
                PersonPose(
                    track_id=int(tid),
                    bbox_xywh=bboxes[idx].astype(np.float32),
                    keypoints_xyc=kp.astype(np.float32),
                    score=float(confs[idx]),
                )
            )
        return poses


def load_tracker_from_config(
    config_or_path: Path | str | AppConfig | Mapping[str, Any],
    yolo_instance: Any | None = None,
) -> PoseTracker:
    """Create PoseTracker from application configuration."""
    cfg = load_app_config(config_or_path)
    pt_cfg = cfg.pose_track
    return PoseTracker(
        model_name=str(pt_cfg["yolo_model"]),
        tracker_config=str(pt_cfg["tracker_config"]),
        conf_threshold=float(pt_cfg["conf_threshold"]),
        iou_threshold=float(pt_cfg["iou_threshold"]),
        device=str(pt_cfg["device"]),
        yolo_instance=yolo_instance,
    )


def main() -> None:
    """CLI smoke test for PoseTracker on a video source."""
    parser = argparse.ArgumentParser(description="Test YOLOv8-pose + ByteTrack on a video.")
    parser.add_argument("--source", type=str, required=True, help="Video path or camera index.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"), help="Path to YAML config."
    )
    parser.add_argument("--max-frames", type=int, default=15, help="Max frames to process.")
    args = parser.parse_args()

    reader = load_source_from_config(args.config, args.source)
    tracker = load_tracker_from_config(args.config)
    total_detections = 0
    unique_ids: set[int] = set()

    with reader:
        for pkt in reader:
            poses = tracker.process_frame(pkt.frame)
            total_detections += len(poses)
            unique_ids.update(p.track_id for p in poses)
            if pkt.frame_idx + 1 >= args.max_frames:
                break

    print(
        f"[PoseTracker] Processed {args.max_frames} frames | "
        f"Total detections: {total_detections} | Unique track IDs: {sorted(unique_ids)}"
    )


if __name__ == "__main__":
    main()

