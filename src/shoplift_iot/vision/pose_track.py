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
        model_name: str = "models/yolov8s.pt",
        tracker_config: str = "configs/bytetrack_cctv.yaml",
        conf_threshold: float = 0.15,
        iou_threshold: float = 0.45,
        device: str = "cuda:0",
        yolo_instance: Any | None = None,
        max_coast_frames: int = 15,
    ) -> None:
        """Initialize YOLOv8 + ByteTrack tracker with optional Kalman bbox coasting on short misses."""
        candidate = PROJECT_ROOT / model_name
        self.model_name = str(candidate) if candidate.exists() else model_name
        trk_candidate = PROJECT_ROOT / tracker_config
        self.tracker_config = str(trk_candidate) if trk_candidate.exists() else tracker_config
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.device = _resolve_device(device)
        self._model = yolo_instance
        self.max_coast_frames = max(0, int(max_coast_frames))
        self.last_coasted_ids: set[int] = set()
        self.reset()

    @property
    def model(self) -> Any:
        """Lazy-load Ultralytics YOLO model when needed."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
        return self._model

    def reset(self) -> None:
        """Reset ByteTrack internal state and ID counter (e.g. when starting or looping a video)."""
        try:
            from ultralytics.trackers.basetrack import BaseTrack

            BaseTrack._count = 0
        except Exception:
            pass
        if self._model is not None and getattr(self._model, "predictor", None) is not None:
            trackers = getattr(self._model.predictor, "trackers", None)
            if trackers:
                for trk in trackers:
                    if hasattr(trk, "reset"):
                        trk.reset()
                try:
                    from ultralytics.trackers.basetrack import BaseTrack

                    BaseTrack._count = 0
                except Exception:
                    pass

    def process_frame(self, frame: np.ndarray) -> list[PersonPose]:
        """Run YOLOv8 + ByteTrack on a single BGR frame and return tracked poses (including short-term Kalman predictions)."""
        self.last_coasted_ids.clear()
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            classes=[0],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        poses: list[PersonPose] = []
        active_ids: set[int] = set()

        if results:
            res = results[0]
            if res.boxes is not None and res.boxes.id is not None:
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
                    tid_int = int(tid)
                    active_ids.add(tid_int)
                    poses.append(
                        PersonPose(
                            track_id=tid_int,
                            bbox_xywh=bboxes[idx].astype(np.float32),
                            keypoints_xyc=kp.astype(np.float32),
                            score=float(confs[idx]),
                        )
                    )

        # Coasting: when tracker loses detection for a few frames, use ByteTrack's Kalman-predicted bbox
        if self.max_coast_frames > 0 and self._model is not None:
            predictor = getattr(self._model, "predictor", None)
            trackers = getattr(predictor, "trackers", None) if predictor is not None else None
            if trackers:
                trk = trackers[0]
                fh, fw = frame.shape[:2]
                cur_frame_id = int(getattr(trk, "frame_id", 0))
                for st in getattr(trk, "lost_stracks", []):
                    if not getattr(st, "is_activated", False):
                        continue
                    tid_int = int(st.track_id)
                    if tid_int in active_ids:
                        continue
                    lost_frames = cur_frame_id - int(getattr(st, "end_frame", cur_frame_id))
                    if 1 <= lost_frames <= self.max_coast_frames:
                        xywh = np.asarray(st.xywh, dtype=np.float32).copy()
                        # Clamp center and size inside frame bounds
                        xywh[2] = np.clip(xywh[2], 4.0, float(fw))
                        xywh[3] = np.clip(xywh[3], 4.0, float(fh))
                        xywh[0] = np.clip(xywh[0], xywh[2] * 0.25, float(fw) - xywh[2] * 0.25)
                        xywh[1] = np.clip(xywh[1], xywh[3] * 0.25, float(fh) - xywh[3] * 0.25)
                        self.last_coasted_ids.add(tid_int)
                        poses.append(
                            PersonPose(
                                track_id=tid_int,
                                bbox_xywh=xywh,
                                keypoints_xyc=np.zeros((17, 3), dtype=np.float32),
                                score=float(getattr(st, "score", 0.0)),
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
        max_coast_frames=int(pt_cfg.get("max_coast_frames", 15)),
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

