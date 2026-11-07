"""Shared data contracts exchanged between capture, inference, alerting, and publisher threads."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid

import numpy as np


@dataclass(slots=True)
class PersonPose:
    """Detected and tracked person with bounding box and 17 COCO keypoints."""

    track_id: int
    bbox_xywh: np.ndarray  # shape (4,) float32 [x_center, y_center, width, height]
    keypoints_xyc: np.ndarray  # shape (17, 3) float32 [x, y, conf]
    score: float


@dataclass(slots=True)
class FramePacket:
    """Single downsampled video frame emitted by the Capture thread."""

    frame_idx: int
    raw_frame_idx: int
    timestamp: float
    frame: np.ndarray
    width: int
    height: int
    fps: float


@dataclass(slots=True)
class PoseWindow:
    """A 12-frame sliding window of tracked poses for a single person ID."""

    track_id: int
    start_frame: int
    end_frame: int
    frame_to_kps: dict[int, np.ndarray]  # frame_idx -> (17, 3) float32 [x, y, conf]
    timestamp: float = 0.0
    vid_res: tuple[int, int] = (1920, 1080)


# Alias for backward compatibility with existing buffer tests
TrackWindow = PoseWindow


@dataclass(slots=True)
class ScoreEvent:
    """Window-level anomaly score emitted by the Inference thread to the Alerting thread."""

    frame_idx: int
    timestamp: float
    track_id: int
    raw_score: float
    threshold: float
    poses: list[PersonPose] = field(default_factory=list)
    annotated_frame: np.ndarray | None = None


@dataclass(slots=True)
class AlertEvent:
    """Confirmed anomaly alert event passed to the Outbox / MQTT Publisher thread."""

    event_id: str
    ts: str
    cam_id: str
    track_id: int
    score: float
    threshold: float
    clip_path: str

    @classmethod
    def create(
        cls,
        cam_id: str,
        track_id: int,
        score: float,
        threshold: float,
        clip_path: str,
        event_id: str | None = None,
        ts: str | None = None,
    ) -> AlertEvent:
        """Create an AlertEvent with a standard UUID4 `event_id` and ISO-8601 UTC timestamp."""
        return cls(
            event_id=event_id or str(uuid.uuid4()),
            ts=ts or datetime.now(timezone.utc).isoformat(),
            cam_id=str(cam_id),
            track_id=int(track_id),
            score=float(score),
            threshold=float(threshold),
            clip_path=str(clip_path),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize AlertEvent to the strict MQTT JSON payload dictionary."""
        return {
            "event_id": self.event_id,
            "ts": self.ts,
            "cam_id": self.cam_id,
            "track_id": int(self.track_id),
            "score": round(float(self.score), 6),
            "threshold": round(float(self.threshold), 6),
            "clip_path": self.clip_path,
        }

