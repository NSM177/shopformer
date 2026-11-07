"""Step 4: Per-ID sliding window pose buffer (12-frame window with missing-frame tolerance)."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from shoplift_iot.vision.pose_track import PersonPose


@dataclass(slots=True)
class TrackWindow:
    """A 12-frame temporal window of poses for a single tracked person."""

    track_id: int
    start_frame: int
    end_frame: int
    frame_to_kps: dict[int, np.ndarray]  # frame_idx -> (17, 3) float32 [x, y, conf]


class PoseBuffer:
    """Sliding window buffer maintaining the latest pose frames per track ID."""

    def __init__(
        self,
        window_size: int = 12,
        max_missing_frames: int = 2,
        stale_timeout_frames: int = 45,
    ) -> None:
        """Initialize per-ID sliding window buffer.

        Args:
            window_size: Target temporal window length in frames (default 12).
            max_missing_frames: Allowed missing frames inside a window of length `window_size`.
            stale_timeout_frames: Number of frames of inactivity before evicting a track ID.
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        self.window_size = int(window_size)
        self.max_missing_frames = int(max_missing_frames)
        self.stale_timeout_frames = int(stale_timeout_frames)
        self._buffers: dict[int, OrderedDict[int, np.ndarray]] = {}
        self._last_seen: dict[int, int] = {}

    def update(self, frame_idx: int, poses: list[PersonPose]) -> list[TrackWindow]:
        """Insert newly detected poses at `frame_idx` and return ready 12-frame windows."""
        ready_windows: list[TrackWindow] = []

        for pose in poses:
            tid = int(pose.track_id)
            if tid not in self._buffers:
                self._buffers[tid] = OrderedDict()
            buf = self._buffers[tid]
            buf[int(frame_idx)] = pose.keypoints_xyc.astype(np.float32)
            self._last_seen[tid] = int(frame_idx)

            # Prune frames older than the current 12-frame span
            min_keep_frame = int(frame_idx) - self.window_size + 1
            while buf and next(iter(buf.keys())) < min_keep_frame:
                buf.popitem(last=False)

            # Check if the current span [frame_idx - window_size + 1, frame_idx] has enough frames
            # and starts at or near min_keep_frame (so linear interpolation doesn't purely extrapolate)
            min_required = self.window_size - self.max_missing_frames
            if len(buf) >= min_required:
                first_f = next(iter(buf.keys()))
                last_f = next(reversed(buf.keys()))
                span = last_f - first_f + 1
                if span <= self.window_size and (self.window_size - span) <= self.max_missing_frames:
                    ready_windows.append(
                        TrackWindow(
                            track_id=tid,
                            start_frame=min_keep_frame,
                            end_frame=int(frame_idx),
                            frame_to_kps=dict(buf),
                        )
                    )

        self._evict_stale(frame_idx)
        return ready_windows

    def get_track_frames(self, track_id: int) -> dict[int, np.ndarray]:
        """Return currently buffered frames for `track_id`."""
        return dict(self._buffers.get(track_id, {}))

    def _evict_stale(self, current_frame_idx: int) -> None:
        stale_ids = [
            tid
            for tid, last_f in self._last_seen.items()
            if current_frame_idx - last_f > self.stale_timeout_frames
        ]
        for tid in stale_ids:
            self._buffers.pop(tid, None)
            self._last_seen.pop(tid, None)


def load_buffer_from_config(config_path: Path) -> PoseBuffer:
    """Create PoseBuffer from YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prep = cfg["preprocess"]
    return PoseBuffer(
        window_size=int(prep["window_size"]),
        max_missing_frames=int(prep["max_missing_frames"]),
    )


def main() -> None:
    """CLI smoke test for PoseBuffer."""
    parser = argparse.ArgumentParser(description="Smoke test PoseBuffer with synthetic frames.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"), help="Path to YAML config."
    )
    args = parser.parse_args()

    buf = load_buffer_from_config(args.config)
    emitted = 0
    for f_idx in range(15):
        if f_idx == 5:
            # Simulate 1 dropped frame at index 5
            continue
        dummy_pose = PersonPose(
            track_id=1,
            bbox_xywh=np.array([500.0, 400.0, 100.0, 250.0], dtype=np.float32),
            keypoints_xyc=np.ones((17, 3), dtype=np.float32) * float(f_idx + 1),
            score=0.9,
        )
        windows = buf.update(f_idx, [dummy_pose])
        emitted += len(windows)
    print(f"[PoseBuffer] Emitted {emitted} valid 12-frame windows (with 1 missing frame tolerated).")


if __name__ == "__main__":
    main()

