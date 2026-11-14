"""GD2: YOLO + ByteTrack per-person cropping (~15% margin), temporal buffer, and 16-frame VideoMAE clip builder."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Mapping

import cv2
import numpy as np

from shoplift_iot.capture.source import VideoSource
from shoplift_iot.config import PROJECT_ROOT, AppConfig, load_app_config
from shoplift_iot.contracts import PersonPose
from shoplift_iot.vision.pose_track import PoseTracker, load_tracker_from_config

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from videomae_lite import resize_rgb, uniform_indices  # noqa: E402


@dataclass(slots=True)
class PersonClipWindow:
    """A 16-frame uniformly sampled RGB clip for a single tracked person (`videomae_lite.Predictor` format)."""

    track_id: int
    start_ts: float
    end_ts: float
    frames_rgb: np.ndarray  # uint8 [n_frames, size, size, 3] (RGB), default [16, 224, 224, 3]
    bbox_xywh: np.ndarray  # float32 [x_center, y_center, width, height]
    raw_crops_bgr: list[np.ndarray] = field(default_factory=list)


def expand_bbox_xyxy(
    bbox_xywh: np.ndarray,
    frame_w: int,
    frame_h: int,
    margin_ratio: float = 0.15,
) -> tuple[int, int, int, int]:
    """Convert `[x_center, y_center, w, h]` to `[x1, y1, x2, y2]` expanded by `margin_ratio` (~15%) and clamped."""
    xc, yc, bw, bh = (
        float(bbox_xywh[0]),
        float(bbox_xywh[1]),
        max(2.0, float(bbox_xywh[2])),
        max(2.0, float(bbox_xywh[3])),
    )
    half_w = bw * (0.5 + float(margin_ratio))
    half_h = bh * (0.5 + float(margin_ratio))

    x1 = max(0, min(frame_w - 1, int(round(xc - half_w))))
    y1 = max(0, min(frame_h - 1, int(round(yc - half_h))))
    x2 = max(x1 + 1, min(frame_w, int(round(xc + half_w))))
    y2 = max(y1 + 1, min(frame_h, int(round(yc + half_h))))
    return x1, y1, x2, y2


def crop_person_with_margin(
    frame_bgr: np.ndarray,
    bbox_xywh: np.ndarray,
    margin_ratio: float = 0.15,
) -> np.ndarray:
    """Crop the region around a tracked person's bounding box with ~15% padding on each side."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = expand_bbox_xyxy(bbox_xywh, frame_w=w, frame_h=h, margin_ratio=margin_ratio)
    return np.ascontiguousarray(frame_bgr[y1:y2, x1:x2])


def build_videomae_clip_from_crops(
    crops_bgr: list[np.ndarray],
    n_frames: int = 16,
    size: int = 224,
) -> np.ndarray:
    """Sample `n_frames` uniformly from `crops_bgr` and resize/convert to `uint8 [n_frames, size, size, 3]` RGB."""
    if not crops_bgr:
        raise ValueError("Cannot build VideoMAE clip from an empty list of crops.")
    sel = uniform_indices(len(crops_bgr), n_frames=n_frames)
    rgb_list = [resize_rgb(crops_bgr[int(i)], size=size) for i in sel]
    return np.stack(rgb_list, axis=0).astype(np.uint8)


class PersonClipBuffer:
    """Per-track frame buffer emitting 16-frame RGB clips compatible with `videomae_lite.Predictor`."""

    def __init__(
        self,
        clip_sec: float = 3.0,
        hop_sec: float = 1.0,
        n_frames: int = 16,
        size: int = 224,
        margin_ratio: float = 0.15,
        min_span_ratio: float = 0.8,
        min_frames: int = 8,
        stale_sec: float = 5.0,
    ) -> None:
        if clip_sec <= 0:
            raise ValueError("clip_sec must be positive.")
        if hop_sec <= 0:
            raise ValueError("hop_sec must be positive.")
        if n_frames <= 0:
            raise ValueError("n_frames must be positive.")
        if size <= 0:
            raise ValueError("size must be positive.")

        self.clip_sec = float(clip_sec)
        self.hop_sec = float(hop_sec)
        self.n_frames = int(n_frames)
        self.size = int(size)
        self.margin_ratio = float(margin_ratio)
        self.min_span_ratio = float(min_span_ratio)
        self.min_frames = max(1, int(min_frames))
        self.stale_sec = max(0.5, float(stale_sec))

        self._buffers: dict[int, deque[tuple[float, np.ndarray, np.ndarray]]] = {}
        self._next_eval: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}
        self._emitted_tracks: set[int] = set()

    def update(
        self,
        timestamp: float,
        frame_bgr: np.ndarray,
        poses: list[PersonPose],
    ) -> list[PersonClipWindow]:
        """Crop each tracked person from `frame_bgr`, update their buffer, and return ready 16-frame clips."""
        ts = float(timestamp)
        ready: list[PersonClipWindow] = []

        for pose in poses:
            tid = int(pose.track_id)
            crop = crop_person_with_margin(frame_bgr, pose.bbox_xywh, margin_ratio=self.margin_ratio)
            buf = self._buffers.setdefault(tid, deque())
            buf.append((ts, crop, pose.bbox_xywh.copy()))
            self._last_seen[tid] = ts

            while buf and (ts - buf[0][0]) > self.clip_sec + 1e-6:
                buf.popleft()

            if tid not in self._next_eval:
                self._next_eval[tid] = buf[0][0] + self.clip_sec * self.min_span_ratio

            span = buf[-1][0] - buf[0][0]
            if (
                len(buf) >= self.min_frames
                and span + 1e-6 >= self.clip_sec * self.min_span_ratio
                and ts + 1e-6 >= self._next_eval[tid]
            ):
                self._next_eval[tid] = ts + self.hop_sec
                raw_crops = [item[1] for item in buf]
                frames_rgb = build_videomae_clip_from_crops(
                    raw_crops, n_frames=self.n_frames, size=self.size
                )
                ready.append(
                    PersonClipWindow(
                        track_id=tid,
                        start_ts=buf[0][0],
                        end_ts=buf[-1][0],
                        frames_rgb=frames_rgb,
                        bbox_xywh=pose.bbox_xywh.copy(),
                        raw_crops_bgr=raw_crops,
                    )
                )
                self._emitted_tracks.add(tid)

        self._evict_stale(ts)
        return ready

    def flush_unemitted(self, min_frames: int | None = None) -> list[PersonClipWindow]:
        """Emit a single clip for any buffered track that had at least `min_frames` but ended before `clip_sec`."""
        req = self.min_frames if min_frames is None else max(1, int(min_frames))
        flushed: list[PersonClipWindow] = []
        for tid, buf in list(self._buffers.items()):
            if tid in self._emitted_tracks or len(buf) < req:
                continue
            raw_crops = [item[1] for item in buf]
            frames_rgb = build_videomae_clip_from_crops(
                raw_crops, n_frames=self.n_frames, size=self.size
            )
            flushed.append(
                PersonClipWindow(
                    track_id=tid,
                    start_ts=buf[0][0],
                    end_ts=buf[-1][0],
                    frames_rgb=frames_rgb,
                    bbox_xywh=buf[-1][2].copy(),
                    raw_crops_bgr=raw_crops,
                )
            )
            self._emitted_tracks.add(tid)
        return flushed

    def get_track_buffer_len(self, track_id: int) -> int:
        """Return number of buffered frames currently held for `track_id`."""
        buf = self._buffers.get(int(track_id))
        return len(buf) if buf is not None else 0

    def _evict_stale(self, current_ts: float) -> None:
        stale_ids = [
            tid
            for tid, last_t in self._last_seen.items()
            if (current_ts - last_t) > self.stale_sec
        ]
        for tid in stale_ids:
            self._buffers.pop(tid, None)
            self._next_eval.pop(tid, None)
            self._last_seen.pop(tid, None)


class PersonClipExtractor:
    """Shared real-time and offline pipeline: YOLO + ByteTrack -> per-person crops -> 16-frame VideoMAE clips."""

    def __init__(
        self,
        tracker: PoseTracker,
        clip_buffer: PersonClipBuffer,
    ) -> None:
        self.tracker = tracker
        self.clip_buffer = clip_buffer

    @classmethod
    def from_config(
        cls,
        config_or_path: Path | str | AppConfig | Mapping[str, Any] | None = None,
        yolo_instance: Any | None = None,
        clip_sec_override: float | None = None,
        hop_sec_override: float | None = None,
        n_frames_override: int | None = None,
        size_override: int | None = None,
    ) -> PersonClipExtractor:
        """Build `PersonClipExtractor` from `configs/default.yaml` (no hardcoded paths or thresholds)."""
        cfg = load_app_config(config_or_path)
        tracker = load_tracker_from_config(cfg, yolo_instance=yolo_instance)
        pc_cfg = cfg.person_clips
        clip_buffer = PersonClipBuffer(
            clip_sec=float(clip_sec_override if clip_sec_override is not None else pc_cfg.get("clip_sec", 3.0)),
            hop_sec=float(hop_sec_override if hop_sec_override is not None else pc_cfg.get("hop_sec", 1.0)),
            n_frames=int(n_frames_override if n_frames_override is not None else pc_cfg.get("num_frames", 16)),
            size=int(size_override if size_override is not None else pc_cfg.get("frame_size", 224)),
            margin_ratio=float(pc_cfg.get("margin_ratio", 0.15)),
            min_span_ratio=float(pc_cfg.get("min_span_ratio", 0.8)),
            min_frames=int(pc_cfg.get("min_frames", 8)),
            stale_sec=float(pc_cfg.get("stale_sec", 5.0)),
        )
        return cls(tracker=tracker, clip_buffer=clip_buffer)

    def process_frame(
        self,
        timestamp: float,
        frame_bgr: np.ndarray,
    ) -> tuple[list[PersonPose], list[PersonClipWindow]]:
        """Run YOLO + ByteTrack on a single frame and return `(poses, ready_person_clips)`."""
        poses = self.tracker.process_frame(frame_bgr)
        windows = self.clip_buffer.update(timestamp, frame_bgr, poses)
        return poses, windows

    def extract_from_video(
        self,
        video_path: Path | str,
        target_fps: float = 15.0,
        max_frames: int | None = None,
        flush_short_tracks: bool = True,
    ) -> list[PersonClipWindow]:
        """Run offline extraction on a recorded video file using the exact same real-time logic."""
        emitted: list[PersonClipWindow] = []
        with VideoSource(source=video_path, target_fps=target_fps, auto_reconnect=False) as reader:
            for pkt in reader:
                if max_frames is not None and pkt.frame_idx >= max_frames:
                    break
                _, wins = self.process_frame(pkt.timestamp, pkt.frame)
                emitted.extend(wins)

        if flush_short_tracks:
            emitted.extend(self.clip_buffer.flush_unemitted())
        return emitted


def save_person_clip_mp4(
    window: PersonClipWindow,
    out_path: Path | str,
    fps: float = 15.0,
    use_uniform_rgb: bool = False,
) -> Path:
    """Save a `PersonClipWindow` to an `.mp4` file for VideoMAE training or inspection."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if use_uniform_rgb or not window.raw_crops_bgr:
        h, w = window.frames_rgb.shape[1:3]
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        for rgb in window.frames_rgb:
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        writer.release()
        return target

    size = window.frames_rgb.shape[1]
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (size, size))
    for crop_bgr in window.raw_crops_bgr:
        resized = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_AREA)
        writer.write(resized)
    writer.release()
    return target

