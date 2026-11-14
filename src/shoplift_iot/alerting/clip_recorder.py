"""In-memory ring buffer and pre/post-event video clip recorder."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class _PendingClip:
    """Active post-event frame collection session."""

    event_id: str
    out_file: Path
    frames: list[np.ndarray] = field(default_factory=list)
    remaining_post_frames: int = 0


class ClipRecorder:
    """Keep an in-memory ring buffer of recent frames and save MP4 clips around anomaly events."""

    def __init__(
        self,
        clips_dir: Path,
        target_fps: float = 15.0,
        pre_event_seconds: float = 3.0,
        post_event_seconds: float = 2.0,
        project_root: Path | None = None,
    ) -> None:
        self.clips_dir = Path(clips_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.target_fps = max(1.0, float(target_fps))
        self.pre_event_seconds = max(0.5, float(pre_event_seconds))
        self.post_event_seconds = max(0.0, float(post_event_seconds))
        self.project_root = project_root

        self.pre_frames_count = max(1, int(round(self.pre_event_seconds * self.target_fps)))
        self.post_frames_count = max(0, int(round(self.post_event_seconds * self.target_fps)))
        self._ring_buffer: deque[np.ndarray] = deque(maxlen=self.pre_frames_count)
        self._pending: dict[str, _PendingClip] = {}

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        """Append a frame to the pre-event ring buffer and advance any active post-event clips."""
        copied = frame_bgr.copy()
        self._ring_buffer.append(copied)

        completed_ids: list[str] = []
        for event_id, pending in self._pending.items():
            if pending.remaining_post_frames > 0:
                pending.frames.append(copied)
                pending.remaining_post_frames -= 1
            if pending.remaining_post_frames <= 0:
                self._write_mp4(pending.out_file, pending.frames)
                completed_ids.append(event_id)

        for event_id in completed_ids:
            self._pending.pop(event_id, None)

    def trigger_clip(self, event_id: str) -> str:
        """Save pre-event snapshot immediately and collect post-event frames around `event_id`."""
        out_file = self.clips_dir / f"{event_id}.mp4"
        pre_frames = list(self._ring_buffer)
        if not pre_frames:
            pre_frames = [np.zeros((480, 640, 3), dtype=np.uint8)]

        # Write initial pre-event clip immediately so the file exists at alert creation time
        self._write_mp4(out_file, pre_frames)

        if self.post_frames_count > 0:
            self._pending[event_id] = _PendingClip(
                event_id=event_id,
                out_file=out_file,
                frames=list(pre_frames),
                remaining_post_frames=self.post_frames_count,
            )

        return self._format_path(out_file)

    def flush_all(self) -> None:
        """Finalize and write all pending clips that are still collecting post-event frames."""
        for pending in list(self._pending.values()):
            self._write_mp4(pending.out_file, pending.frames)
        self._pending.clear()

    def _write_mp4(self, out_file: Path, frames: list[np.ndarray]) -> None:
        """Atomically write `frames` to `out_file` as an MP4 video."""
        if not frames:
            frames = [np.zeros((480, 640, 3), dtype=np.uint8)]

        h, w = frames[0].shape[:2]
        tmp_file = out_file.with_name(f".tmp_{out_file.name}")
        writer = cv2.VideoWriter(
            str(tmp_file),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.target_fps,
            (w, h),
        )
        for frm in frames:
            if frm.shape[:2] != (h, w):
                frm = cv2.resize(frm, (w, h))
            writer.write(frm)
        writer.release()

        if tmp_file.exists():
            tmp_file.replace(out_file)

    def _format_path(self, out_file: Path) -> str:
        if self.project_root is not None:
            try:
                return out_file.resolve().relative_to(self.project_root.resolve()).as_posix()
            except ValueError:
                pass
        return out_file.as_posix()
