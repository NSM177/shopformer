"""Step 1: Video source reader with frame rate downsampling to target FPS (15 FPS) and auto-reconnect."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Callable, Iterator, Mapping

import cv2

from shoplift_iot.config import AppConfig, load_app_config
from shoplift_iot.contracts import FramePacket


def _is_live_source(source: str | int) -> bool:
    """Return True if `source` is a webcam index or network stream URL (RTSP/HTTP/RTMP)."""
    if isinstance(source, int):
        return True
    s_lower = str(source).lower()
    return s_lower.startswith(("rtsp://", "rtmp://", "http://", "https://"))


class VideoSource:
    """Read frames from a video file, RTSP stream, or webcam, downsample to target FPS, and auto-reconnect."""

    def __init__(
        self,
        source: str | int | Path,
        target_fps: float = 15.0,
        realtime_pacing: bool = False,
        auto_reconnect: bool | None = None,
        max_reconnect_attempts: int = 3,
        reconnect_delay_seconds: float = 1.0,
        capture_factory: Callable[[str | int], Any] | None = None,
    ) -> None:
        """Initialize video source.

        Args:
            source: Camera index (e.g. 0), RTSP URL, or path to a video file.
            target_fps: Target output frame rate in FPS (default 15.0).
            realtime_pacing: Whether to sleep between frames when reading from a file.
            auto_reconnect: Whether to reconnect automatically when stream drops (defaults to True for live streams).
            max_reconnect_attempts: Maximum consecutive reconnect attempts before giving up.
            reconnect_delay_seconds: Seconds to wait between reconnect attempts.
            capture_factory: Optional custom factory (default `cv2.VideoCapture`) for testing.
        """
        if isinstance(source, Path):
            self.source: str | int = str(source)
        elif isinstance(source, str) and source.isdigit():
            self.source = int(source)
        else:
            self.source = source

        self.target_fps = float(target_fps)
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive.")
        self.realtime_pacing = realtime_pacing
        self.auto_reconnect = (
            _is_live_source(self.source) if auto_reconnect is None else bool(auto_reconnect)
        )
        self.max_reconnect_attempts = max(0, int(max_reconnect_attempts))
        self.reconnect_delay_seconds = max(0.0, float(reconnect_delay_seconds))
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._cap: Any | None = None
        self.src_fps: float = self.target_fps
        self.width: int = 0
        self.height: int = 0
        self.reconnect_count: int = 0

    def open(self) -> None:
        """Open the underlying OpenCV VideoCapture."""
        self._cap = self._capture_factory(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        raw_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.src_fps = raw_fps if (raw_fps > 0 and raw_fps < 240) else self.target_fps
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def close(self) -> None:
        """Release the underlying OpenCV VideoCapture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _try_reconnect(self) -> bool:
        """Attempt to reconnect to `self.source` up to `self.max_reconnect_attempts` times."""
        if not self.auto_reconnect or self.max_reconnect_attempts <= 0:
            return False

        for _ in range(self.max_reconnect_attempts):
            self.close()
            if self.reconnect_delay_seconds > 0:
                time.sleep(self.reconnect_delay_seconds)
            try:
                self.open()
                self.reconnect_count += 1
                return True
            except Exception:
                continue
        return False

    def __enter__(self) -> VideoSource:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __iter__(self) -> Iterator[FramePacket]:
        """Yield frames downsampled to target_fps, reconnecting on stream loss if enabled."""
        if self._cap is None:
            self.open()
        assert self._cap is not None

        ratio = max(1.0, self.src_fps / self.target_fps)
        next_target_raw = 0.0
        raw_idx = 0
        out_idx = 0
        start_wall = time.monotonic()

        while True:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                if self._try_reconnect():
                    continue
                break

            if raw_idx + 1e-6 >= next_target_raw:
                ts = out_idx / self.target_fps
                h, w = frame.shape[:2]
                yield FramePacket(
                    frame_idx=out_idx,
                    raw_frame_idx=raw_idx,
                    timestamp=ts,
                    frame=frame,
                    width=w,
                    height=h,
                    fps=self.target_fps,
                )
                out_idx += 1
                next_target_raw += ratio

                if self.realtime_pacing:
                    expected_elapsed = out_idx / self.target_fps
                    actual_elapsed = time.monotonic() - start_wall
                    sleep_s = expected_elapsed - actual_elapsed
                    if sleep_s > 0:
                        time.sleep(sleep_s)

            raw_idx += 1


def load_source_from_config(
    config_or_path: Path | str | AppConfig | Mapping[str, Any],
    source: str | int | Path,
) -> VideoSource:
    """Create a VideoSource using target_fps and reconnect settings from application configuration."""
    cfg = load_app_config(config_or_path)
    target_fps = float(cfg.video["target_fps"])
    pipe_cfg = cfg.pipeline
    return VideoSource(
        source=source,
        target_fps=target_fps,
        max_reconnect_attempts=int(pipe_cfg.get("max_reconnect_attempts", 3)),
        reconnect_delay_seconds=float(pipe_cfg.get("reconnect_delay_seconds", 1.0)),
    )


def main() -> None:
    """CLI smoke test for VideoSource."""
    parser = argparse.ArgumentParser(description="Test VideoSource reader at target FPS.")
    parser.add_argument("--source", type=str, required=True, help="Video path or camera index.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"), help="Path to YAML config."
    )
    parser.add_argument("--max-frames", type=int, default=30, help="Max frames to read.")
    args = parser.parse_args()

    reader = load_source_from_config(args.config, args.source)
    count = 0
    with reader:
        print(
            f"[VideoSource] Opened '{args.source}' | src_fps={reader.src_fps:.2f} -> "
            f"target_fps={reader.target_fps:.2f} | res={reader.width}x{reader.height}"
        )
        for pkt in reader:
            count += 1
            if count >= args.max_frames:
                break
    print(f"[VideoSource] Successfully read {count} downsampled frames.")


if __name__ == "__main__":
    main()
