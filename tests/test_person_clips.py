"""Unit and integration tests for GD2: `vision/person_clips.py` and `scripts/extract_person_clips.py`."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_person_clips import extract_clips_from_video_file  # noqa: E402
from shoplift_iot.contracts import PersonPose  # noqa: E402
from shoplift_iot.vision.person_clips import (  # noqa: E402
    PersonClipBuffer,
    PersonClipExtractor,
    crop_person_with_margin,
    expand_bbox_xyxy,
    save_person_clip_mp4,
)
from shoplift_iot.vision.pose_track import PoseTracker  # noqa: E402
from videomae_lite import read_video_frames, to_tensor  # noqa: E402


class _MockTwoPersonYOLO:
    """Mock Ultralytics YOLO tracker emitting 2 persistent person tracks (IDs 1 and 2)."""

    def track(self, source: np.ndarray, **kwargs: object) -> list[SimpleNamespace]:
        boxes = SimpleNamespace(
            id=torch.tensor([1, 2], dtype=torch.int32),
            xywh=torch.tensor(
                [[100.0, 120.0, 60.0, 140.0], [220.0, 120.0, 50.0, 130.0]],
                dtype=torch.float32,
            ),
            conf=torch.tensor([0.92, 0.88], dtype=torch.float32),
            cls=torch.tensor([0, 0], dtype=torch.int32),
        )
        keypoints = SimpleNamespace(
            data=torch.ones((2, 17, 3), dtype=torch.float32),
        )
        return [SimpleNamespace(boxes=boxes, keypoints=keypoints)]


def test_expand_bbox_and_crop_with_15pct_margin() -> None:
    """Verify bounding box expansion adds ~15% margin on each side and clamps to frame boundaries."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[40:200, 60:140] = 255  # person region

    bbox = np.array([100.0, 120.0, 80.0, 160.0], dtype=np.float32)
    x1, y1, x2, y2 = expand_bbox_xyxy(bbox, frame_w=320, frame_h=240, margin_ratio=0.15)
    # half_w = 80 * 0.65 = 52 -> [48, 152], half_h = 160 * 0.65 = 104 -> [16, 224]
    assert (x1, y1, x2, y2) == (48, 16, 152, 224)

    crop = crop_person_with_margin(frame, bbox, margin_ratio=0.15)
    assert crop.shape == (208, 104, 3)

    # Boundary clamping test near top-left corner
    edge_bbox = np.array([10.0, 10.0, 40.0, 60.0], dtype=np.float32)
    ex1, ey1, ex2, ey2 = expand_bbox_xyxy(edge_bbox, frame_w=320, frame_h=240, margin_ratio=0.15)
    assert ex1 == 0 and ey1 == 0
    assert ex2 > ex1 and ey2 > ey1


def test_person_clip_buffer_emits_16_frame_videomae_tensor() -> None:
    """Verify per-ID buffer emits uint8 (16, 224, 224, 3) RGB clips matching `videomae_lite.Predictor`."""
    buf = PersonClipBuffer(
        clip_sec=2.0,
        hop_sec=1.0,
        n_frames=16,
        size=224,
        margin_ratio=0.15,
        min_span_ratio=0.8,
        min_frames=8,
        stale_sec=5.0,
    )
    frame = np.full((240, 320, 3), fill_value=128, dtype=np.uint8)
    pose = PersonPose(
        track_id=10,
        bbox_xywh=np.array([160.0, 120.0, 70.0, 150.0], dtype=np.float32),
        keypoints_xyc=np.ones((17, 3), dtype=np.float32),
        score=0.9,
    )

    emitted = []
    # Feed 35 frames at 15 FPS (t = 0.0s .. 2.26s)
    for idx in range(35):
        ts = idx / 15.0
        emitted.extend(buf.update(ts, frame, [pose]))

    assert len(emitted) >= 1
    win = emitted[0]
    assert win.track_id == 10
    assert win.frames_rgb.shape == (16, 224, 224, 3)
    assert win.frames_rgb.dtype == np.uint8

    # Verify compatibility with videomae_lite.to_tensor -> [16, 3, 224, 224] float32
    tensor = to_tensor(win.frames_rgb)
    assert tensor.shape == (16, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_offline_script_and_realtime_extractor_share_code(tmp_path: Path) -> None:
    """Verify `extract_clips_from_video_file` cuts per-person MP4 clips readable by `videomae_lite`."""
    vid_path = tmp_path / "an_001.mp4"
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (320, 240))
    for i in range(36):
        frm = np.full((240, 320, 3), fill_value=(i * 7) % 255, dtype=np.uint8)
        writer.write(frm)
    writer.release()

    out_dir = tmp_path / "clips" / "normal"
    saved_files = extract_clips_from_video_file(
        video_path=vid_path,
        out_dir=out_dir,
        yolo_instance=_MockTwoPersonYOLO(),
        clip_sec_override=1.5,
        hop_sec_override=1.0,
    )
    # 2 tracked people (ID 1 and ID 2) -> both should have saved MP4 clips starting with `an_001_tid`
    assert len(saved_files) >= 2
    track_prefixes = {p.name.split("_")[2] for p in saved_files}
    assert "tid1" in track_prefixes and "tid2" in track_prefixes

    # Verify videomae_lite.read_video_frames reads the saved clip back as (16, 224, 224, 3)
    frames_back = read_video_frames(saved_files[0], n_frames=16, size=224, train=False)
    assert frames_back.shape == (16, 224, 224, 3)
    assert frames_back.dtype == np.uint8
