"""Unit tests for Stage C: source, pose_track, buffer, preprocess, and extract_poses."""

from __future__ import annotations

import json
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

from scripts.extract_poses import extract_video_poses_to_poselift_dict  # noqa: E402
from shoplift_iot.capture.source import VideoSource  # noqa: E402
from shoplift_iot.model.scorer import PoseSegDataset  # noqa: E402
from shoplift_iot.vision.buffer import PoseBuffer  # noqa: E402
from shoplift_iot.vision.pose_track import PersonPose, PoseTracker  # noqa: E402
from shoplift_iot.vision.preprocess import (  # noqa: E402
    PosePreprocessor,
    coco17_to_openpose18,
    interpolate_missing_frames,
    normalize_shopformer_pose,
    smooth_pose_sequence,
)


def _make_dummy_video(path: Path, fps: float = 30.0, num_frames: int = 30) -> None:
    """Create a small synthetic MP4 video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    for i in range(num_frames):
        frame = np.full((240, 320, 3), fill_value=(i * 5) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class _DummyYOLO:
    """Deterministic mock YOLO model for fast unit testing of PoseTracker."""

    def __init__(self) -> None:
        self.call_count = 0

    def track(self, **kwargs):
        self.call_count += 1
        kps = torch.zeros((1, 17, 3), dtype=torch.float32)
        kps[0, :, 0] = torch.linspace(100.0, 200.0, 17) + float(self.call_count)
        kps[0, :, 1] = torch.linspace(50.0, 220.0, 17) + float(self.call_count * 0.5)
        kps[0, :, 2] = 0.92
        boxes = SimpleNamespace(
            id=torch.tensor([7], dtype=torch.int32),
            xywh=torch.tensor([[150.0, 130.0, 60.0, 170.0]], dtype=torch.float32),
            conf=torch.tensor([0.88], dtype=torch.float32),
        )
        keypoints = SimpleNamespace(data=kps)
        return [SimpleNamespace(boxes=boxes, keypoints=keypoints)]


def test_video_source_downsamples_to_15fps(tmp_path: Path) -> None:
    """Verify VideoSource downsamples a 30 FPS video of 30 frames to 15 frames at 15 FPS."""
    vid_path = tmp_path / "test_30fps.mp4"
    _make_dummy_video(vid_path, fps=30.0, num_frames=30)

    with VideoSource(vid_path, target_fps=15.0) as src:
        packets = list(src)

    assert len(packets) == 15
    assert packets[0].frame_idx == 0
    assert packets[-1].frame_idx == 14
    assert packets[0].width == 320
    assert packets[0].height == 240


def test_pose_tracker_extracts_coco17_and_track_id() -> None:
    """Verify PoseTracker parses boxes.id, boxes.xywh, and 17x3 keypoints."""
    tracker = PoseTracker(device="cpu", yolo_instance=_DummyYOLO())
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    poses = tracker.process_frame(dummy_frame)
    assert len(poses) == 1
    assert poses[0].track_id == 7
    assert poses[0].bbox_xywh.shape == (4,)
    assert poses[0].keypoints_xyc.shape == (17, 3)


def test_pose_buffer_sliding_window_and_missing_frames() -> None:
    """Verify PoseBuffer emits 12-frame windows per ID and tolerates up to 2 missing frames."""
    buf = PoseBuffer(window_size=12, max_missing_frames=2, stale_timeout_frames=20)
    emitted_windows = []

    for f_idx in range(15):
        if f_idx in (3, 7):
            # Drop 2 frames inside window
            continue
        p = PersonPose(
            track_id=42,
            bbox_xywh=np.array([100.0, 100.0, 50.0, 120.0], dtype=np.float32),
            keypoints_xyc=np.ones((17, 3), dtype=np.float32) * float(f_idx),
            score=0.95,
        )
        emitted_windows.extend(buf.update(f_idx, [p]))

    assert len(emitted_windows) >= 3
    first_win = emitted_windows[0]
    assert first_win.track_id == 42
    assert first_win.end_frame - first_win.start_frame + 1 == 12
    assert len(first_win.frame_to_kps) == 10


def test_preprocess_interpolation_smoothing_and_normalization() -> None:
    """Verify linear interpolation, 8-frame smoothing, 17->18 conversion, and normalization."""
    frame_to_kps: dict[int, np.ndarray] = {}
    for t in range(12):
        if t == 5:
            continue  # Missing frame 5
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[:, 0] = 400.0 + 10.0 * t + np.arange(17, dtype=np.float32) * 5.0
        kp[:, 1] = 200.0 + 20.0 * t + np.arange(17, dtype=np.float32) * 15.0
        kp[:, 2] = 0.9
        frame_to_kps[t] = kp

    dense = interpolate_missing_frames(frame_to_kps, start_frame=0, end_frame=11, max_missing_frames=2)
    assert dense is not None
    assert dense.shape == (12, 17, 3)
    # Frame 5 should equal the exact midpoint between frame 4 and frame 6
    expected_f5_x0 = 0.5 * (frame_to_kps[4][0, 0] + frame_to_kps[6][0, 0])
    assert np.isclose(dense[5, 0, 0], expected_f5_x0, atol=1e-4)

    smoothed = smooth_pose_sequence(dense, window_size=8)
    assert smoothed.shape == (12, 17, 3)

    kp18 = coco17_to_openpose18(smoothed)
    assert kp18.shape == (12, 18, 3)

    normed = normalize_shopformer_pose(kp18, vid_res=(1920, 1080))
    assert normed.shape == (2, 12, 18)
    assert not np.isnan(normed).any()

    prep = PosePreprocessor(vid_res=(1920, 1080), window_size=12, smoothing_window=8, max_missing_frames=2)
    out = prep.process_window(frame_to_kps, start_frame=0, end_frame=11)
    assert out is not None
    assert out.shape == (2, 12, 18)


def test_extract_poses_matches_poselift_format_and_loads_in_dataset(tmp_path: Path) -> None:
    """Verify extract_poses outputs exact PoseLift JSON schema and loads into PoseSegDataset."""
    vid_path = tmp_path / "sample.mp4"
    _make_dummy_video(vid_path, fps=15.0, num_frames=18)

    tracker = PoseTracker(device="cpu", yolo_instance=_DummyYOLO())
    json_dict, pkl_dict = extract_video_poses_to_poselift_dict(
        video_path=vid_path,
        tracker=tracker,
        target_fps=15.0,
        max_missing_frames=2,
        smoothing_window=8,
        swap_xy_to_yx_for_poselift=True,
    )

    # Compare schema against a real PoseLift JSON file
    real_poselift_files = sorted((PROJECT_ROOT / "data" / "PoseLift" / "pose" / "test").glob("*.json"))
    assert len(real_poselift_files) > 0
    with open(real_poselift_files[0], "r", encoding="utf-8") as rf:
        real_data = json.load(rf)
    real_pid = next(iter(real_data.keys()))
    real_fid = next(iter(real_data[real_pid].keys()))
    real_entry = real_data[real_pid][real_fid]

    assert "7" in json_dict
    assert "0" in json_dict["7"]
    our_entry = json_dict["7"]["0"]
    assert set(our_entry.keys()) == set(real_entry.keys()) == {"keypoints", "scores"}
    assert len(our_entry["keypoints"]) == len(real_entry["keypoints"]) == 51
    assert our_entry["scores"] is None

    # Save JSON and verify PoseSegDataset loads it identically to PoseLift
    out_dir = tmp_path / "poselift_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_file = out_dir / "01_0900_alphapose_tracked_person.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_dict, f)

    ds = PoseSegDataset(
        str(out_dir),
        normalize_pose_segs=True,
        evaluate=True,
        seg_len=12,
        seg_stride=1,
        dataset="Poselift",
        vid_res=[1920, 1080],
        swap_yx=True,
    )
    assert len(ds) == 6
    assert ds[0][0].shape == (3, 12, 18)
    assert len(pkl_dict) == 18

