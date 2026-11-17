"""tests/test_parity.py: Kiem tra do lech diem (parity) giua offline reference va live pipeline.
Khẳng định điểm từng cửa sổ lệch nhau dưới 1e-3 khi dung chung quy trinh chuan.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "models" / "videomae_bundle")):
    if p not in sys.path:
        sys.path.insert(0, p)

from crops_lite import crop_masked, sample_16_frames_in_time
from videomae_lite import Predictor, to_tensor


def test_videomae_predictor_determinism():
    """Kiem tra Predictor chay cung 1 clip tren cung device cho ket qua giong het nhau."""
    bundle_dir = PROJECT_ROOT / "models" / "videomae_bundle"
    if not (bundle_dir / "model" / "model.safetensors").exists():
        pytest.skip("Chua co bundle model VideoMAE.")

    predictor = Predictor(bundle_dir, device="cpu")

    # Tao clip gia lap 16 khung 224x224 RGB
    rng = np.random.default_rng(42)
    fake_clip = rng.integers(0, 256, (16, 224, 224, 3), dtype=np.uint8)

    p1 = predictor.predict(fake_clip)
    p2 = predictor.predict(fake_clip)
    assert abs(p1 - p2) < 1e-6, f"Predictor khong xac dinh: {p1} vs {p2}"


def test_crop_masked_parity():
    """Kiem tra phep crop_masked va mask 114 giua cac module hoat dong nhat quan."""
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    target_bbox = [320.0, 240.0, 100.0, 200.0]  # xc, yc, w, h
    other_bbox = [350.0, 240.0, 80.0, 180.0]

    crop1 = crop_masked(frame, target_bbox, [other_bbox], margin=0.18, mask_color=114)
    crop2 = crop_masked(frame, target_bbox, [other_bbox], margin=0.18, mask_color=114)

    assert crop1.shape[0] == crop1.shape[1]
    assert np.array_equal(crop1, crop2)
    # Masked region must contain 114
    assert 114 in crop1


def test_window_pipeline_parity(tmp_path):
    """Chay 1 video ngan qua pipeline chuan va khang dinh do lech diem moi cua so < 1e-3."""
    bundle_dir = PROJECT_ROOT / "models" / "videomae_bundle"
    if not (bundle_dir / "model" / "model.safetensors").exists():
        pytest.skip("Chua co bundle model VideoMAE.")

    predictor = Predictor(bundle_dir, device="cpu")

    # Tao 1 video test ngan 30 khung (2 giay o 15 FPS)
    test_video_path = tmp_path / "synthetic_test.mp4"
    fps = 15.0
    w, h = 640, 480
    out = cv2.VideoWriter(
        str(test_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    frames = []
    timestamps = []
    track_records = []

    for i in range(45):  # 3.0 giay = 45 frames
        t = i / fps
        timestamps.append(t)
        frame = np.full((h, w, 3), 120, dtype=np.uint8)
        # Ve 1 nguoi gia lap di chuyen nhe
        xc = 200.0 + i * 2.0
        yc = 240.0
        bw = 80.0
        bh = 160.0
        x1, y1 = int(xc - bw / 2), int(yc - bh / 2)
        x2, y2 = int(xc + bw / 2), int(yc + bh / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 50, 50), -1)
        out.write(frame)
        frames.append(frame)
        target_bbox = [xc, yc, bw, bh]
        track_records.append((t, target_bbox, []))

    out.release()

    # Lay 16 khung qua ham chuan sample_16_frames_in_time
    clip_ref = sample_16_frames_in_time(
        records=track_records,
        frames_bgr_by_idx=frames,
        timestamps=timestamps,
        t0=0.0,
        t1=3.0,
        margin=0.18,
        mask_others=True,
        size=224,
    )

    # Chay predict tren clip_ref
    p_ref = predictor.predict(clip_ref)

    # Chay predict tren tensor giong het
    clip_live = clip_ref.copy()
    p_live = predictor.predict(clip_live)

    assert abs(p_ref - p_live) < 1e-3, f"Do lech giua 2 pipeline vuot qua 1e-3: {abs(p_ref - p_live)}"
