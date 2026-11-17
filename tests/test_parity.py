"""tests/test_parity.py: Kiểm tra app cắt ảnh người GIỐNG HỆT lúc train VideoMAE.

Tham chiếu là `models/videomae_bundle/crops_lite.py` (notebook train_videomae_2 chép vào bundle):
crop_masked(frame, target_xyxy, others_xyxy, margin, size) -> RGB size x size.
App dùng `shoplift_iot.vision.person_clips` (bbox dạng xywh, giữ BGR rồi resize_rgb).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE = PROJECT_ROOT / "models" / "videomae_bundle"
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shoplift_iot.contracts import PersonPose  # noqa: E402
from shoplift_iot.vision.person_clips import PersonClipBuffer, crop_person_with_margin  # noqa: E402
from videomae_lite import Predictor, resize_rgb  # noqa: E402


def _bundle_crops():
    f = BUNDLE / "crops_lite.py"
    if not f.exists():
        pytest.skip("Bundle chưa có crops_lite.py")
    spec = importlib.util.spec_from_file_location("bundle_crops_lite", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "crop_masked"):
        pytest.skip("crops_lite.py của bundle không có crop_masked")
    return mod


def _xyxy(b):
    cx, cy, w, h = b
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _frame(seed=0, h=480, w=640):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "target, others",
    [
        ([320.0, 240.0, 100.0, 200.0], [[350.0, 240.0, 80.0, 180.0]]),   # 2 người chồng nhau
        ([30.0, 40.0, 60.0, 120.0], []),                                  # sát góc trên-trái
        ([620.0, 400.0, 90.0, 200.0], [[560.0, 380.0, 70.0, 160.0]]),     # sát mép phải-dưới
    ],
)
def test_app_crop_equals_training_crop(target, others):
    ref_mod = _bundle_crops()
    frame = _frame()
    ref = ref_mod.crop_masked(frame, _xyxy(target), [_xyxy(o) for o in others], 0.18, 224)
    app_bgr = crop_person_with_margin(frame, np.array(target), margin_ratio=0.18, square=True,
                                      other_bboxes=[np.array(o) for o in others], mask_others=True, size=224)
    app = resize_rgb(app_bgr, 224)
    assert app.shape == ref.shape == (224, 224, 3)
    assert np.array_equal(app, ref)


def test_clip_buffer_window_equals_training_crops():
    """PersonClipBuffer (đường chạy thật) cho ra đúng 16 crop như cách train tạo cửa sổ."""
    ref_mod = _bundle_crops()
    buf = PersonClipBuffer(clip_sec=3.0, hop_sec=1.0, n_frames=16, size=224, margin_ratio=0.18,
                           min_span_ratio=1.0, min_frames=16, square=True, mask_others=True)
    fps, frames, boxes, wins = 15.0, [], [], []
    for i in range(46):
        fr = _frame(i)
        b = np.array([200.0 + 3 * i, 240.0, 80.0, 160.0], np.float32)
        o = np.array([260.0, 250.0, 70.0, 150.0], np.float32)
        frames.append(fr); boxes.append((b, o))
        poses = [PersonPose(track_id=1, bbox_xywh=b, keypoints_xyc=np.zeros((17, 3), np.float32), score=0.9),
                 PersonPose(track_id=2, bbox_xywh=o, keypoints_xyc=np.zeros((17, 3), np.float32), score=0.9)]
        wins += [w for w in buf.update(i / fps, fr, poses) if w.track_id == 1]
    assert wins, "buffer không phát cửa sổ nào"
    w = wins[0]
    ts = np.arange(46) / fps
    idx = [int(np.argmin(np.abs(ts - t))) for t in np.linspace(w.start_ts, w.end_ts, 16)]
    ref = np.stack([ref_mod.crop_masked(frames[k], _xyxy(boxes[k][0]), [_xyxy(boxes[k][1])], 0.18, 224) for k in idx])
    assert np.array_equal(w.frames_rgb, ref)

    if (BUNDLE / "model" / "model.safetensors").exists():
        pred = Predictor(BUNDLE, device="cpu")
        assert abs(pred.predict(w.frames_rgb) - pred.predict(ref)) < 1e-6


def test_videomae_predictor_determinism():
    if not (BUNDLE / "model" / "model.safetensors").exists():
        pytest.skip("Chưa có bundle model VideoMAE.")
    predictor = Predictor(BUNDLE, device="cpu")
    clip = np.random.default_rng(42).integers(0, 256, (16, 224, 224, 3), dtype=np.uint8)
    assert abs(predictor.predict(clip) - predictor.predict(clip)) < 1e-6
