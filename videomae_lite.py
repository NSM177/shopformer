"""Tiện ích VideoMAE cho demo phát hiện hành vi nghi trộm cắp.

Dùng chung cho notebook huấn luyện và script demo thời gian thực:
  - đọc clip từ file video (chỉ giữ các khung cần thiết, tiết kiệm RAM)
  - tiền xử lý (resize, chuẩn hóa) và tăng cường dữ liệu cho huấn luyện
  - Predictor: nạp model đã lưu và trả xác suất "shoplifting" cho một cửa sổ khung hình
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def uniform_indices(n_total: int, n_frames: int = 16, offset: float = 0.0,
                    span: float = 1.0) -> np.ndarray:
    """n_frames chỉ số đều nhau trong một đoạn con của [0, n_total).

    span   : tỉ lệ độ dài đoạn con so với cả clip (1.0 = cả clip).
    offset : vị trí bắt đầu của đoạn con (0 = đầu, 1 = sát cuối).
    """
    length = max(1, int(round(n_total * span)))
    start = int(round((n_total - length) * offset))
    return np.linspace(start, start + length - 1, n_frames).round().astype(int)


def clip_info(path) -> tuple[int, float]:
    """(số khung hình, fps) của một file video."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    cap.release()
    return n, fps


def read_video_frames(path, n_frames: int = 16, size: int = 224, train: bool = False,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Đọc n_frames khung hình đều nhau -> uint8 [T, size, size, 3] (RGB).

    Khi train=True lấy ngẫu nhiên một đoạn con 70-100% của clip để đa dạng hóa nhịp độ.
    """
    rng = rng or np.random.default_rng()
    n_total, _ = clip_info(path)
    if n_total <= 0:
        raise ValueError(f"Không đọc được số khung hình: {path}")
    span, offset = (float(rng.uniform(0.7, 1.0)), float(rng.uniform(0, 1))) if train else (1.0, 0.0)
    idx = uniform_indices(n_total, n_frames, offset, span)
    need = set(idx.tolist())
    cap = cv2.VideoCapture(str(path))
    got, last = {}, None
    for i in range(int(idx.max()) + 1):
        if not cap.grab():
            break
        if i in need:
            ok, f = cap.retrieve()
            if ok:
                last = cv2.cvtColor(cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA),
                                    cv2.COLOR_BGR2RGB)
                got[i] = last
    cap.release()
    if last is None:
        raise ValueError(f"Không giải mã được khung nào: {path}")
    return np.stack([got.get(int(i), last) for i in idx])   # thiếu khung -> lặp khung cuối


def augment(frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Tăng cường nhẹ, áp dụng giống nhau cho mọi khung của clip."""
    t, h, w, _ = frames.shape
    if rng.random() < 0.5:                                   # lật gương
        frames = frames[:, :, ::-1]
    s = float(rng.uniform(0.8, 1.0))                         # cắt ngẫu nhiên rồi phóng lại
    ch, cw = int(h * s), int(w * s)
    y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    crop = frames[:, y0:y0 + ch, x0:x0 + cw]
    frames = np.stack([cv2.resize(np.ascontiguousarray(f), (w, h), interpolation=cv2.INTER_LINEAR)
                       for f in crop])
    alpha, beta = float(rng.uniform(0.8, 1.2)), float(rng.uniform(-20, 20))   # tương phản, sáng
    return np.clip(frames.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def to_tensor(frames: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """uint8 [T,H,W,3] -> float [T,3,H,W] đã chuẩn hóa (đúng dạng pixel_values của VideoMAE)."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).float() / 255.0
    x = (x - torch.tensor(mean)) / torch.tensor(std)
    return x.permute(0, 3, 1, 2).contiguous()


def resize_rgb(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB)


class Predictor:
    """Nạp bundle (model/ + preprocess.json + calibration.json) và chấm điểm một cửa sổ."""

    def __init__(self, bundle_dir, device: str | None = None):
        from transformers import VideoMAEForVideoClassification
        d = Path(bundle_dir)
        self.meta = json.loads((d / "preprocess.json").read_text(encoding="utf-8"))
        self.cal = json.loads((d / "calibration.json").read_text(encoding="utf-8"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VideoMAEForVideoClassification.from_pretrained(d / "model").to(self.device).eval()
        self.pos = int(self.model.config.label2id["shoplifting"])
        self.n_frames = int(self.meta["num_frames"])
        self.size = int(self.meta["size"])
        self.mean, self.std = self.meta["mean"], self.meta["std"]
        self.threshold = float(self.cal["threshold"])
        self.clip_sec = float(self.meta["clip_sec"])

    @torch.no_grad()
    def predict(self, frames_rgb: np.ndarray) -> float:
        """frames_rgb: uint8 [n_frames, size, size, 3]. Trả xác suất 'shoplifting'."""
        x = to_tensor(frames_rgb, self.mean, self.std).unsqueeze(0).to(self.device)
        return float(torch.softmax(self.model(pixel_values=x).logits, -1)[0, self.pos])
