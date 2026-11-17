"""crops_lite.py: Tiện ích cắt vùng người, mask_others và tracking dùng chung cho train và suy luận chuẩn."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import cv2
import numpy as np


def crop_square(
    frame_bgr: np.ndarray,
    bbox_xywh: np.ndarray | list[float],
    margin: float = 0.18,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Cắt vuông quanh bbox với lề margin, trả về (crop_bgr, (x1, y1, x2, y2))."""
    h, w = frame_bgr.shape[:2]
    cx, cy, bw, bh = bbox_xywh
    half_w = bw * (0.5 + float(margin))
    half_h = bh * (0.5 + float(margin))
    half_side = max(half_w, half_h)
    side = min(float(max(w, h)), 2.0 * half_side)
    half_s = side / 2.0

    cx_clamped = max(half_s, min(float(w) - half_s, cx)) if side <= w else w / 2.0
    cy_clamped = max(half_s, min(float(h) - half_s, cy)) if side <= h else h / 2.0

    x1 = max(0, min(w - 1, int(round(cx_clamped - half_s))))
    y1 = max(0, min(h - 1, int(round(cy_clamped - half_s))))
    x2 = max(x1 + 1, min(w, int(round(cx_clamped + half_s))))
    y2 = max(y1 + 1, min(h, int(round(cy_clamped + half_s))))

    crop = np.ascontiguousarray(frame_bgr[y1:y2, x1:x2])
    ch, cw = crop.shape[:2]
    if ch != cw:
        max_s = max(ch, cw)
        pad_t = (max_s - ch) // 2
        pad_b = max_s - ch - pad_t
        pad_l = (max_s - cw) // 2
        pad_r = max_s - cw - pad_l
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)

    return crop, (x1, y1, x2, y2)


def crop_masked(
    frame_bgr: np.ndarray,
    target_bbox_xywh: np.ndarray | list[float],
    other_bboxes_xywh: list[np.ndarray | list[float]] | None = None,
    margin: float = 0.18,
    mask_others: bool = True,
    mask_color: int = 114,
) -> np.ndarray:
    """Cắt vuông quanh target_bbox; nếu mask_others=True thì tô xám (114) các bbox người khác."""
    crop, (x1, y1, x2, y2) = crop_square(frame_bgr, target_bbox_xywh, margin=margin)
    if mask_others and other_bboxes_xywh:
        # Làm việc trên bản sao của crop để tô xám người khác
        crop = crop.copy()
        for ob in other_bboxes_xywh:
            ocx, ocy, obw, obh = ob
            ox1 = int(round(ocx - obw / 2.0))
            oy1 = int(round(ocy - obh / 2.0))
            ox2 = int(round(ocx + obw / 2.0))
            oy2 = int(round(ocy + obh / 2.0))

            # Tìm phần giao nhau giữa bbox người khác và vùng crop (x1, y1, x2, y2)
            ix1 = max(x1, ox1)
            iy1 = max(y1, oy1)
            ix2 = min(x2, ox2)
            iy2 = min(y2, oy2)

            if ix2 > ix1 and iy2 > iy1:
                # Tọa độ tương đối bên trong crop
                cx1 = ix1 - x1
                cy1 = iy1 - y1
                cx2 = ix2 - x1
                cy2 = iy2 - y1
                crop[cy1:cy2, cx1:cx2] = mask_color

    return crop


class YoloTracker:
    """YOLOv8s ByteTrack chuẩn dùng chung cho offline và evaluation."""

    def __init__(
        self,
        weights: str = "models/yolov8s.pt",
        conf: float = 0.15,
        classes: list[int] | None = None,
        tracker: str = "bytetrack.yaml",
        device: str | None = None,
    ):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = float(conf)
        self.classes = classes if classes is not None else [0]
        t_cfg = tracker if (str(tracker).endswith(".yaml") or str(tracker).endswith(".yml")) else f"{tracker}.yaml"
        self.tracker = t_cfg
        self.device = device

    def reset(self) -> None:
        """Reset trạng thái tracker."""
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                self.model.predictor.trackers = []

    def track_frame(self, frame_bgr: np.ndarray) -> list[tuple[int, list[float], float]]:
        """Trả về danh sách (track_id, [cx, cy, w, h], conf) cho từng người."""
        res = self.model.track(
            frame_bgr,
            persist=True,
            conf=self.conf,
            classes=self.classes,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
        )
        tracks = []
        if not res or res[0].boxes is None:
            return tracks

        boxes = res[0].boxes
        if boxes.id is None:
            return tracks

        for i in range(len(boxes)):
            tid = int(boxes.id[i].item())
            xywh = [float(v) for v in boxes.xywh[i].cpu().numpy().tolist()]
            c = float(boxes.conf[i].item())
            tracks.append((tid, xywh, c))
        return tracks


def make_windows(
    track_records: dict[int, list[tuple[float, list[float], list[list[float]]]]],
    duration_sec: float,
    clip_sec: float = 3.0,
    hop_sec: float = 1.0,
    min_cover: float = 0.9,
) -> list[dict[str, Any]]:
    """Tạo danh sách cửa sổ suy luận [t0, t0 + clip_sec] theo thời gian thực (timestamp).
    
    track_records: {track_id: [(ts, target_bbox_xywh, other_bboxes_xywh), ...]}
    min_cover: độ phủ tối thiểu (mặc định 90% cửa sổ = 2.7s).
    """
    windows = []
    t0 = 0.0
    while t0 + clip_sec <= duration_sec + 1e-4:
        t1 = t0 + clip_sec
        for tid, records in track_records.items():
            in_win = [r for r in records if t0 <= r[0] <= t1 + 1e-4]
            if not in_win:
                continue
            span = in_win[-1][0] - in_win[0][0]
            if span >= clip_sec * min_cover:
                windows.append({
                    "track_id": tid,
                    "t0": round(t0, 2),
                    "t1": round(t1, 2),
                    "records": in_win,
                })
        t0 = round(t0 + hop_sec, 2)
    return windows


def sample_16_frames_in_time(
    records: list[tuple[float, list[float], list[list[float]]]],
    frames_bgr_by_idx: list[np.ndarray],
    timestamps: list[float],
    t0: float,
    t1: float,
    margin: float = 0.18,
    mask_others: bool = True,
    size: int = 224,
) -> np.ndarray:
    """Lấy đúng 16 khung đều nhau theo THỜI GIAN trong [t0, t1], cắt vuông (mask other) -> uint8 [16, size, size, 3] RGB."""
    target_ts = np.linspace(t0, t1, 16)
    out_rgb = []
    # Bản đồ thời gian -> index
    ts_arr = np.array(timestamps)
    rec_ts_arr = np.array([r[0] for r in records])

    for tgt in target_ts:
        # Tìm frame gốc gần nhất theo timestamp
        f_idx = int(np.argmin(np.abs(ts_arr - tgt)))
        frame = frames_bgr_by_idx[f_idx]

        # Tìm bbox của track và other_bboxes tại thời điểm gần nhất
        r_idx = int(np.argmin(np.abs(rec_ts_arr - tgt)))
        _, target_bbox, other_bboxes = records[r_idx]

        crop = crop_masked(
            frame,
            target_bbox_xywh=target_bbox,
            other_bboxes_xywh=other_bboxes,
            margin=margin,
            mask_others=mask_others,
        )
        crop_resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        out_rgb.append(crop_rgb)

    return np.stack(out_rgb, axis=0).astype(np.uint8)


def render_windows(
    frames_rgb_16: np.ndarray,
    out_path: Path | str,
    title: str = "",
) -> None:
    """Lưu lưới ảnh 4x4 từ 16 khung hình RGB [16, 224, 224, 3] ra file PNG."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in range(4):
        row = np.hstack([frames_rgb_16[r * 4 + c] for c in range(4)])
        rows.append(row)
    grid_rgb = np.vstack(rows)
    grid_bgr = cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR)

    if title:
        vis = np.zeros((grid_bgr.shape[0] + 30, grid_bgr.shape[1], 3), dtype=np.uint8)
        vis[:30] = (15, 23, 42)
        cv2.putText(vis, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        vis[30:] = grid_bgr
        grid_bgr = vis

    cv2.imwrite(str(p), grid_bgr)


def alarm_events(
    scores: list[float],
    threshold: float = 0.55,
    smooth_windows: int = 2,
    consecutive_k: int = 2,
) -> tuple[list[float], list[int]]:
    """Áp dụng làm mượt và đếm k cửa sổ liên tiếp >= threshold."""
    smoothed = []
    w = max(1, int(smooth_windows))
    for i in range(len(scores)):
        sub = scores[max(0, i - w + 1): i + 1]
        smoothed.append(float(np.mean(sub)))

    alarm_indices = []
    streak = 0
    for i, s in enumerate(smoothed):
        if s >= threshold:
            streak += 1
            if streak >= consecutive_k:
                alarm_indices.append(i)
        else:
            streak = 0
    return smoothed, alarm_indices
