"""Cắt vùng vuông quanh từng người và tạo cửa sổ clip cho VideoMAE.

Dùng chung cho notebook huấn luyện và pipeline chạy thật, để lúc train và lúc chạy
tiền xử lý GIỐNG HỆT nhau: bbox người -> thêm lề -> vuông hóa -> resize -> 16 khung đều nhau.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np


# ---------------------------------------------------------------- cắt vuông quanh người
def square_box(xyxy, margin: float = 0.18):
    """(tâm x, tâm y, cạnh) của hình vuông bao quanh bbox, đã cộng lề mỗi phía."""
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    side = max(x1 - x0, y1 - y0) * (1.0 + 2.0 * margin)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0, max(side, 8.0)


def crop_square(frame_bgr: np.ndarray, xyxy, margin: float = 0.18, size: int = 224) -> np.ndarray:
    """Cắt vuông quanh bbox, phóng về size x size, trả RGB. Phần vượt mép khung lặp lại pixel biên."""
    cx, cy, side = square_box(xyxy, margin)
    s = size / side
    m = np.array([[s, 0.0, size / 2.0 - s * cx], [0.0, s, size / 2.0 - s * cy]], np.float32)
    out = cv2.warpAffine(frame_bgr, m, (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def crop_masked(frame_bgr: np.ndarray, target, others, margin: float = 0.18, size: int = 224,
                gray: int = 114) -> np.ndarray:
    """Như crop_square nhưng tô xám vùng của NHỮNG NGƯỜI KHÁC, giữ nguyên bên trong bbox của người đang xét.

    Mục đích: crop quanh một người không còn chứa hành động của người đứng cạnh, nên điểm của
    người này không bị kéo theo người kia. Dùng giống hệt nhau lúc train và lúc chạy thật.
    """
    if others:
        h, w = frame_bgr.shape[:2]
        f = frame_bgr.copy()
        for ob in others:
            x0, y0, x1, y1 = (int(round(v)) for v in ob)
            f[max(0, y0):max(0, min(h, y1)), max(0, x0):max(0, min(w, x1))] = gray
        tx0, ty0, tx1, ty1 = (int(round(v)) for v in target)
        tx0, ty0, tx1, ty1 = max(0, tx0), max(0, ty0), min(w, tx1), min(h, ty1)
        f[ty0:ty1, tx0:tx1] = frame_bgr[ty0:ty1, tx0:tx1]
        frame_bgr = f
    return crop_square(frame_bgr, target, margin, size)


# ---------------------------------------------------------------- bộ phát hiện + theo dõi
class YoloTracker:
    """YOLO (chỉ lớp person) + ByteTrack có giữ trạng thái giữa các khung hình.

    Mỗi video cần reset() để ID bắt đầu lại và bộ lọc Kalman không mang từ video trước sang.
    """

    def __init__(self, weights: str = "yolov8s.pt", conf: float = 0.15,
                 tracker: str = "bytetrack.yaml", imgsz: int = 640, device=None):
        self.weights, self.conf, self.tracker = weights, conf, tracker
        self.imgsz, self.device = imgsz, device
        self.model = None
        self.reset()

    def reset(self):
        """Bắt đầu video mới: ID và Kalman về trạng thái sạch, nhưng chỉ nạp trọng số YOLO một lần."""
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.weights)
        try:
            from ultralytics.trackers.basetrack import BaseTrack
            BaseTrack.reset_id()
        except Exception:
            pass
        self.model.predictor = None            # buộc tạo lại tracker ở lần track() kế tiếp

    def update(self, frame_bgr: np.ndarray):
        """-> list[(track_id, (x0, y0, x1, y1), conf)] của khung hiện tại."""
        r = self.model.track(frame_bgr, persist=True, classes=[0], conf=self.conf,
                             tracker=self.tracker, imgsz=self.imgsz, device=self.device,
                             verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            return []
        ids = r.boxes.id.int().cpu().tolist()
        xyxy = r.boxes.xyxy.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        return [(int(i), tuple(float(v) for v in b), float(c)) for i, b, c in zip(ids, xyxy, cf)]


class MotionBlobTracker:
    """Bộ theo dõi giả dựa trên trừ nền, CHỈ để kiểm thử mã bằng video tổng hợp. Không dùng thật."""

    def __init__(self, min_area: int = 80):
        self.min_area = min_area
        self.reset()

    def reset(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=25, detectShadows=False)

    def update(self, frame_bgr: np.ndarray):
        mask = self.bg.apply(frame_bgr)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= self.min_area]
        if not cnts:
            return []
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        return [(1, (float(x), float(y), float(x + w), float(y + h)), 1.0)]


# ---------------------------------------------------------------- quỹ đạo và cửa sổ
def extract_tracks(path, tracker, max_gap_sec: float = 1.5, min_track_sec: float = 1.5,
                   start_f: int = 0, end_f: int | None = None) -> dict:
    """Chạy tracker qua video (hoặc chỉ đoạn khung [start_f, end_f)). Nội suy tuyến tính các khoảng mất phát hiện ngắn.

    Số khung trong tracks là số khung TUYỆT ĐỐI của video. Trả {"fps", "n_frames" (số khung đã phân tích),
    "start_f", "size": (W, H), "tracks": {track_id: {khung: (x0, y0, x1, y1)}}}.
    """
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker.reset()
    raw: dict[int, dict[int, tuple]] = defaultdict(dict)
    if start_f > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_f))
    i = int(start_f)
    while True:
        if end_f is not None and i >= end_f:
            break
        ok, frame = cap.read()
        if not ok:
            break
        for tid, box, _ in tracker.update(frame):
            raw[tid][i] = box
        i += 1
    cap.release()
    det_frames = len({f for d in raw.values() for f in d})
    max_gap, tracks = int(round(max_gap_sec * fps)), {}
    for tid, d in raw.items():
        fr = sorted(d)
        if len(fr) < min_track_sec * fps:
            continue
        full = {}
        for a, b in zip(fr[:-1], fr[1:]):
            full[a] = d[a]
            gap = b - a - 1
            if 0 < gap <= max_gap:
                pa, pb = np.array(d[a]), np.array(d[b])
                for k in range(1, gap + 1):
                    full[a + k] = tuple(float(v) for v in (1 - k / (gap + 1)) * pa + (k / (gap + 1)) * pb)
        full[fr[-1]] = d[fr[-1]]
        tracks[tid] = full
    return dict(fps=fps, n_frames=i - int(start_f), start_f=int(start_f), size=(w, h), tracks=tracks, det_frames=det_frames, raw_tracks=len(raw))


def make_windows(info: dict, clip_sec: float, hop_sec: float, n_frames: int = 16,
                 min_cover: float = 0.9) -> list[dict]:
    """Cửa sổ trượt theo từng track. Chỉ giữ cửa sổ mà track có mặt >= min_cover số khung."""
    fps = info["fps"]
    length, hop = max(n_frames, int(round(clip_sec * fps))), max(1, int(round(hop_sec * fps)))
    out = []
    for tid, tr in info["tracks"].items():
        fr = np.array(sorted(tr))
        first, last = int(fr[0]), int(fr[-1])
        starts = list(range(first, last - length + 2, hop))
        if starts and starts[-1] != last - length + 1:
            starts.append(last - length + 1)                       # cửa sổ sát cuối track
        for s in starts:
            if s < 0:
                continue
            inside = np.count_nonzero((fr >= s) & (fr < s + length))
            if inside / length < min_cover:
                continue
            idx = np.linspace(s, s + length - 1, n_frames).round().astype(int)
            boxes, others = [], []
            for j in idx:
                k = fr[np.argmin(np.abs(fr - j))]                    # khung gần nhất còn có bbox
                boxes.append(tr[int(k)])
                others.append([t2[int(j)] for o, t2 in info["tracks"].items() if o != tid and int(j) in t2])
            out.append(dict(track=int(tid), start_f=int(s), end_f=int(s + length),
                            t0=s / fps, t1=(s + length) / fps, idx=idx, boxes=boxes, others=others))
    return out


def render_windows(path, windows: list[dict], size: int = 224, margin: float = 0.18,
                   mask_others: bool = False) -> list[np.ndarray]:
    """Giải mã video đúng một lần, trả mỗi cửa sổ là uint8 [n_frames, size, size, 3] (RGB).

    mask_others=True: tô xám người khác trong crop (xem crop_masked).
    """
    if not windows:
        return []
    n = len(windows[0]["idx"])
    arrays = [np.zeros((n, size, size, 3), np.uint8) for _ in windows]
    need: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for wi, w in enumerate(windows):
        for k, f in enumerate(w["idx"]):
            need[int(f)].append((wi, k))
    cap = cv2.VideoCapture(str(path))
    first, last = min(need), max(need)
    if first > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(first))      # nhảy tới khung đầu tiên cần, khỏi giải mã từ đầu
    for i in range(first, last + 1):
        if not cap.grab():
            break
        if i in need:
            ok, frame = cap.retrieve()
            if not ok:
                continue
            for wi, k in need[i]:
                w = windows[wi]
                arrays[wi][k] = (crop_masked(frame, w["boxes"][k], w["others"][k], margin, size)
                                 if mask_others else crop_square(frame, w["boxes"][k], margin, size))
    cap.release()
    return arrays


# ---------------------------------------------------------------- quy tắc cảnh báo
def alarm_events(times, probs, thr: float, smooth: int = 1, k: int = 1) -> list[float]:
    """Thời điểm (giây) các sự kiện cảnh báo trên MỘT track.

    Điểm làm mượt = trung bình `smooth` cửa sổ gần nhất; báo khi vượt ngưỡng `k` cửa sổ liên tiếp.
    Mỗi chuỗi vượt ngưỡng liên tục chỉ tính một sự kiện.
    """
    probs = np.asarray(probs, float)
    sm = np.array([probs[max(0, i - smooth + 1): i + 1].mean() for i in range(len(probs))])
    events, streak = [], 0
    for t, s in zip(times, sm):
        streak = streak + 1 if s >= thr else 0
        if streak == k:
            events.append(float(t))
    return events
