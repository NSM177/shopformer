#!/usr/bin/env python3
"""Phát luồng Video từ Camera USB / Webcam trên laptop thành HTTP MJPEG Stream.

Tính năng thông minh (Auto-Detect):
- Mặc định tự động quét tìm camera USB cắm ngoài vào laptop trước (index 1, 2...).
- Nếu không có camera ngoài, tự động chuyển về dùng camera tích hợp của laptop (index 0).
- Phát luồng HTTP MJPEG nội bộ mượt mà (30 FPS, DirectShow trên Windows) để Docker AI đọc được:
    http://host.docker.internal:8554/stream.mjpg

Cách dùng:
    python scripts/stream_usb_cam.py             (Tự động chọn: ưu tiên cam USB ngoài, nếu không có thì lấy cam laptop)
    python scripts/stream_usb_cam.py --cam 1     (Chỉ định camera 1)
    python scripts/stream_usb_cam.py --cam 0     (Chỉ định camera 0)
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import threading
import time

import cv2


def auto_detect_camera() -> int:
    """Tự động tìm camera: ưu tiên Camera USB cắm ngoài (index 1, 2), nếu không có thì lấy camera laptop (index 0)."""
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    print("\n🔍 [AUTO-DETECT] Đang quét các camera kết nối với máy tính...")

    # 1. Quét tìm camera USB cắm ngoài trước (ưu tiên index 1, 2, 3)
    for idx in [1, 2, 3]:
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"   👉 ĐÃ PHÁT HIỆN CAMERA USB NGOÀI (Index {idx}) ➔ ƯU TIÊN SỬ DỤNG.")
                return idx

    # 2. Nếu không có camera USB ngoài, fallback về camera laptop (index 0)
    cap0 = cv2.VideoCapture(0, backend)
    if cap0.isOpened():
        ret, _ = cap0.read()
        cap0.release()
        if ret:
            print("   👉 Không có camera USB cắm ngoài ➔ Sử dụng CAMERA TÍCH HỢP LAPTOP (Index 0).")
            return 0

    print("   ⚠️ Không mở được camera nào qua kiểm tra nhanh. Thử dùng index 0...")
    return 0


class CameraStreamer:
    """Quản lý đọc camera USB và nén thành frame JPEG chia sẻ cho các client HTTP với độ trễ tối thiểu (Real-time)."""

    def __init__(self, cam_index: int = 0, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.target_fps = fps
        self.running = False
        self.latest_jpeg: bytes | None = None
        self.cond = threading.Condition()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.cam_index, backend)

        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở Camera ở index {self.cam_index}.")

        # Tối ưu độ trễ thấp nhất (Low Latency):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm bộ đệm buffer xuống 1 để chống trễ hình
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self.running = True

        def _capture_loop():
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cam_label = "Camera USB ngoài" if self.cam_index >= 1 else "Camera tích hợp laptop"
            print(f"[STREAMER] Đang thu hình thời gian thực từ {cam_label} (#{self.cam_index}) - {actual_w}x{actual_h}px @ 30 FPS...")

            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 75]

            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # Encode JPEG nhanh
                ret, buf = cv2.imencode(".jpg", frame, encode_params)
                if ret:
                    jpeg_bytes = buf.tobytes()
                    with self.cond:
                        self.latest_jpeg = jpeg_bytes
                        self.cond.notify_all()  # Đẩy ngay lập tức sang HTTP, không chờ đợi

                # Nghỉ cực ngắn để không quá tải 1 lõi CPU (giữ mượt mà 30 FPS)
                time.sleep(0.002)

            cap.release()
            print("[STREAMER] Đã dừng camera.")

        self._thread = threading.Thread(target=_capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_frame(self) -> bytes | None:
        with self.cond:
            return self.latest_jpeg


class MJPEGHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler phục vụ luồng multipart/x-mixed-replace độ trễ siêu thấp."""

    streamer: CameraStreamer | None = None

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.path in ("/stream.mjpg", "/video", "/cam", "/"):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            try:
                streamer = self.streamer
                if not streamer:
                    return

                while streamer.running:
                    with streamer.cond:
                        streamer.cond.wait(timeout=0.04)
                        frame = streamer.latest_jpeg

                    if frame:
                        part = (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                            + frame + b"\r\n"
                        )
                        self.wfile.write(part)
            except (ConnectionResetError, BrokenPipeError):
                pass
        else:
            self.send_error(404, "Not Found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phát luồng Camera USB / Webcam ra mạng nội bộ HTTP MJPEG.")
    parser.add_argument(
        "--cam",
        type=str,
        default="auto",
        help="Chỉ số Camera (mặc định: 'auto' tự dò tìm camera USB ngoài trước, không có thì lấy camera laptop).",
    )
    parser.add_argument("--port", type=int, default=8554, help="Cổng HTTP stream (mặc định: 8554).")
    parser.add_argument("--width", type=int, default=640, help="Độ rộng khung hình (mặc định: 640 để xử lý thời gian thực).")
    parser.add_argument("--height", type=int, default=480, help="Độ cao khung hình (mặc định: 480 để xử lý thời gian thực).")
    args = parser.parse_args()

    # Tự động chọn camera nếu để 'auto'
    if args.cam.lower() == "auto":
        cam_idx = auto_detect_camera()
    else:
        cam_idx = int(args.cam)

    streamer = CameraStreamer(cam_index=cam_idx, width=args.width, height=args.height)
    try:
        streamer.start()
    except Exception as e:
        print(f"[LỖI] Không thể mở camera #{cam_idx}: {e}")
        return

    MJPEGHandler.streamer = streamer

    server = ThreadingHTTPServer(("0.0.0.0", args.port), MJPEGHandler)
    server.daemon_threads = True

    cam_type_str = "Camera USB ngoài" if cam_idx >= 1 else "Camera tích hợp laptop"
    print("\n" + "=" * 65)
    print(f"🚀 ĐANG PHÁT LUỒNG {cam_type_str.upper()} (Index #{cam_idx}):")
    print(f"   - Trên laptop:        http://localhost:{args.port}/stream.mjpg")
    print(f"   - Trong Docker:       http://host.docker.internal:{args.port}/stream.mjpg")
    print("=" * 65)
    print("\n👉 CÁCH DÙNG TRÊN DASHBOARD SHOPFORMER:")
    print("   1. Mở http://localhost:8000")
    print("   2. Bấm chọn nút 'Webcam' rồi bấm 'Bắt đầu'")
    print("      (Hệ thống sẽ tự động nhận diện luồng camera này và hiển thị lên màn hình!)")
    print("\nBấm Ctrl+C để dừng phát luồng.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[HỆ THỐNG] Đang dừng luồng phát...")
    finally:
        streamer.stop()
        server.server_close()
        print("[HỆ THỐNG] Đã giải phóng camera. Tạm biệt!")


if __name__ == "__main__":
    main()
