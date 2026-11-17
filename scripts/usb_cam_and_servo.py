#!/usr/bin/env python3
"""Script điều khiển Động cơ Servo và Kết nối Camera USB với Laptop.

Tính năng:
1. Tự động dò tìm camera USB và cổng COM kết nối với mạch điều khiển Servo (ESP32 / Arduino).
2. Đọc luồng video từ Camera USB thời gian thực (hỗ trợ DirectShow trên Windows để chống delay).
3. Gửi lệnh điều khiển góc quay Servo (0° mở cửa, 90° khóa cửa kèm thời gian TTL tự mở).
4. Điều khiển trực tiếp bằng phím tắt trên cửa sổ video:
   - [L]: Khóa cửa (Servo 90°, TTL 45s)
   - [U]: Mở cửa ngay lập tức (Servo 0°)
   - [A]: Chỉnh góc Servo tùy ý (0° - 180°)
   - [Q] hoặc [ESC]: Thoát an toàn
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import cv2

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

import subprocess


def list_com_ports() -> list[str]:
    """Liệt kê các cổng COM hiện có trên máy tính."""
    if HAS_SERIAL:
        return [p.device for p in serial.tools.list_ports.comports()]
    if sys.platform.startswith("win"):
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "[System.IO.Ports.SerialPort]::GetPortNames()"],
                text=True,
                timeout=2.0,
            )
            return [line.strip() for line in out.strip().splitlines() if line.strip()]
        except Exception:
            return []
    return []


class ServoController:
    """Quản lý kết nối và gửi lệnh điều khiển Servo qua cổng USB Serial hoặc Giả lập."""

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200) -> None:
        self.port = port
        self.baudrate = baudrate
        self.serial_conn: Optional[serial.Serial] = None
        self.state = "unlocked"  # 'unlocked' | 'locked'
        self.angle = 0           # 0 độ: Mở, 90 độ: Khóa
        self.lock_deadline = 0.0 # Mốc thời gian hết hạn TTL

        self._connect()

    def _connect(self) -> None:
        """Tự động kết nối tới cổng COM chỉ định hoặc cổng COM đầu tiên khả dụng."""
        target_port = self.port
        if not target_port:
            available = list_com_ports()
            if available:
                target_port = available[0]

        self.port = target_port

        if HAS_SERIAL and target_port:
            try:
                self.serial_conn = serial.Serial(target_port, self.baudrate, timeout=0.1)
                time.sleep(1.5)  # Chờ Arduino/ESP32 khởi động sau khi reset DTR
                print(f"[SERVO] Đã kết nối với cổng Serial qua pySerial: {target_port} ({self.baudrate} bps)")
                self.unlock("init_startup")
                return
            except Exception as e:
                print(f"[SERVO] Cảnh báo pySerial ({target_port}): {e}. Sẽ dùng driver Windows native.")
                self.serial_conn = None

        if target_port:
            print(f"[SERVO] Đã phát hiện cổng COM: {target_port}. Sẽ gửi lệnh qua Windows Serial driver.")
            self.unlock("init_startup")
        else:
            print("[SERVO] Chưa phát hiện cổng COM nào cắm vào laptop.")
            print("        Đang chạy ở chế độ MÔ PHỎNG (Simulation Mode).")

    def send_command(self, cmd: str) -> None:
        """Gửi lệnh chuỗi xuống mạch điều khiển qua Serial (pySerial hoặc Windows native)."""
        clean_cmd = cmd.strip()
        if not clean_cmd:
            return

        # Cách 1: Qua pySerial nếu có
        if self.serial_conn and self.serial_conn.is_open:
            try:
                line = (clean_cmd + "\n").encode("utf-8")
                self.serial_conn.write(line)
                self.serial_conn.flush()
                return
            except Exception as e:
                print(f"[SERVO] Lỗi khi gửi dữ liệu qua pySerial: {e}")

        # Cách 2: Qua Windows PowerShell SerialPort nếu pySerial không khả dụng
        if sys.platform.startswith("win") and self.port:
            ps_script = (
                f"$p = New-Object System.IO.Ports.SerialPort('{self.port}', {self.baudrate}, "
                f"[System.IO.Ports.Parity]::None, 8, [System.IO.Ports.StopBits]::One); "
                f"$p.Open(); $p.WriteLine('{clean_cmd}'); Start-Sleep -Milliseconds 20; $p.Close();"
            )
            try:
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_script],
                    timeout=1.5,
                    capture_output=True,
                )
            except Exception as exc:
                print(f"[SERVO] Gửi lệnh qua native COM thất bại: {exc}")

    def unlock(self, reason: str = "user_command") -> None:
        """Mở khóa cửa ngay lập tức: Quay Servo về góc 0 độ."""
        self.state = "unlocked"
        self.angle = 0
        self.lock_deadline = 0.0
        self.send_command("UNLOCK")
        print(f"[SERVO] 🔓 MỞ CỬA (Servo 0°) - Lý do: {reason}")

    def lock(self, ttl_sec: int = 45, reason: str = "user_command") -> None:
        """Khóa cửa: Quay Servo sang góc 90 độ, tự động mở lại sau ttl_sec giây."""
        self.state = "locked"
        self.angle = 90
        self.lock_deadline = time.time() + ttl_sec
        self.send_command(f"LOCK:{ttl_sec}")
        print(f"[SERVO] 🔒 KHÓA CỬA (Servo 90°) - TTL: {ttl_sec}s - Lý do: {reason}")

    def set_angle(self, angle: int) -> None:
        """Đặt góc quay thủ công cho Servo (0° - 180°)."""
        angle = max(0, min(180, angle))
        self.angle = angle
        self.state = "locked" if angle >= 45 else "unlocked"
        self.send_command(f"ANGLE:{angle}")
        print(f"[SERVO] ⚙️ Đặt góc Servo: {angle}°")

    def update(self) -> None:
        """Kiểm tra hết hạn TTL trong vòng lặp chính để tự động mở khóa."""
        if self.state == "locked" and self.lock_deadline > 0:
            if time.time() >= self.lock_deadline:
                self.unlock("ttl_expired")

    def close(self) -> None:
        """Đóng kết nối an toàn: Luôn mở khóa trước khi thoát."""
        self.unlock("shutdown")
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("[SERVO] Đã ngắt kết nối cổng Serial.")


def scan_usb_cameras(max_tested: int = 4) -> list[int]:
    """Quét các chỉ số camera USB khả dụng trên máy tính."""
    available_indices = []
    print("\n🔍 Đang quét các camera USB kết nối với laptop...")
    for idx in range(max_tested):
        # Dùng CAP_DSHOW trên Windows để quét nhanh không bị treo
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"  [+] Tìm thấy Camera index {idx}: Độ phân giải {w}x{h} @ {fps:.1f} FPS")
                available_indices.append(idx)
            cap.release()
    return available_indices


def open_usb_camera(cam_index: int = 0, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """Mở camera USB với cấu hình phân giải và DirectShow chống giật lag."""
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_index, backend)
    
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở camera USB ở index {cam_index}. Hãy kiểm tra kết nối cáp USB.")

    # Thiết lập độ phân giải mong muốn
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30.0)

    # Đọc thử 1 khung hình để kiểm tra
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError(f"Camera index {cam_index} mở được nhưng không truyền được khung hình.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n[CAMERA] Đã kết nối Camera USB (Index: {cam_index}) - {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
    return cap


def draw_hud(frame: cv2.Mat, servo: ServoController, fps: float, cam_index: int) -> None:
    """Vẽ giao diện hiển thị trạng thái Cửa & Servo trực tiếp lên luồng video."""
    h, w = frame.shape[:2]

    # Thanh header phía trên
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 54), (10, 15, 26), -1)
    # Thanh footer phía dưới
    cv2.rectangle(overlay, (0, h - 38), (w, h), (10, 15, 26), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Trạng thái Cửa & Servo
    if servo.state == "locked":
        ttl_left = max(0, int(servo.lock_deadline - time.time()))
        status_text = f"CUA DANG KHOA (Servo: {servo.angle} do) | TTL con: {ttl_left}s"
        status_color = (60, 60, 240)  # Đỏ BGR
    else:
        status_text = f"CUA DANG MO (Servo: {servo.angle} do)"
        status_color = (60, 220, 60)  # Xanh lá BGR

    # Vẽ thông tin lên header
    cv2.putText(frame, f"CAM USB #{cam_index}  |  FPS: {fps:.1f}", (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

    cv2.putText(frame, status_text, (w - 480, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, status_color, 2)

    # Vẽ hướng dẫn phím bấm ở footer
    help_text = "[L]: Khoa cua (90 do)  |  [U]: Mo cua (0 do)  |  [1]: 45 do  |  [2]: 180 do  |  [Q]: Thoat"
    cv2.putText(frame, help_text, (14, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kết nối Camera USB và điều khiển Servo cho hệ thống an ninh.")
    parser.add_argument("--cam", type=int, default=0, help="Chỉ số camera USB (mặc định: 0, webcam/USB cam).")
    parser.add_argument("--port", type=str, default=None, help="Cổng COM của vi điều khiển (VD: COM3, /dev/ttyUSB0).")
    parser.add_argument("--baud", type=int, default=115200, help="Baudrate cổng Serial (mặc định: 115200).")
    parser.add_argument("--scan", action="store_true", help="Chỉ quét danh sách camera USB và cổng COM rồi thoát.")
    args = parser.parse_args()

    # Quét thiết bị nếu có cờ --scan hoặc khi khởi động
    if args.scan:
        cameras = scan_usb_cameras()
        print(f"\n=> Tổng cộng có {len(cameras)} camera USB khả dụng: {cameras}")
        ports = list_com_ports()
        print(f"\n🔍 Quét cổng COM Serial: Tìm thấy {len(ports)} cổng:")
        for p in ports:
            print(f"  [+] {p}")
        return

    # 1. Khởi tạo bộ điều khiển Servo
    servo = ServoController(port=args.port, baudrate=args.baud)

    # 2. Kết nối Camera USB
    try:
        cap = open_usb_camera(cam_index=args.cam)
    except Exception as e:
        print(f"\n[LỖI] {e}")
        print("\n💡 Gợi ý: Hãy chạy lại lệnh sau để kiểm tra camera:")
        print("   python scripts/usb_cam_and_servo.py --scan")
        return

    window_name = "Shopformer - Camera USB & Dieu Khien Servo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 600)

    print("\n" + "=" * 65)
    print("🚀 HỆ THỐNG ĐÃ SẴN SÀNG:")
    print("   - Nhấn phím 'L': Khóa cửa (Servo 90°, tự mở sau 45s)")
    print("   - Nhấn phím 'U': Mở cửa ngay (Servo 0°)")
    print("   - Nhấn phím '1': Xoay góc 45°")
    print("   - Nhấn phím '2': Xoay góc 180°")
    print("   - Nhấn phím 'Q' hoặc ESC: Dừng chương trình")
    print("=" * 65 + "\n")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[CAMERA] Không nhận được khung hình từ camera USB...")
                time.sleep(0.05)
                continue

            # Tính toán FPS hiển thị
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Cập nhật trạng thái Servo (tự mở nếu hết TTL)
            servo.update()

            # Vẽ thanh thông tin trực quan lên khung hình
            draw_hud(frame, servo, fps, args.cam)

            # Hiển thị ra màn hình
            cv2.imshow(window_name, frame)

            # Bắt phím điều khiển (chờ 1ms)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), ord('Q'), 27]:  # Phím Q hoặc ESC
                print("[SYSTEM] Đang thoát chương trình...")
                break
            elif key in [ord('l'), ord('L')]:
                servo.lock(ttl_sec=45, reason="Phím tắt 'L'")
            elif key in [ord('u'), ord('U')]:
                servo.unlock(reason="Phím tắt 'U'")
            elif key == ord('1'):
                servo.set_angle(45)
            elif key == ord('2'):
                servo.set_angle(180)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Nhận tín hiệu ngắt bàn phím (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        servo.close()
        print("[SYSTEM] Hoàn tất dọn dẹp tài nguyên. Tạm biệt!\n")


if __name__ == "__main__":
    main()
