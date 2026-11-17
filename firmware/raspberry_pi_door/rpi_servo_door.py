#!/usr/bin/env python3
"""Điều khiển Động cơ Servo khóa cửa trên Raspberry Pi cho hệ thống Shopformer.

Sơ đồ nối chân Servo (SG90 / MG996R) với Raspberry Pi 3/4/5:
  - Dây Vàng / Cam (PWM Signal) : Nối vào chân GPIO 18 (Chân vật lý số 12 - Hardware PWM)
  - Dây Đỏ (VCC 5V)              : Nối vào chân 5V (Chân vật lý số 2 hoặc 4)
  - Dây Nâu / Đen (GND)          : Nối vào chân GND (Chân vật lý số 6, 9, 14, 20...)
  * Lưu ý: Nếu dùng Servo tải nặng (MG996R), nên dùng nguồn 5V riêng cho Servo và nối chung mass GND với Raspberry Pi.

Hợp đồng MQTT:
  - Nhận lệnh: store/{store_id}/door/cmd
      JSON: {"action": "lock"|"unlock", "ttl_sec": 45, "event_id": "...", "reason": "..."}
  - Gửi trạng thái: store/{store_id}/door/status (retained)
      JSON: {"state": "locked"|"unlocked", "ts": "...", "servo_angle": 0|90, "event_id": "..."}

Ràng buộc an toàn (Fail-safe):
  1. Khi khởi động: Servo luôn quay về góc 0° (MỞ CỬA).
  2. Khi ngắt script hoặc mất mạng: Servo luôn quay về góc 0°.
  3. Lệnh LOCK luôn có thời hạn ttl_sec (mặc định 45s); hết hạn tự xoay về 0°.
  4. Lệnh UNLOCK luôn được ưu tiên tuyệt đối.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import os
import signal
import sys
import threading
import time
from typing import Any, Optional

# Thử import các thư viện GPIO trên Raspberry Pi
HAVE_GPIOZERO = False
HAVE_RPI_GPIO = False

try:
    from gpiozero import AngularServo
    from gpiozero.pins.pigpio import PiGPIOFactory
    HAVE_GPIOZERO = True
except ImportError:
    pass

if not HAVE_GPIOZERO:
    try:
        import RPi.GPIO as GPIO
        HAVE_RPI_GPIO = True
    except ImportError:
        pass

# Thử import paho-mqtt
try:
    import paho.mqtt.client as mqtt
    HAVE_PAHO_MQTT = True
except ImportError:
    HAVE_PAHO_MQTT = False


# Múi giờ Việt Nam (UTC+7)
VN_TZ = timezone(timedelta(hours=7), name="Asia/Ho_Chi_Minh")


def get_vn_iso() -> str:
    return datetime.now(VN_TZ).isoformat(timespec="seconds")


class RPiServoHardware:
    """Điều khiển phần cứng Động cơ Servo trên Raspberry Pi qua GPIO PWM."""

    def __init__(self, pin: int = 18, min_angle: int = 0, max_angle: int = 180) -> None:
        self.pin = pin
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.current_angle = 0
        self.driver_type = "mock"
        self._servo = None
        self._pwm = None

        if HAVE_GPIOZERO:
            try:
                # Sử dụng gpiozero với dải xung chuẩn 500us - 2500us
                self._servo = AngularServo(
                    self.pin,
                    min_angle=self.min_angle,
                    max_angle=self.max_angle,
                    min_pulse_width=0.0005,
                    max_pulse_width=0.0025,
                )
                self.driver_type = "gpiozero"
                print(f"[HARDWARE] Đang sử dụng driver: gpiozero (BCM GPIO {self.pin})")
            except Exception as e:
                print(f"[HARDWARE] Lỗi khởi tạo gpiozero ({e}), chuyển sang RPi.GPIO...")

        if self.driver_type == "mock" and HAVE_RPI_GPIO:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                GPIO.setup(self.pin, GPIO.OUT)
                self._pwm = GPIO.PWM(self.pin, 50)  # Tần số 50Hz chuẩn Servo
                self._pwm.start(0)
                self.driver_type = "rpi_gpio"
                print(f"[HARDWARE] Đang sử dụng driver: RPi.GPIO (BCM GPIO {self.pin}, 50Hz)")
            except Exception as e:
                print(f"[HARDWARE] Lỗi khởi tạo RPi.GPIO ({e}), chuyển sang chế độ MÔ PHỎNG...")

        if self.driver_type == "mock":
            print(f"[HARDWARE] Chạy ở chế độ MÔ PHỎNG (Simulation Mode) - không có GPIO thực tế.")

        # Khởi động luôn ở góc 0 độ (MỞ)
        self.set_angle(0)

    def set_angle(self, angle: int) -> None:
        """Xoay Servo tới góc chỉ định (0° - 180°)."""
        angle = max(self.min_angle, min(self.max_angle, angle))
        self.current_angle = angle

        if self.driver_type == "gpiozero" and self._servo is not None:
            self._servo.angle = angle
            time.sleep(0.25)
            # Ngắt xung giữ để chống rung và giảm nhiệt cho servo nhỏ (SG90)
            self._servo.detach()

        elif self.driver_type == "rpi_gpio" and self._pwm is not None:
            # Duty cycle: 0 độ ~ 2.5%, 180 độ ~ 12.5%
            duty = 2.5 + (angle / 180.0) * 10.0
            self._pwm.ChangeDutyCycle(duty)
            time.sleep(0.25)
            self._pwm.ChangeDutyCycle(0)  # Ngắt xung chống rung

        print(f"[SERVO HARDWARE] Đã xoay Servo về góc: {angle}° ({self.driver_type})")

    def cleanup(self) -> None:
        """Dọn dẹp tài nguyên GPIO an toàn."""
        self.set_angle(0)
        if self.driver_type == "gpiozero" and self._servo is not None:
            self._servo.close()
        elif self.driver_type == "rpi_gpio" and self._pwm is not None:
            self._pwm.stop()
            GPIO.cleanup()
        print("[HARDWARE] Đã giải phóng chân GPIO.")


class RPiDoorController:
    """Quản lý trạng thái khóa cửa, hẹn giờ tự mở (TTL), và tích hợp MQTT."""

    def __init__(
        self,
        gpio_pin: int = 18,
        store_id: str = "store_01",
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
        default_ttl_sec: int = 45,
    ) -> None:
        self.gpio_pin = gpio_pin
        self.store_id = store_id
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.default_ttl_sec = default_ttl_sec

        self.hw = RPiServoHardware(pin=gpio_pin)
        self.state = "unlocked"  # 'unlocked' | 'locked'
        self.current_event_id = ""
        self.lock_deadline = 0.0
        self.running = True
        self._lock = threading.Lock()

        # MQTT Topics
        self.cmd_topic = f"store/{self.store_id}/door/cmd"
        self.status_topic = f"store/{self.store_id}/door/status"

        self.mqtt_client: Optional[mqtt.Client] = None
        if HAVE_PAHO_MQTT:
            self._setup_mqtt()
        else:
            print("[MQTT] Cảnh báo: Thư viện 'paho-mqtt' chưa được cài đặt.")
            print("       Chạy lệnh: 'pip install paho-mqtt' để kết nối với Dashboard.")

    def _setup_mqtt(self) -> None:
        client_id = f"rpi-door-{self.store_id}-{int(time.time())}"
        self.mqtt_client = mqtt.Client(client_id=client_id, clean_session=True)

        # LWT: Nếu Raspberry Pi mất mạng đột ngột, broker tự phát thông báo mở cửa
        lwt_payload = json.dumps({
            "state": "unlocked",
            "ts": get_vn_iso(),
            "event_id": "",
            "servo_angle": 0,
            "note": "rpi_disconnect_fail_safe",
        })
        self.mqtt_client.will_set(self.status_topic, lwt_payload, qos=1, retain=True)

        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect

    def _on_mqtt_connect(self, client: Any, userdata: Any, flags: Any, rc: int) -> None:
        if rc == 0:
            print(f"[MQTT] Đã kết nối MQTT Broker tại {self.mqtt_host}:{self.mqtt_port}")
            client.subscribe(self.cmd_topic, qos=1)
            print(f"[MQTT] Đã lắng nghe lệnh trên topic: {self.cmd_topic}")
            self.publish_status()
        else:
            print(f"[MQTT] Kết nối MQTT thất bại với mã lỗi rc={rc}")

    def _on_mqtt_disconnect(self, client: Any, userdata: Any, rc: int) -> None:
        print(f"[MQTT] Bị ngắt kết nối MQTT (rc={rc}). Đang tự mở cửa an toàn (Fail-safe)...")
        self.unlock("mqtt_disconnected")

    def _on_mqtt_message(self, client: Any, userdata: Any, msg: Any) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            action = str(payload.get("action", "")).lower()
            reason = str(payload.get("reason", "mqtt_cmd"))
            event_id = str(payload.get("event_id", ""))
            ttl_sec = int(payload.get("ttl_sec", self.default_ttl_sec))

            print(f"[MQTT RX] Lệnh: {action.upper()} | Lý do: {reason} | TTL: {ttl_sec}s | Event: {event_id}")

            if action == "unlock":
                self.unlock(reason=reason, event_id=event_id)
            elif action == "lock":
                self.lock(ttl_sec=ttl_sec, reason=reason, event_id=event_id)
            else:
                print(f"[MQTT] Lệnh không hợp lệ: {action}. Bỏ qua.")
        except Exception as e:
            print(f"[MQTT] Lỗi phân tích gói tin JSON: {e}")
            self.unlock("json_parse_error")

    def publish_status(self) -> None:
        """Gửi trạng thái cửa và góc Servo lên MQTT Dashboard."""
        if not self.mqtt_client or not self.mqtt_client.is_connected():
            return
        payload = {
            "state": self.state,
            "ts": get_vn_iso(),
            "event_id": self.current_event_id,
            "servo_angle": self.hw.current_angle,
        }
        try:
            self.mqtt_client.publish(
                self.status_topic,
                json.dumps(payload),
                qos=1,
                retain=True,
            )
        except Exception as e:
            print(f"[MQTT] Lỗi khi gửi status: {e}")

    def unlock(self, reason: str = "manual", event_id: str = "") -> None:
        """Mở cửa ngay lập tức: Quay Servo về 0°."""
        with self._lock:
            self.state = "unlocked"
            self.current_event_id = event_id
            self.lock_deadline = 0.0
            self.hw.set_angle(0)
            print(f"[DOOR] 🔓 CỬA ĐANG MỞ (Servo 0°) - Lý do: {reason}")
            self.publish_status()

    def lock(self, ttl_sec: int = 45, reason: str = "manual", event_id: str = "") -> None:
        """Khóa cửa: Quay Servo sang 90°, tự động mở lại sau ttl_sec giây."""
        with self._lock:
            ttl_sec = max(5, min(180, ttl_sec))
            self.state = "locked"
            self.current_event_id = event_id
            self.lock_deadline = time.time() + ttl_sec
            self.hw.set_angle(90)
            print(f"[DOOR] 🔒 CỬA ĐÃ KHÓA (Servo 90°) - Tự mở sau {ttl_sec}s - Lý do: {reason}")
            self.publish_status()

    def update_loop(self) -> None:
        """Vòng lặp giám sát thời gian TTL để tự động mở khóa."""
        while self.running:
            if self.state == "locked" and self.lock_deadline > 0:
                if time.time() >= self.lock_deadline:
                    self.unlock(reason="ttl_expired")
            time.sleep(0.2)

    def start(self) -> None:
        """Khởi động tiến trình MQTT và giám sát."""
        if self.mqtt_client:
            try:
                self.mqtt_client.connect(self.mqtt_host, self.mqtt_port, keepalive=30)
                self.mqtt_client.loop_start()
            except Exception as e:
                print(f"[MQTT] Không kết nối được tới Broker {self.mqtt_host}:{self.mqtt_port} ({e})")
                print("       Hệ thống vẫn hoạt động ở chế độ độc lập tại chỗ.")

        # Bật luồng đếm thời gian TTL
        self.timer_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.timer_thread.start()

    def stop(self) -> None:
        """Dừng tiến trình và dọn dẹp an toàn."""
        self.running = False
        self.unlock("system_shutdown")
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        self.hw.cleanup()
        print("[DOOR] Đã dừng bộ điều khiển cửa trên Raspberry Pi.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy bộ điều khiển Servo khóa cửa trên Raspberry Pi.")
    parser.add_argument("--pin", type=int, default=18, help="Chân GPIO BCM điều khiển Servo (mặc định: 18 - chân vật lý 12).")
    parser.add_argument("--store", type=str, default="store_01", help="Mã cửa hàng store_id (mặc định: store_01).")
    parser.add_argument("--mqtt-host", type=str, default="localhost", help="Địa chỉ MQTT Broker (VD: IP của laptop hoặc Raspberry Pi).")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="Cổng MQTT Broker (mặc định: 1883).")
    parser.add_argument("--ttl", type=int, default=45, help="Thời gian khóa tự động mở mặc định (giây).")
    parser.add_argument("--test", action="store_true", help="Chạy chế độ test quay Servo 0° -> 90° -> 0° rồi thoát.")
    args = parser.parse_args()

    # Chế độ tự kiểm tra góc quay Servo
    if args.test:
        print("\n🧪 Đang chạy chế độ kiểm tra phần cứng Servo trên Raspberry Pi...")
        hw = RPiServoHardware(pin=args.pin)
        print("1. Xoay góc 0° (Mở cửa)...")
        hw.set_angle(0)
        time.sleep(2)
        print("2. Xoay góc 90° (Khóa cửa)...")
        hw.set_angle(90)
        time.sleep(2)
        print("3. Xoay góc 0° (Mở cửa an toàn)...")
        hw.set_angle(0)
        time.sleep(1)
        hw.cleanup()
        print("✅ Kiểm tra hoàn tất! Động cơ Servo hoạt động bình thường.\n")
        return

    controller = RPiDoorController(
        gpio_pin=args.pin,
        store_id=args.store,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        default_ttl_sec=args.ttl,
    )

    def handle_signal(sig: Any, frame: Any) -> None:
        print("\n[SYSTEM] Nhận tín hiệu dừng chương trình (Ctrl+C)...")
        controller.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    controller.start()

    print("\n" + "=" * 65)
    print(f"🚀 RASPBERRY PI DOOR CONTROLLER ĐANG HOẠT ĐỘNG:")
    print(f"   - Chân GPIO điều khiển : BCM {args.pin} (Chân vật lý 12)")
    print(f"   - MQTT Broker          : {args.mqtt_host}:{args.mqtt_port}")
    print(f"   - Lắng nghe lệnh tại   : store/{args.store}/door/cmd")
    print(f"   - Báo cáo trạng thái   : store/{args.store}/door/status")
    print("=" * 65)
    print("💡 Phím bấm tương tác tại terminal Raspberry Pi:")
    print("   [L] + Enter : Khóa cửa (Servo 90°, TTL 45s)")
    print("   [U] + Enter : Mở cửa ngay lập tức (Servo 0°)")
    print("   [Q] + Enter : Thoát chương trình")
    print("=" * 65 + "\n")

    try:
        while True:
            # Nhận lệnh phím từ terminal để test nhanh tại chỗ
            line = sys.stdin.readline()
            if not line:
                break
            cmd = line.strip().upper()
            if cmd == "L":
                controller.lock(ttl_sec=args.ttl, reason="terminal_manual_input")
            elif cmd == "U":
                controller.unlock(reason="terminal_manual_input")
            elif cmd == "Q":
                break
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
