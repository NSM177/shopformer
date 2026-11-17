#!/usr/bin/env python3
"""GD5: Giả lập cửa từ ESP32 nhận lệnh từ `store/{store_id}/door/cmd` và báo `store/{store_id}/door/status`."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shoplift_iot.actuators.door_controller import DoorController  # noqa: E402
from shoplift_iot.contracts import DoorStatus  # noqa: E402


def _print_banner(status: DoorStatus, detail: str) -> None:
    is_locked = (status.state == "locked")
    icon = "🔒 KHÓA (LOCKED)" if is_locked else "🔓 MỞ (UNLOCKED - FAIL-SAFE)"
    servo_angle = "90° (ĐÃ XOAY ĐÓNG CHỐT CỬA)" if is_locked else "0° (ĐÃ XOAY MỞ CHỐT CỬA)"
    print(
        f"\n============================================================\n"
        f"[ESP32 SERVO DOOR] Trạng thái: {icon}\n"
        f"  Động cơ Servo : {servo_angle}\n"
        f"  Thời gian     : {status.ts}\n"
        f"  Sự kiện       : {status.event_id or '(không có)'}\n"
        f"  Chi tiết      : {detail}\n"
        f"============================================================",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Giả lập bộ điều khiển cửa từ ESP32 qua MQTT.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="File cấu hình YAML.")
    parser.add_argument("--store-id", type=str, default=None, help="Ghi đè store_id.")
    parser.add_argument("--mqtt-host", "--host", dest="mqtt_host", type=str, default=None, help="Ghi đè địa chỉ MQTT broker.")
    parser.add_argument("--mqtt-port", "--port", dest="mqtt_port", type=int, default=None, help="Ghi đè cổng MQTT broker.")
    parser.add_argument("--duration", type=float, default=None, help="Số giây chạy thử (mặc định chạy vô hạn).")
    args = parser.parse_args()

    controller = DoorController.from_config(
        args.config,
        auto_connect=False,
        on_state_change=_print_banner,
    )
    if args.store_id:
        controller.store_id = args.store_id
        controller.cmd_topic = f"store/{controller.store_id}/door/cmd"
        controller.status_topic = f"store/{controller.store_id}/door/status"
    if args.mqtt_host:
        controller.broker_host = args.mqtt_host
    if args.mqtt_port:
        controller.broker_port = int(args.mqtt_port)

    running = True

    def _stop_handler(_sig: int, _frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    connected = controller.connect()
    print(
        f"[fake_door] Bắt đầu giả lập ESP32 | broker={controller.broker_host}:{controller.broker_port} "
        f"(connected={connected}) | sub={controller.cmd_topic} | pub={controller.status_topic}",
        flush=True,
    )
    _print_banner(DoorStatus.create(state=controller.state, event_id=""), "Khởi động ở trạng thái MỞ (fail-safe)")

    t0 = time.monotonic()
    try:
        while running:
            now = time.monotonic()
            if args.duration is not None and (now - t0) >= args.duration:
                break
            controller.tick(now=now)
            time.sleep(0.2)
    finally:
        controller.close()
        print("[fake_door] Đã dừng và đưa cửa về trạng thái MỞ (fail-safe).", flush=True)


if __name__ == "__main__":
    main()

