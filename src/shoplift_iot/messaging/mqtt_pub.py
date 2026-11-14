"""Step 8: Publish anomaly alert events (QoS 1), Last Will, and status telemetry over MQTT."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import paho.mqtt.client as mqtt

from shoplift_iot.config import AppConfig, load_app_config
from shoplift_iot.contracts import AlertEvent

REQUIRED_KEYS = (
    "event_id",
    "ts",
    "cam_id",
    "track_id",
    "score",
    "threshold",
    "clip_path",
)


class MQTTPublisher:
    """Publish AlertEvent JSON messages to `store/{store_id}/cam/{cam_id}/alert` and status telemetry."""

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        store_id: str = "store_01",
        cam_id: str = "cam_01",
        topic_template: str = "store/{store_id}/cam/{cam_id}/alert",
        status_topic_template: str = "store/{store_id}/cam/{cam_id}/status",
        keepalive: int = 60,
        auto_connect: bool = True,
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = int(broker_port)
        self.store_id = store_id
        self.cam_id = cam_id
        self.topic_template = topic_template
        self.status_topic_template = status_topic_template
        self.keepalive = int(keepalive)
        self.topic = self.topic_template.format(store_id=self.store_id, cam_id=self.cam_id)
        self.status_topic = self.status_topic_template.format(
            store_id=self.store_id, cam_id=self.cam_id
        )

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.will_set(
            self.status_topic,
            payload="offline",
            qos=1,
            retain=True,
        )
        self._connected = False
        self.published_history: list[dict[str, Any]] = []
        self.status_history: list[str] = []

        if auto_connect:
            self.connect()

    @classmethod
    def from_config(
        cls,
        config_or_path: Path | str | AppConfig | Mapping[str, Any],
        auto_connect: bool = True,
    ) -> MQTTPublisher:
        """Create MQTTPublisher from application configuration."""
        cfg = load_app_config(config_or_path).mqtt
        return cls(
            broker_host=str(cfg["broker_host"]),
            broker_port=int(cfg["broker_port"]),
            store_id=str(cfg["store_id"]),
            cam_id=str(cfg["cam_id"]),
            topic_template=str(cfg["topic_template"]),
            status_topic_template=str(
                cfg.get("status_topic_template", "store/{store_id}/cam/{cam_id}/status")
            ),
            keepalive=int(cfg.get("keepalive", 60)),
            auto_connect=auto_connect,
        )

    def connect(self) -> bool:
        """Connect to the Mosquitto MQTT broker, returning True on success."""
        try:
            self._client.connect(self.broker_host, self.broker_port, keepalive=self.keepalive)
            self._client.loop_start()
            self._connected = True
            self.publish_status("online")
            return True
        except Exception as exc:
            self._connected = False
            print(
                f"[MQTTPublisher] Warning: Could not connect to {self.broker_host}:{self.broker_port} ({exc})"
            )
            return False

    @staticmethod
    def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Ensure payload has exact required keys and no raw image/video data."""
        clean: dict[str, Any] = {}
        for key in REQUIRED_KEYS:
            if key not in payload:
                raise ValueError(f"Missing required MQTT payload field: '{key}'")
            clean[key] = payload[key]
        return clean

    def publish_event(self, event: AlertEvent | dict[str, Any]) -> bool:
        """Publish an AlertEvent or dict to the configured MQTT alert topic with QoS 1."""
        raw_dict = event.to_dict() if isinstance(event, AlertEvent) else dict(event)
        payload_dict = self.validate_payload(raw_dict)
        topic = self.topic_template.format(
            store_id=self.store_id,
            cam_id=payload_dict.get("cam_id", self.cam_id),
        )
        payload_json = json.dumps(payload_dict, ensure_ascii=False)
        self.published_history.append(payload_dict)

        if not self._connected:
            self.connect()

        if self._connected:
            try:
                info = self._client.publish(topic, payload=payload_json, qos=1)
                info.wait_for_publish(timeout=2.0)
                return info.is_published()
            except Exception as exc:
                print(f"[MQTTPublisher] Publish error: {exc}")
                return False
        return False

    def publish_status(self, state: str = "online") -> bool:
        """Publish camera status ('online' or 'offline') to `store/{store_id}/cam/{cam_id}/status` (retained)."""
        if state not in ("online", "offline"):
            raise ValueError("Camera status must be 'online' or 'offline'")
        self.status_history.append(state)

        if not self._connected:
            return False

        try:
            info = self._client.publish(
                self.status_topic,
                payload=state,
                qos=1,
                retain=True,
            )
            info.wait_for_publish(timeout=2.0)
            return info.is_published()
        except Exception:
            return False

    def close(self) -> None:
        """Publish 'offline' status, stop background network loop, and disconnect cleanly."""
        if self._connected:
            try:
                self.publish_status("offline")
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._connected = False


def main() -> None:
    """CLI smoke test for MQTTPublisher."""
    parser = argparse.ArgumentParser(description="Publish a test alert event over MQTT.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    pub = MQTTPublisher.from_config(Path(args.config), auto_connect=True)
    sample_event = AlertEvent.create(
        cam_id=pub.cam_id,
        track_id=1,
        score=0.8123,
        threshold=0.25,
        clip_path="data/clips/evt_smoke_test_001.mp4",
    )
    sent = pub.publish_event(sample_event)
    pub.publish_status("online")
    print(f"[MQTTPublisher] Topic: {pub.topic} | Broker sent: {sent} | Payload: {sample_event.to_dict()}")
    pub.close()


if __name__ == "__main__":
    main()
