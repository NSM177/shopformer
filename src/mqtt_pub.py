"""Step 8: Publish anomaly alert events over MQTT without raw images or video."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import paho.mqtt.client as mqtt
import yaml

from src.alerting import AlertEvent

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
    """Publish AlertEvent JSON messages to `store/{store_id}/cam/{cam_id}/alert`."""

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        store_id: str = "store_01",
        cam_id: str = "cam_01",
        topic_template: str = "store/{store_id}/cam/{cam_id}/alert",
        keepalive: int = 60,
        auto_connect: bool = True,
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = int(broker_port)
        self.store_id = store_id
        self.cam_id = cam_id
        self.topic_template = topic_template
        self.keepalive = int(keepalive)
        self.topic = self.topic_template.format(store_id=self.store_id, cam_id=self.cam_id)

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._connected = False
        self.published_history: list[dict[str, Any]] = []

        if auto_connect:
            self.connect()

    @classmethod
    def from_config(cls, config_path: Path, auto_connect: bool = True) -> MQTTPublisher:
        """Create MQTTPublisher from YAML configuration file."""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["mqtt"]
        return cls(
            broker_host=str(cfg["broker_host"]),
            broker_port=int(cfg["broker_port"]),
            store_id=str(cfg["store_id"]),
            cam_id=str(cfg["cam_id"]),
            topic_template=str(cfg["topic_template"]),
            keepalive=int(cfg.get("keepalive", 60)),
            auto_connect=auto_connect,
        )

    def connect(self) -> bool:
        """Connect to the Mosquitto MQTT broker, returning True on success."""
        try:
            self._client.connect(self.broker_host, self.broker_port, keepalive=self.keepalive)
            self._client.loop_start()
            self._connected = True
            return True
        except Exception as exc:
            self._connected = False
            print(f"[MQTTPublisher] Warning: Could not connect to {self.broker_host}:{self.broker_port} ({exc})")
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
        """Publish an AlertEvent or dict to the configured MQTT topic."""
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

    def close(self) -> None:
        """Stop background network loop and disconnect cleanly."""
        if self._connected:
            try:
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
    sample_event = AlertEvent(
        event_id="evt_smoke_test_001",
        ts=datetime.now(timezone.utc).isoformat(),
        cam_id=pub.cam_id,
        track_id=1,
        score=0.8123,
        threshold=0.25,
        clip_path="data/clips/evt_smoke_test_001.mp4",
    )
    sent = pub.publish_event(sample_event)
    print(f"[MQTTPublisher] Topic: {pub.topic} | Broker sent: {sent} | Payload: {sample_event.to_dict()}")
    pub.close()


if __name__ == "__main__":
    main()

