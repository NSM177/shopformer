"""GD6: Optional Telegram bot for sending shoplifting alerts and receiving admin `/lock` & `/unlock` commands."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping
import urllib.request

from shoplift_iot.contracts import DoorCommand


class TelegramAlertBot:
    """Optional Telegram bot reading token/chat_id from env vars (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`)."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        default_ttl_sec: int = 45,
        max_ttl_sec: int = 120,
        audit_log_path: Path | str | None = None,
        on_door_command: Callable[[DoorCommand, str], Any] | None = None,
        http_sender: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.bot_token = (
            bot_token if bot_token is not None else os.environ.get("TELEGRAM_BOT_TOKEN", "")
        ).strip()
        self.chat_id = (
            chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID", "")
        ).strip()
        self.default_ttl_sec = max(1, int(default_ttl_sec))
        self.max_ttl_sec = max(self.default_ttl_sec, int(max_ttl_sec))
        self.audit_log_path = Path(audit_log_path) if audit_log_path is not None else None
        self.on_door_command = on_door_command
        self._http_sender = http_sender or self._default_http_sender
        self.sent_messages: list[dict[str, Any]] = []

    @property
    def is_enabled(self) -> bool:
        """Return True only when both `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are configured."""
        return bool(self.bot_token and self.chat_id)

    @staticmethod
    def format_alert_caption(event: Mapping[str, Any]) -> str:
        """Format an alert payload into a concise Vietnamese Telegram notification."""
        event_id = str(event.get("event_id", "unknown"))
        cam_id = str(event.get("cam_id", "cam_01"))
        track_id = int(event.get("track_id", -1))
        score = float(event.get("score", 0.0))
        threshold = float(event.get("threshold", 0.0))
        ts = str(event.get("ts", ""))
        clip_path = str(event.get("clip_path", ""))
        return (
            f"🚨 [CẢNH BÁO NGHI TRỘM] Sự kiện: {event_id}\n"
            f"• Camera: {cam_id} | Track ID: #{track_id} (ẩn danh)\n"
            f"• Điểm: {score:.4f} (Ngưỡng: {threshold:.4f})\n"
            f"• Thời gian: {ts}\n"
            f"• Clip: {clip_path}\n\n"
            f"Lệnh điều khiển cửa (chỉ Admin):\n"
            f"👉 /lock {event_id} 45\n"
            f"👉 /unlock"
        )

    def _default_http_sender(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call Telegram Bot HTTP API (`sendMessage`) using standard library `urllib.request`."""
        url = f"https://api.telegram.org/bot{self.bot_token}/{method}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def send_alert(
        self,
        event: Mapping[str, Any],
        clip_file: Path | str | None = None,
    ) -> bool:
        """Send an alert notification to the configured admin Telegram chat."""
        if not self.is_enabled:
            return False

        caption = self.format_alert_caption(event)
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": caption,
        }
        if clip_file is not None and Path(clip_file).exists():
            payload["clip_exists"] = True
            payload["clip_file"] = str(clip_file)

        try:
            self._http_sender("sendMessage", payload)
            self.sent_messages.append(payload)
            return True
        except Exception as exc:
            print(f"[TelegramAlertBot] Warning: Failed to send Telegram alert ({exc})")
            return False

    def parse_command_text(self, text: str) -> DoorCommand | None:
        """Parse `/lock [event_id] [ttl_sec]` or `/unlock [event_id]` from admin Telegram message."""
        clean = str(text or "").strip()
        if not clean.startswith("/"):
            return None

        parts = clean.split()
        cmd_token = parts[0].lower().split("@")[0]

        if cmd_token in ("/unlock", "/mo", "/mocua"):
            event_id = parts[1] if len(parts) >= 2 else "telegram_manual"
            return DoorCommand(
                action="unlock",
                reason="admin_telegram:unlock",
                event_id=event_id,
                ttl_sec=0,
            )

        if cmd_token in ("/lock", "/khoa", "/khoacua"):
            event_id = "telegram_manual"
            ttl_sec = self.default_ttl_sec
            if len(parts) == 2:
                # Could be `/lock 60` or `/lock evt_123`
                if parts[1].isdigit():
                    ttl_sec = int(parts[1])
                else:
                    event_id = parts[1]
            elif len(parts) >= 3:
                event_id = parts[1]
                try:
                    ttl_sec = int(parts[2])
                except ValueError:
                    ttl_sec = self.default_ttl_sec

            ttl_sec = max(1, min(int(ttl_sec), self.max_ttl_sec))
            return DoorCommand(
                action="lock",
                reason="admin_telegram:lock",
                event_id=event_id,
                ttl_sec=ttl_sec,
            )

        return None

    def handle_update(self, update: Mapping[str, Any]) -> DoorCommand | None:
        """Process a Telegram update dict; authorize `chat_id`, log audit entry, and emit `DoorCommand`."""
        msg = update.get("message") or update.get("edited_message")
        if not isinstance(msg, Mapping):
            return None

        chat = msg.get("chat") or {}
        incoming_chat_id = str(chat.get("id", "")).strip()
        if self.chat_id and incoming_chat_id != self.chat_id:
            # Safety Constraint #1: Reject commands from unauthorized Telegram chats
            return None

        text = str(msg.get("text", "")).strip()
        cmd = self.parse_command_text(text)
        if cmd is None:
            return None

        sender = msg.get("from") or {}
        username = str(sender.get("username") or sender.get("id") or "admin_telegram")
        actor = f"telegram:{username}"

        if self.audit_log_path is not None:
            try:
                self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
                audit_entry = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "actor": actor,
                    "action": cmd.action,
                    "reason": cmd.reason,
                    "event_id": cmd.event_id,
                    "ttl_sec": cmd.ttl_sec,
                }
                with open(self.audit_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

        if self.on_door_command is not None:
            self.on_door_command(cmd, actor)

        return cmd

