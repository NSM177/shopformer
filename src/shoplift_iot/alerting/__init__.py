"""Alerting subpackage: temporal score smoothing, K-consecutive threshold, cooldown, and clip recorder."""

from shoplift_iot.alerting.alerting import AlertEvent, AlertManager
from shoplift_iot.alerting.clip_recorder import ClipRecorder

__all__ = ["AlertEvent", "AlertManager", "ClipRecorder"]
