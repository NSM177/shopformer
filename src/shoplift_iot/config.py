"""Centralized configuration loader (`configs/default.yaml` is read once here and passed down)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass(slots=True)
class AppConfig(Mapping[str, Any]):
    """Typed and mapping-compatible application configuration loaded once from YAML."""

    raw: dict[str, Any] = field(default_factory=dict)
    project_root: Path = PROJECT_ROOT
    config_path: Path = DEFAULT_CONFIG_PATH

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw["paths"]

    @property
    def video(self) -> dict[str, Any]:
        return self.raw["video"]

    @property
    def pose_track(self) -> dict[str, Any]:
        return self.raw["pose_track"]

    @property
    def preprocess(self) -> dict[str, Any]:
        return self.raw["preprocess"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def alerting(self) -> dict[str, Any]:
        return self.raw["alerting"]

    @property
    def mqtt(self) -> dict[str, Any]:
        return self.raw["mqtt"]

    @property
    def dashboard(self) -> dict[str, Any]:
        return self.raw["dashboard"]

    @property
    def pipeline(self) -> dict[str, Any]:
        return self.raw.get(
            "pipeline",
            {
                "capture_queue_size": 2,
                "score_queue_size": 32,
                "reconnect_delay_seconds": 1.0,
                "max_reconnect_attempts": 3,
                "pre_event_seconds": 3.0,
                "post_event_seconds": 2.0,
                "status_interval_seconds": 5.0,
                "outbox_poll_interval_seconds": 0.5,
                "outbox_max_retries": 10,
            },
        )

    def resolve_path(self, relative_or_abs: str | Path) -> Path:
        """Resolve a path relative to `project_root` if it is not already absolute."""
        p = Path(relative_or_abs)
        return p if p.is_absolute() else (self.project_root / p)

    @property
    def bundle_dir(self) -> Path:
        """Return resolved path to the Shopformer model bundle directory."""
        bundle_rel = self.paths.get("bundle_dir", "checkpoints/shopformer_v1")
        return self.resolve_path(bundle_rel)


def load_app_config(
    config_path: Path | str | AppConfig | Mapping[str, Any] | None = None,
    project_root: Path | None = None,
) -> AppConfig:
    """Load `configs/default.yaml` once into an `AppConfig` instance (or return existing one)."""
    root = project_root or PROJECT_ROOT
    if isinstance(config_path, AppConfig):
        return config_path
    if isinstance(config_path, Mapping):
        return AppConfig(raw=dict(config_path), project_root=root, config_path=DEFAULT_CONFIG_PATH)

    cfg_file = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not cfg_file.is_absolute():
        candidate = root / cfg_file
        if candidate.exists():
            cfg_file = candidate

    with open(cfg_file, "r", encoding="utf-8") as f:
        raw_dict: dict[str, Any] = yaml.safe_load(f)

    return AppConfig(raw=raw_dict, project_root=root, config_path=cfg_file.resolve())

