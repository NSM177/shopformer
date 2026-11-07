"""Model bundle manager for `checkpoints/shopformer_v1/{model.pt, calibration.json, preprocess.json, meta.json}`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from shoplift_iot.config import PROJECT_ROOT, load_app_config


@dataclass(slots=True)
class ModelBundle:
    """Self-contained model bundle packaging weights, threshold calibration, and preprocessing config."""

    bundle_dir: Path
    model_path: Path
    calibration: dict[str, Any]
    preprocess: dict[str, Any]
    meta: dict[str, Any]

    @property
    def threshold(self) -> float:
        """Return operational anomaly score threshold bundled with this model."""
        if "operational_threshold" in self.calibration:
            return float(self.calibration["operational_threshold"])
        return float(self.calibration.get("threshold", 0.30))

    @property
    def calibrated_p99_threshold(self) -> float:
        """Return raw 99th-percentile normal training window MSE threshold."""
        return float(self.calibration.get("threshold", self.threshold))

    @property
    def model_config(self) -> dict[str, Any]:
        """Return model architecture hyperparameters stored in `meta.json`."""
        return dict(self.meta.get("model", {}))

    @classmethod
    def export_from_legacy(
        cls,
        bundle_dir: Path | str,
        cfg: Mapping[str, Any] | None = None,
        project_root: Path = PROJECT_ROOT,
    ) -> ModelBundle:
        """Build `shopformer_v1` bundle directory from existing checkpoints and config."""
        app_cfg = load_app_config(cfg, project_root=project_root)
        b_dir = app_cfg.resolve_path(bundle_dir)
        b_dir.mkdir(parents=True, exist_ok=True)

        gcae_ckpt = app_cfg.resolve_path(app_cfg.paths["gcae_ckpt"])
        shopformer_ckpt = app_cfg.resolve_path(app_cfg.paths["shopformer_ckpt"])
        th_file = app_cfg.resolve_path(app_cfg.paths["threshold_file"])
        eval_file = app_cfg.resolve_path(app_cfg.paths["checkpoints_dir"]) / "eval_metrics.json"

        calibration_data: dict[str, Any] = {
            "operational_threshold": float(app_cfg.alerting.get("default_threshold", 0.30)),
            "calibration_percentile": float(app_cfg.alerting.get("calibration_percentile", 99.0)),
            "use_calibrated_p99": bool(app_cfg.alerting.get("use_threshold_file", False)),
        }
        if th_file.exists():
            with open(th_file, "r", encoding="utf-8") as f:
                calibration_data.update(json.load(f))
        if calibration_data.get("use_calibrated_p99") and "threshold" in calibration_data:
            calibration_data["operational_threshold"] = float(calibration_data["threshold"])

        preprocess_data: dict[str, Any] = {
            **dict(app_cfg.preprocess),
            "vid_res": list(app_cfg.video.get("vid_res", [1920, 1080])),
            "target_fps": float(app_cfg.video.get("target_fps", 15)),
            "swap_yx_from_poselift_json": bool(
                app_cfg.video.get("swap_yx_from_poselift_json", True)
            ),
        }

        eval_metrics: dict[str, Any] = {}
        if eval_file.exists():
            with open(eval_file, "r", encoding="utf-8") as f:
                eval_metrics = json.load(f)

        meta_data: dict[str, Any] = {
            "bundle_name": b_dir.name,
            "version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": dict(app_cfg.model),
            "eval_metrics": eval_metrics,
            "source_checkpoints": {
                "gcae_ckpt": str(app_cfg.paths["gcae_ckpt"]),
                "shopformer_ckpt": str(app_cfg.paths["shopformer_ckpt"]),
            },
        }

        cal_path = b_dir / "calibration.json"
        prep_path = b_dir / "preprocess.json"
        meta_path = b_dir / "meta.json"
        model_path = b_dir / "model.pt"

        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(calibration_data, f, indent=2)
        with open(prep_path, "w", encoding="utf-8") as f:
            json.dump(preprocess_data, f, indent=2)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        gcae_state: dict[str, Any] | None = None
        transformer_state: dict[str, Any] | None = None
        if gcae_ckpt.exists() or shopformer_ckpt.exists():
            import torch

            if gcae_ckpt.exists():
                raw_g = torch.load(gcae_ckpt, map_location="cpu", weights_only=False)
                gcae_state = raw_g.get("state_dict", raw_g)
            if shopformer_ckpt.exists():
                raw_t = torch.load(shopformer_ckpt, map_location="cpu", weights_only=False)
                transformer_state = raw_t.get("state_dict", raw_t)

            if gcae_state is not None or transformer_state is not None:
                torch.save(
                    {
                        "gcae_state_dict": gcae_state,
                        "transformer_state_dict": transformer_state,
                        "bundle_version": "1.0.0",
                    },
                    model_path,
                )

        return cls(
            bundle_dir=b_dir,
            model_path=model_path,
            calibration=calibration_data,
            preprocess=preprocess_data,
            meta=meta_data,
        )

    @classmethod
    def load(
        cls,
        bundle_dir: Path | str,
        cfg: Mapping[str, Any] | None = None,
        project_root: Path = PROJECT_ROOT,
    ) -> ModelBundle:
        """Load `ModelBundle` from `bundle_dir`, auto-exporting from legacy files if not yet built."""
        b_dir = Path(bundle_dir)
        if not b_dir.is_absolute():
            b_dir = project_root / b_dir

        cal_path = b_dir / "calibration.json"
        prep_path = b_dir / "preprocess.json"
        meta_path = b_dir / "meta.json"
        model_path = b_dir / "model.pt"

        if not (cal_path.exists() and prep_path.exists() and meta_path.exists()):
            return cls.export_from_legacy(b_dir, cfg=cfg, project_root=project_root)

        with open(cal_path, "r", encoding="utf-8") as f:
            calibration_data = json.load(f)
        with open(prep_path, "r", encoding="utf-8") as f:
            preprocess_data = json.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        # If model.pt is gitignored and missing on disk, rebuild it from legacy checkpoints if present
        if not model_path.exists():
            src_ckpts = meta_data.get("source_checkpoints", {})
            gcae_rel = src_ckpts.get("gcae_ckpt", "checkpoints/gcae_best.pt")
            shop_rel = src_ckpts.get("shopformer_ckpt", "checkpoints/shopformer_best.pt")
            gcae_ckpt = project_root / gcae_rel
            shop_ckpt = project_root / shop_rel
            if gcae_ckpt.exists() or shop_ckpt.exists():
                import torch

                gcae_state = (
                    torch.load(gcae_ckpt, map_location="cpu", weights_only=False).get("state_dict")
                    if gcae_ckpt.exists()
                    else None
                )
                transformer_state = (
                    torch.load(shop_ckpt, map_location="cpu", weights_only=False).get("state_dict")
                    if shop_ckpt.exists()
                    else None
                )
                torch.save(
                    {
                        "gcae_state_dict": gcae_state,
                        "transformer_state_dict": transformer_state,
                        "bundle_version": meta_data.get("version", "1.0.0"),
                    },
                    model_path,
                )

        return cls(
            bundle_dir=b_dir,
            model_path=model_path,
            calibration=calibration_data,
            preprocess=preprocess_data,
            meta=meta_data,
        )
