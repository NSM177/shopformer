"""Two-stage training script for Shopformer (Stage 1: GCAE, Stage 2: Transformer)."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import trans_list  # noqa: E402
from src.scorer import (  # noqa: E402
    Encoder_T,
    GCAE,
    PoseSegDataset,
    TransformerPredictor,
    build_Models_from_config,
    calc_reg_loss,
    resolve_device,
    score_dataset,
    tokenize_batch,
)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_poselift_loaders(
    cfg: dict[str, Any], project_root: Path
) -> tuple[PoseSegDataset, PoseSegDataset, DataLoader, DataLoader]:
    """Create PoseLift train and test datasets and DataLoaders."""
    paths_cfg = cfg["paths"]
    prep_cfg = cfg["preprocess"]
    vid_cfg = cfg["video"]
    train_cfg = cfg["training"]

    train_json_dir = project_root / paths_cfg["pose_train_dir"]
    test_json_dir = project_root / paths_cfg["pose_test_dir"]
    vid_res = list(vid_cfg["vid_res"])
    swap_yx = bool(vid_cfg.get("swap_yx_from_poselift_json", True))
    num_transform = int(train_cfg.get("num_transform", 2))

    common_args = {
        "headless": bool(cfg["model"].get("headless", False)),
        "seg_len": int(prep_cfg["window_size"]),
        "dataset": "Poselift",
        "train_seg_conf_th": 0.0,
        "vid_res": vid_res,
        "symm_range": bool(prep_cfg.get("symm_range", False)),
        "swap_yx": swap_yx,
        "relative": False,
        "traj": False,
    }

    train_ds = PoseSegDataset(
        str(train_json_dir),
        normalize_pose_segs=True,
        evaluate=False,
        seg_stride=int(prep_cfg["window_stride_train"]),
        trans_list=trans_list[:num_transform] if num_transform > 1 else None,
        **common_args,
    )
    test_ds = PoseSegDataset(
        str(test_json_dir),
        normalize_pose_segs=True,
        evaluate=True,
        seg_stride=int(prep_cfg["window_stride_eval"]),
        trans_list=None,
        **common_args,
    )

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_ds, test_ds, train_loader, test_loader


def train_stage1_gcae(
    gcae: GCAE,
    train_loader: DataLoader,
    cfg: dict[str, Any],
    device: str,
    ckpt_path: Path,
    epochs_override: int | None = None,
) -> float:
    """Train Stage 1 GCAE unsupervised pose autoencoder on normal sequences."""
    train_cfg = cfg["training"]
    epochs = epochs_override if epochs_override is not None else int(train_cfg["gcae_epochs"])
    lr = float(train_cfg["gcae_lr"])
    wd = float(train_cfg["gcae_weight_decay"])
    alpha = float(train_cfg.get("gcae_reg_alpha", 1.0))

    optimizer = torch.optim.Adam(gcae.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    mse_fn = nn.MSELoss()

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(epochs):
        gcae.train()
        losses: list[float] = []
        t0 = time.time()
        for batch in tqdm(train_loader, desc=f"[Stage 1 GCAE] Epoch {epoch + 1}/{epochs}"):
            data = batch[0][:, 0:2, :, :].to(device=device, dtype=torch.float32)
            reco = gcae(data)
            if reco.ndim == 5 and reco.shape[-1] == 1:
                reco = reco.squeeze(-1)
            reco_loss = mse_fn(reco, data)
            reg_loss = calc_reg_loss(gcae).to(device)
            loss = reco_loss + 1e-3 * alpha * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        scheduler.step()
        mean_loss = float(np.mean(losses))
        print(
            f"[Stage 1 GCAE] Epoch {epoch + 1}/{epochs} - loss: {mean_loss:.6f} ({time.time() - t0:.1f}s)"
        )
        if mean_loss <= best_loss:
            best_loss = mean_loss
            torch.save({"epoch": epoch + 1, "state_dict": gcae.state_dict()}, ckpt_path)

    return best_loss


def evaluate_shopformer_lists(
    encoder: Encoder_T,
    transformer: TransformerPredictor,
    test_loader: DataLoader,
    device: str,
) -> np.ndarray:
    """Compute window-level MSE reconstruction scores on a DataLoader."""
    encoder.eval()
    transformer.eval()
    scores: list[float] = []
    with torch.no_grad():
        for batch in test_loader:
            data = batch[0][:, 0:2, :, :].to(device=device, dtype=torch.float32)
            tokens = tokenize_batch(encoder, data)
            recon = transformer(tokens, tokens)
            mse_per_sample = torch.mean((tokens - recon) ** 2, dim=(1, 2))
            scores.extend(mse_per_sample.cpu().numpy().tolist())
    return np.array(scores, dtype=np.float64)


def train_stage2_transformer(
    gcae: GCAE,
    transformer: TransformerPredictor,
    train_loader: DataLoader,
    test_loader: DataLoader,
    test_ds: PoseSegDataset,
    cfg: dict[str, Any],
    project_root: Path,
    device: str,
    ckpt_path: Path,
    epochs_override: int | None = None,
) -> dict[str, float]:
    """Freeze GCAE encoder as tokenizer and train Stage 2 Transformer encoder-decoder."""
    for param in gcae.parameters():
        param.requires_grad = False
    gcae.eval()
    encoder = Encoder_T(gcae).to(device)
    encoder.eval()

    train_cfg = cfg["training"]
    epochs = (
        epochs_override if epochs_override is not None else int(train_cfg["transformer_epochs"])
    )
    lr = float(train_cfg["transformer_lr"])
    wd = float(train_cfg["transformer_weight_decay"])

    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    eval_args = argparse.Namespace(
        mask_root=str(project_root / cfg["paths"]["gt_mask_dir"]),
        dataset="Poselift",
        seg_len=int(cfg["preprocess"]["window_size"]),
        save_results=False,
        save_results_dir="",
    )

    best_auc = -1.0
    best_metrics: dict[str, float] = {}

    for epoch in range(epochs):
        transformer.train()
        losses: list[float] = []
        t0 = time.time()
        for batch in tqdm(train_loader, desc=f"[Stage 2 Transformer] Epoch {epoch + 1}/{epochs}"):
            data = batch[0][:, 0:2, :, :].to(device=device, dtype=torch.float32)
            tokens = tokenize_batch(encoder, data)
            recon = transformer(tokens, tokens)
            loss_per_sample = torch.mean((tokens - recon) ** 2, dim=(1, 2))
            loss = loss_per_sample.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        scheduler.step()
        mean_loss = float(np.mean(losses))
        scores = evaluate_shopformer_lists(encoder, transformer, test_loader, device)
        auc_roc, auc_pr, eer, eer_th, _, _ = score_dataset(scores, test_ds.metadata, args=eval_args)
        print(
            f"[Stage 2 Transformer] Epoch {epoch + 1}/{epochs} - train_mse: {mean_loss:.6f} | "
            f"AUC-ROC: {auc_roc * 100:.2f}% | AUC-PR: {auc_pr * 100:.2f}% | EER: {eer:.4f} ({time.time() - t0:.1f}s)"
        )
        if auc_roc >= best_auc:
            best_auc = float(auc_roc)
            best_metrics = {
                "auc_roc": float(auc_roc * 100.0),
                "auc_pr": float(auc_pr * 100.0),
                "eer": float(eer),
                "eer_threshold": float(eer_th),
                "train_mse": mean_loss,
            }
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": transformer.state_dict(),
                    "metrics": best_metrics,
                },
                ckpt_path,
            )

    return best_metrics


def main() -> None:
    """CLI entrypoint for training Shopformer."""
    parser = argparse.ArgumentParser(description="Train two-stage Shopformer on PoseLift.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--gcae-epochs", type=int, default=None, help="Override Stage 1 epochs.")
    parser.add_argument(
        "--transformer-epochs", type=int, default=None, help="Override Stage 2 epochs."
    )
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda:0).")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    cfg = load_config(config_path)
    set_seed(int(cfg["training"]["seed"]))

    requested_device = args.device or str(cfg["pose_track"]["device"])
    device = resolve_device(requested_device)
    print(f"[INFO] Using device: {device}")

    _, test_ds, train_loader, test_loader = build_poselift_loaders(cfg, PROJECT_ROOT)
    gcae, transformer = build_Models_from_config(cfg, device)

    gcae_ckpt = PROJECT_ROOT / cfg["paths"]["gcae_ckpt"]
    shopformer_ckpt = PROJECT_ROOT / cfg["paths"]["shopformer_ckpt"]

    train_stage1_gcae(
        gcae=gcae,
        train_loader=train_loader,
        cfg=cfg,
        device=device,
        ckpt_path=gcae_ckpt,
        epochs_override=args.gcae_epochs,
    )
    gcae_state = torch.load(gcae_ckpt, map_location=device, weights_only=False)
    gcae.load_state_dict(gcae_state["state_dict"])

    metrics = train_stage2_transformer(
        gcae=gcae,
        transformer=transformer,
        train_loader=train_loader,
        test_loader=test_loader,
        test_ds=test_ds,
        cfg=cfg,
        project_root=PROJECT_ROOT,
        device=device,
        ckpt_path=shopformer_ckpt,
        epochs_override=args.transformer_epochs,
    )
    print("\n=== Training Complete ===")
    print(
        f"Best Test Metrics -> AUC-ROC: {metrics['auc_roc']:.2f}% | "
        f"AUC-PR: {metrics['auc_pr']:.2f}% | EER: {metrics['eer']:.4f}"
    )


if __name__ == "__main__":
    main()
