"""Step 6: Shopformer model wrapper, PoseLift dataset loader, and evaluation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import yaml

from src.preprocess import keypoints17_to_coco18, normalize_pose

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHOPFORMER_DIR = PROJECT_ROOT / "third_party" / "Shopformer"
if str(SHOPFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(SHOPFORMER_DIR))

from models_graph.gcae.gcae import Encoder_T, GCAE  # type: ignore[import-not-found] # noqa: E402

SHANGHAITECH_HR_SKIP = [(1, 130), (1, 135), (1, 136), (6, 144), (6, 145), (12, 152)]
TARGET_FNR = 0.1


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for token sequences."""

    def __init__(self, d_model: int, max_len: int = 1000, device: str = "cpu") -> None:
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0))
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        buf = self.get_buffer("encoding")
        return x + buf[:, : x.size(1)].to(x.device).detach()


class TransformerPredictor(nn.Module):
    """Transformer encoder-decoder reconstructing GCAE pose tokens."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        num_layers: int,
        max_len: int = 1000,
        device: str = "cpu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model, num_heads, d_feedforward, dropout=dropout, batch_first=True
        )
        self.decoder_layers = nn.TransformerDecoderLayer(
            d_model, num_heads, d_feedforward, dropout=dropout, batch_first=True
        )
        self.positional_encoding = PositionalEncoding(d_model, max_len, device)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, num_layers=num_layers)
        self.device = device

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_pe = self.positional_encoding(src)
        memory = self.encoder(src_pe)
        tgt_pe = self.positional_encoding(tgt)
        return self.decoder(tgt_pe, memory)


class ReformerPredictor(TransformerPredictor):
    """Compatibility alias for ReformerPredictor referenced in Shopformer."""


def calc_reg_loss(model: nn.Module) -> torch.Tensor:
    """Compute L2 regularization loss across trainable parameters."""
    reg_loss: torch.Tensor | None = None
    for param in model.parameters():
        if param.requires_grad:
            term = torch.sum(param**2)
            reg_loss = term if reg_loss is None else (reg_loss + term)
    if reg_loss is None:
        return torch.tensor(0.0)
    return reg_loss


def resolve_device(requested: str) -> str:
    """Resolve PyTorch device string, falling back to CPU if CUDA is unavailable."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def build_Models_from_config(
    cfg: dict[str, Any], device: str
) -> tuple[GCAE, TransformerPredictor]:
    """Instantiate GCAE and TransformerPredictor from YAML configuration dictionary."""
    model_cfg = cfg["model"]
    prep_cfg = cfg["preprocess"]
    gcae = GCAE(
        in_channels=int(model_cfg["in_channels"]),
        h_dim=int(model_cfg["h_dim"]),
        dropout=float(model_cfg["dropout"]),
        input_frames=int(prep_cfg["window_size"]),
        conv_oper=str(model_cfg.get("conv_oper", "sagc")),
        act=str(model_cfg.get("act", "relu")),
        headless=bool(model_cfg.get("headless", False)),
    ).to(device)

    input_dim = int(model_cfg["h_dim"]) * int(prep_cfg["num_keypoints_coco18"]) * 4
    transformer = TransformerPredictor(
        d_model=input_dim,
        num_heads=int(model_cfg["transformer_heads"]),
        d_feedforward=int(model_cfg["transformer_latent_dim"]),
        num_layers=int(model_cfg["transformer_layers"]),
        max_len=1000,
        device=device,
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    return gcae, transformer


def tokenize_batch(encoder: Encoder_T, data: torch.Tensor) -> torch.Tensor:
    """Encode pose batch (B, 2, T, 18) into token sequence (B, num_tokens, token_dim)."""
    with torch.no_grad():
        encoded = encoder(data)
        b_sz, c_out, t_tok, num_kp, m_sz = encoded.shape
        tokens = (
            encoded.permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(b_sz, t_tok, c_out * num_kp * m_sz)
        )
    return tokens


def _is_seg_continuous(
    sorted_seg_keys: list[int], start_key: int, seg_len: int, missing_th: int = 2
) -> bool:
    """Check whether a pose segment of length `seg_len` has at most `missing_th` missing frames."""
    start_idx = sorted_seg_keys.index(start_key)
    expected_idxs = list(range(start_key, start_key + seg_len))
    act_idxs = sorted_seg_keys[start_idx : start_idx + seg_len]
    min_overlap = seg_len - missing_th
    return len(set(act_idxs).intersection(expected_idxs)) >= min_overlap


def _single_pose_dict2np(
    person_dict: dict[str, Any], idx: str
) -> tuple[np.ndarray, list[int], list[str], np.ndarray]:
    """Convert a single tracked person's dictionary into numpy arrays with integer frame sorting."""
    single_person = person_dict[str(idx)]
    if isinstance(single_person, list):
        merged: dict[str, Any] = {}
        for sub_dict in single_person:
            merged.update(**sub_dict)
        single_person = merged

    # Integer frame sort fix (prevents lexicographical '10' < '7' ordering bug in original repo)
    single_person_dict_keys = sorted(single_person.keys(), key=lambda x: int(float(x)))
    sing_pose_meta = [int(float(idx)), int(float(single_person_dict_keys[0]))]
    sing_pose_np = []
    sing_scores_np = []
    for key in single_person_dict_keys:
        curr_pose_np = np.array(single_person[key]["keypoints"], dtype=np.float32).reshape(-1, 3)
        sing_pose_np.append(curr_pose_np)
        sing_scores_np.append(single_person[key].get("scores"))
    return (
        np.stack(sing_pose_np, axis=0),
        sing_pose_meta,
        single_person_dict_keys,
        np.stack(sing_scores_np, axis=0),
    )


def _split_pose_to_segments(
    single_pose_np: np.ndarray,
    single_pose_meta: list[int],
    single_pose_keys: list[str],
    start_ofst: int = 0,
    seg_dist: int = 6,
    seg_len: int = 12,
    scene_id: str | int = "",
    clip_id: str | int = "",
    single_score_np: np.ndarray | None = None,
    dataset: str = "Poselift",
) -> tuple[np.ndarray, list[list[Any]], np.ndarray]:
    clip_t, kp_count, kp_dim = single_pose_np.shape
    pose_segs_np = np.empty((0, seg_len, kp_count, kp_dim), dtype=np.float32)
    pose_score_np = np.empty((0, seg_len), dtype=np.float32)
    pose_segs_meta: list[list[Any]] = []
    num_segs = int(np.ceil((clip_t - seg_len) / seg_dist)) if clip_t > seg_len else 0
    single_pose_keys_sorted = sorted([int(float(i)) for i in single_pose_keys])

    for seg_ind in range(num_segs):
        start_ind = start_ofst + seg_ind * seg_dist
        start_key = single_pose_keys_sorted[start_ind]
        if _is_seg_continuous(single_pose_keys_sorted, start_key, seg_len):
            curr_segment = single_pose_np[start_ind : start_ind + seg_len].reshape(
                1, seg_len, kp_count, kp_dim
            )
            curr_score = (
                single_score_np[start_ind : start_ind + seg_len].reshape(1, seg_len)
                if single_score_np is not None
                else np.zeros((1, seg_len), dtype=np.float32)
            )
            pose_segs_np = np.append(pose_segs_np, curr_segment, axis=0)
            pose_score_np = np.append(pose_score_np, curr_score, axis=0)
            if dataset == "UBnormal":
                pose_segs_meta.append(
                    [int(scene_id), clip_id, int(single_pose_meta[0]), int(start_key)]
                )
            else:
                pose_segs_meta.append(
                    [int(scene_id), int(clip_id), int(single_pose_meta[0]), int(start_key)]
                )
    return pose_segs_np, pose_segs_meta, pose_score_np


def gen_clip_seg_data_np(
    clip_dict: dict[str, Any],
    start_ofst: int = 0,
    seg_stride: int = 4,
    seg_len: int = 12,
    scene_id: str | int = "",
    clip_id: str | int = "",
    ret_keys: bool = False,
    global_pose_data: list[np.ndarray] | None = None,
    dataset: str = "Poselift",
) -> tuple[Any, ...]:
    """Slice a clip dictionary of tracked persons into fixed-length pose windows."""
    if global_pose_data is None:
        global_pose_data = []
    pose_segs_data = []
    score_segs_data = []
    pose_segs_meta: list[list[Any]] = []
    person_keys: dict[str, list[str]] = {}

    for idx in sorted(clip_dict.keys(), key=lambda x: int(float(x))):
        sing_pose_np, sing_pose_meta, sing_pose_keys, sing_scores_np = _single_pose_dict2np(
            clip_dict, idx
        )
        if dataset == "UBnormal":
            key = f"{int(scene_id):02d}_{clip_id}_{int(idx):02d}"
        else:
            key = f"{int(scene_id):02d}_{int(clip_id):04d}_{int(float(idx)):02d}"
        person_keys[key] = sing_pose_keys
        curr_pose_segs_np, curr_pose_segs_meta, curr_pose_score_np = _split_pose_to_segments(
            sing_pose_np,
            sing_pose_meta,
            sing_pose_keys,
            start_ofst,
            seg_stride,
            seg_len,
            scene_id=scene_id,
            clip_id=clip_id,
            single_score_np=sing_scores_np,
            dataset=dataset,
        )
        if dataset in ("NWPUC", "Poselift"):
            curr_pose_score_np = np.zeros(curr_pose_score_np.shape, dtype=np.float32)
        pose_segs_data.append(curr_pose_segs_np)
        score_segs_data.append(curr_pose_score_np)
        if sing_pose_np.shape[0] > seg_len:
            global_pose_data.append(sing_pose_np)
        pose_segs_meta += curr_pose_segs_meta

    if len(pose_segs_data) == 0:
        pose_segs_data_np = np.empty((0, seg_len, 17, 3), dtype=np.float32)
        score_segs_data_np = np.empty((0, seg_len), dtype=np.float32)
    else:
        pose_segs_data_np = np.concatenate(pose_segs_data, axis=0)
        score_segs_data_np = np.concatenate(score_segs_data, axis=0)

    global_pose_data_np = (
        np.concatenate(global_pose_data, axis=0)
        if len(global_pose_data) > 0
        else np.empty((0, 17, 3), dtype=np.float32)
    )
    if ret_keys:
        return (
            pose_segs_data_np,
            pose_segs_meta,
            person_keys,
            global_pose_data_np,
            global_pose_data,
            score_segs_data_np,
        )
    return (
        pose_segs_data_np,
        pose_segs_meta,
        global_pose_data_np,
        global_pose_data,
        score_segs_data_np,
    )


class PoseSegDataset(Dataset):
    """Generates sliced pose windows and metadata for PoseLift training and evaluation."""

    def __init__(
        self,
        path_to_json_dir: str | Path,
        path_to_vid_dir: str | Path | None = None,
        normalize_pose_segs: bool = True,
        return_indices: bool = False,
        return_metadata: bool = False,
        debug: bool = False,
        return_global: bool = True,
        evaluate: bool = False,
        abnormal_train_path: str | Path | None = None,
        **dataset_args: Any,
    ) -> None:
        super().__init__()
        self.args = dataset_args
        self.path_to_json = str(path_to_json_dir)
        self.normalize_pose_segs = normalize_pose_segs
        self.headless = bool(dataset_args.get("headless", False))
        self.path_to_vid_dir = path_to_vid_dir
        self.eval = evaluate
        self.debug = debug
        num_clips = dataset_args.get("specific_clip", None)
        self.return_indices = return_indices
        self.return_metadata = return_metadata
        self.return_global = return_global
        self.transform_list = dataset_args.get("trans_list", None)
        if self.transform_list is None:
            self.apply_transforms = False
            self.num_transform = 1
        else:
            self.apply_transforms = True
            self.num_transform = len(self.transform_list)
        self.train_seg_conf_th = float(dataset_args.get("train_seg_conf_th", 0.0))
        self.seg_len = int(dataset_args.get("seg_len", 12))
        self.seg_stride = int(dataset_args.get("seg_stride", 1))
        (
            self.segs_data_np,
            self.segs_meta,
            self.person_keys,
            self.global_data_np,
            self.global_data,
            self.segs_score_np,
        ) = gen_dataset(
            self.path_to_json,
            num_clips=num_clips,
            ret_keys=True,
            ret_global_data=return_global,
            **dataset_args,
        )
        self.segs_meta = np.array(self.segs_meta)
        self.labels = np.ones(self.segs_data_np.shape[0])
        self.person_keys = {k: [int(float(i)) for i in v] for k, v in self.person_keys.items()}
        self.metadata = self.segs_meta
        self.num_samples, self.C, self.T, self.V = self.segs_data_np.shape

    def __getitem__(self, index: int) -> list[Any]:
        if self.apply_transforms:
            sample_index = index % self.num_samples
            trans_index = math.floor(index / self.num_samples)
            data_numpy = np.array(self.segs_data_np[sample_index])
            data_transformed = self.transform_list[trans_index](data_numpy)
        else:
            sample_index = index
            data_transformed = np.array(self.segs_data_np[index])
            trans_index = 0

        if self.normalize_pose_segs:
            data_transformed = (
                normalize_pose(data_transformed.transpose((1, 2, 0))[None, ...], **self.args)
                .squeeze(axis=0)
                .transpose(2, 0, 1)
            )

        ret_arr: list[Any] = [
            data_transformed,
            trans_index,
            self.segs_score_np[sample_index],
            self.labels[sample_index],
        ]
        return ret_arr

    def __len__(self) -> int:
        return self.num_transform * self.num_samples


def gen_dataset(
    person_json_root: str | Path,
    num_clips: int | None = None,
    kp18_format: bool = True,
    ret_keys: bool = False,
    ret_global_data: bool = True,
    **dataset_args: Any,
) -> tuple[Any, ...]:
    """Load PoseLift JSON files from `person_json_root` and slice into (N, 3, T, 18) windows."""
    segs_data_np = []
    segs_score_np = []
    segs_meta: list[list[Any]] = []
    global_data: list[np.ndarray] = []
    person_keys: dict[str, list[str]] = {}
    start_ofst = int(dataset_args.get("start_ofst", 0))
    seg_stride = int(dataset_args.get("seg_stride", 1))
    seg_len = int(dataset_args.get("seg_len", 12))
    headless = bool(dataset_args.get("headless", False))
    seg_conf_th = float(dataset_args.get("train_seg_conf_th", 0.0))
    dataset = str(dataset_args.get("dataset", "Poselift"))
    swap_yx = bool(dataset_args.get("swap_yx", False))

    root_str = str(person_json_root)
    json_list = sorted([fn for fn in os.listdir(root_str) if fn.endswith(".json")])
    if num_clips is not None:
        json_list = [json_list[num_clips]]
    split_name = os.path.basename(root_str.rstrip("/\\"))

    for person_dict_fn in tqdm(json_list, desc=f"Loading {split_name}"):
        if dataset == "UBnormal":
            type_, scene_id, clip_id = re.findall(
                "(abnormal|normal)_scene_(\\d+)_scenario(.*)_alphapose_.*", person_dict_fn
            )[0]
            clip_id = type_ + "_" + clip_id
        else:
            scene_id, clip_id = person_dict_fn.split("_")[:2]
            if dataset == "ShanghaiTech-HR" and (int(scene_id), int(clip_id)) in SHANGHAITECH_HR_SKIP:
                continue
        clip_json_path = os.path.join(root_str, person_dict_fn)
        with open(clip_json_path, "r", encoding="utf-8") as f:
            clip_dict = json.load(f)
        (
            clip_segs_data_np,
            clip_segs_meta,
            clip_keys,
            _,
            _,
            score_segs_data_np,
        ) = gen_clip_seg_data_np(
            clip_dict,
            start_ofst,
            seg_stride,
            seg_len,
            scene_id=scene_id,
            clip_id=clip_id,
            ret_keys=ret_keys,
            dataset="Poselift" if dataset.lower() == "poselift" else dataset,
        )
        _, _, _, global_data_np, global_data, _ = gen_clip_seg_data_np(
            clip_dict,
            start_ofst,
            1,
            1,
            scene_id=scene_id,
            clip_id=clip_id,
            ret_keys=ret_keys,
            global_pose_data=global_data,
            dataset="Poselift" if dataset.lower() == "poselift" else dataset,
        )
        segs_data_np.append(clip_segs_data_np)
        segs_score_np.append(score_segs_data_np)
        segs_meta += clip_segs_meta
        person_keys = {**person_keys, **clip_keys}

    global_data_np = (
        np.expand_dims(np.concatenate(global_data, axis=0), axis=1)
        if len(global_data) > 0
        else np.empty((0, 1, 17, 3), dtype=np.float32)
    )
    segs_data_np = np.concatenate(segs_data_np, axis=0)
    segs_score_np = np.concatenate(segs_score_np, axis=0)

    if swap_yx:
        segs_data_np = segs_data_np[..., [1, 0, 2]]
        global_data_np = global_data_np[..., [1, 0, 2]]
        global_data = [data[..., [1, 0, 2]] for data in global_data]

    if kp18_format and segs_data_np.shape[-2] == 17:
        segs_data_np = keypoints17_to_coco18(segs_data_np)
        if global_data_np.shape[-2] == 17:
            global_data_np = keypoints17_to_coco18(global_data_np)
        global_data = [keypoints17_to_coco18(data) for data in global_data]
    if headless:
        segs_data_np = segs_data_np[:, :, 5:]
        global_data_np = global_data_np[:, :, 5:]
        global_data = [data[:, 5:, :] for data in global_data]

    segs_data_np = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)
    global_data_np = np.transpose(global_data_np, (0, 3, 1, 2)).astype(np.float32)

    if seg_conf_th > 0.0:
        sum_confs = segs_score_np.mean(axis=1)
        segs_data_np = segs_data_np[sum_confs > seg_conf_th]
        segs_meta = list(np.array(segs_meta)[sum_confs > seg_conf_th])
        segs_score_np = segs_score_np[sum_confs > seg_conf_th]

    if ret_global_data:
        if ret_keys:
            return segs_data_np, segs_meta, person_keys, global_data_np, global_data, segs_score_np
        return segs_data_np, segs_meta, global_data_np, global_data, segs_score_np
    if ret_keys:
        return segs_data_np, segs_meta, person_keys, segs_score_np
    return segs_data_np, segs_meta, segs_score_np


def score_auc(
    scores_np: np.ndarray, gt: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """Compute AUC-ROC, AUC-PR, EER, and EER threshold safely."""
    scores_clean = np.array(scores_np, dtype=np.float64, copy=True)
    finite_pos = scores_clean[scores_clean != np.inf]
    finite_neg = scores_clean[scores_clean != -np.inf]
    scores_clean[scores_clean == np.inf] = finite_pos.max() if finite_pos.size > 0 else 1.0
    scores_clean[scores_clean == -np.inf] = finite_neg.min() if finite_neg.size > 0 else 0.0

    auc_roc = float(roc_auc_score(gt, scores_clean))
    precision, recall, _ = precision_recall_curve(gt, scores_clean)
    auc_precision_recall = float(auc(recall, precision))

    fpr, tpr, threshold = roc_curve(gt, scores_clean, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.absolute(fnr - fpr)))
    eer_threshold = float(threshold[eer_idx])
    eer = float(fpr[eer_idx])

    idx_closest_fnr = int(np.argmin(np.abs(fnr - TARGET_FNR)))
    threshold_at_target_fnr = float(threshold[idx_closest_fnr])
    fpr_at_target_fnr = float(fpr[idx_closest_fnr])
    return (
        auc_roc,
        auc_precision_recall,
        eer,
        eer_threshold,
        fpr_at_target_fnr,
        threshold_at_target_fnr,
    )


def _smooth_scores(scores_arr: list[np.ndarray], sigma: int = 7) -> list[np.ndarray]:
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr


def _get_clip_score(
    scores: np.ndarray,
    clip: str,
    metadata_np: np.ndarray,
    metadata: np.ndarray,
    per_frame_scores_root: str,
    args: Any,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    scene_id, clip_id = [int(i) for i in clip.split(".")[0].split("_")[:2]]
    if getattr(args, "dataset", "Poselift") == "ShanghaiTech-HR" and (
        scene_id,
        clip_id,
    ) in SHANGHAITECH_HR_SKIP:
        return None, None

    clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) & (metadata_np[:, 0] == scene_id))[
        0
    ]
    clip_metadata = metadata[clip_metadata_inds]
    clip_fig_idxs = set([arr[2] for arr in clip_metadata])
    clip_res_fn = os.path.join(per_frame_scores_root, clip)
    clip_gt = np.load(clip_res_fn)

    scores_zeros = np.ones(clip_gt.shape[0], dtype=np.float64) * (-np.inf)
    if len(clip_fig_idxs) == 0:
        clip_person_scores_dict = {0: np.copy(scores_zeros)}
    else:
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}

    for person_id in clip_fig_idxs:
        person_metadata_inds = np.where(
            (metadata_np[:, 1] == clip_id)
            & (metadata_np[:, 0] == scene_id)
            & (metadata_np[:, 2] == person_id)
        )[0]
        pid_scores = scores[person_metadata_inds]
        pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds]).astype(int)
        # Bounds clip fix (prevents IndexError on final frames of clip)
        target_inds = np.clip(pid_frame_inds + int(args.seg_len / 2), 0, clip_gt.shape[0] - 1)
        clip_person_scores_dict[person_id][target_inds] = pid_scores

    clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
    clip_score = np.amax(clip_ppl_score_arr, axis=0)
    return clip_gt, clip_score


def score_dataset(
    score: np.ndarray, metadata: np.ndarray, args: Any = None
) -> tuple[float, float, float, float, float, float]:
    """Aggregate window-level scores to frame-level clip scores and compute PoseLift metrics."""
    dataset_gt_arr: list[np.ndarray] = []
    dataset_scores_arr: list[np.ndarray] = []
    metadata_np = np.array(metadata)

    clip_list = sorted(fn for fn in os.listdir(args.mask_root) if fn.endswith(".npy"))
    print(f"Scoring {len(clip_list)} clips")
    for clip in tqdm(clip_list):
        clip_gt, clip_score = _get_clip_score(
            score, clip, metadata_np, metadata, args.mask_root, args
        )
        if clip_score is not None and clip_gt is not None:
            dataset_gt_arr.append(clip_gt)
            dataset_scores_arr.append(clip_score)
            if getattr(args, "save_results", False):
                os.makedirs(args.save_results_dir, exist_ok=True)
                out_csv = os.path.join(args.save_results_dir, clip.split(".")[0] + ".csv")
                with open(out_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(clip_gt, clip_score))

    scores_np = np.concatenate(dataset_scores_arr, axis=0)
    finite_pos = scores_np[scores_np != np.inf]
    finite_neg = scores_np[scores_np != -np.inf]
    scores_np[scores_np == np.inf] = finite_pos.max() if finite_pos.size > 0 else 1.0
    scores_np[scores_np == -np.inf] = finite_neg.min() if finite_neg.size > 0 else 0.0

    index = 0
    for sc_idx in range(len(dataset_scores_arr)):
        for t in range(dataset_scores_arr[sc_idx].shape[0]):
            dataset_scores_arr[sc_idx][t] = scores_np[index]
            index += 1

    dataset_scores_arr = _smooth_scores(dataset_scores_arr)
    gt_np = np.concatenate(dataset_gt_arr)
    smoothed_np = np.concatenate(dataset_scores_arr)
    return score_auc(smoothed_np, gt_np)


class ShopformerScorer:
    """Wrap frozen GCAE tokenizer and Transformer encoder-decoder to score 12-frame pose windows."""

    def __init__(
        self,
        gcae: GCAE,
        transformer: TransformerPredictor,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.gcae = gcae.to(device)
        self.gcae.eval()
        self.encoder = Encoder_T(self.gcae).to(device)
        self.encoder.eval()
        self.transformer = transformer.to(device)
        self.transformer.eval()

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        project_root: Path | None = None,
        device_override: str | None = None,
    ) -> ShopformerScorer:
        """Load ShopformerScorer and pretrained checkpoints specified in YAML config."""
        root = project_root or PROJECT_ROOT
        with open(config_path, "r", encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f)

        device = resolve_device(device_override or str(cfg["pose_track"]["device"]))
        gcae, transformer = build_Models_from_config(cfg, device)

        gcae_ckpt = root / cfg["paths"]["gcae_ckpt"]
        shopformer_ckpt = root / cfg["paths"]["shopformer_ckpt"]

        if gcae_ckpt.exists():
            state_g = torch.load(gcae_ckpt, map_location=device, weights_only=False)
            gcae.load_state_dict(state_g["state_dict"])
        if shopformer_ckpt.exists():
            state_t = torch.load(shopformer_ckpt, map_location=device, weights_only=False)
            transformer.load_state_dict(state_t["state_dict"])

        return cls(gcae=gcae, transformer=transformer, device=device)

    def score_batch(self, pose_batch: np.ndarray | torch.Tensor) -> np.ndarray:
        """Compute anomaly score (MSE reconstruction error) for a batch of windows (B, 2, 12, 18)."""
        if isinstance(pose_batch, np.ndarray):
            tensor_in = torch.from_numpy(pose_batch).to(device=self.device, dtype=torch.float32)
        else:
            tensor_in = pose_batch.to(device=self.device, dtype=torch.float32)

        if tensor_in.ndim == 3:
            tensor_in = tensor_in.unsqueeze(0)
        tensor_in = tensor_in[:, 0:2, :, :]

        with torch.no_grad():
            tokens = tokenize_batch(self.encoder, tensor_in)
            recon = self.transformer(tokens, tokens)
            mse = torch.mean((tokens - recon) ** 2, dim=(1, 2))
        return mse.detach().cpu().numpy().astype(np.float64)

    def score_window(self, pose_window: np.ndarray | torch.Tensor) -> float:
        """Compute scalar MSE anomaly score for a single normalized window of shape (2, 12, 18)."""
        scores = self.score_batch(pose_window)
        return float(scores[0])


def main() -> None:
    """CLI smoke test for ShopformerScorer."""
    parser = argparse.ArgumentParser(
        description="Smoke test ShopformerScorer on a 12-frame window."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"), help="Path to YAML config."
    )
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    scorer = ShopformerScorer.from_config(cfg_path, project_root=PROJECT_ROOT)
    dummy_window = np.random.randn(2, 12, 18).astype(np.float32)
    score = scorer.score_window(dummy_window)
    print(f"[ShopformerScorer] Device: {scorer.device} | Single window MSE score: {score:.6f}")


if __name__ == "__main__":
    main()
