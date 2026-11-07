"""Step 5: Pose preprocessing (linear interpolation, 8-frame smoothing, Shopformer normalization)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import uniform_filter1d
import torch

from shoplift_iot.config import AppConfig, load_app_config


def get_aff_trans_mat(
    sx: float = 1.0,
    sy: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
    rot: float = 0.0,
    shearx: float = 0.0,
    sheary: float = 0.0,
    flip: bool = False,
) -> torch.Tensor:
    """Generate affine transformation matrix for augmenting pose sequences."""
    cos_r = math.cos(math.radians(rot))
    sin_r = math.sin(math.radians(rot))
    flip_mat = torch.eye(3, dtype=torch.float32)
    if flip:
        flip_mat[0, 0] = -1.0
    trans_scale_mat = torch.tensor([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=torch.float32)
    shear_mat = torch.tensor([[1, shearx, 0], [sheary, 1, 0], [0, 0, 1]], dtype=torch.float32)
    rot_mat = torch.tensor(
        [[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]], dtype=torch.float32
    )
    aff_mat = torch.matmul(rot_mat, trans_scale_mat)
    aff_mat = torch.matmul(shear_mat, aff_mat)
    aff_mat = torch.matmul(flip_mat, aff_mat)
    return aff_mat


def apply_pose_transform(pose: np.ndarray, trans_mat: torch.Tensor | np.ndarray) -> np.ndarray:
    """Apply affine transformation matrix to a pose sequence of shape (C, T, V) or (C, T, V, M)."""
    conf = np.expand_dims(pose[2], axis=0)
    ones_vec = np.ones_like(conf)
    pose_w_ones = np.concatenate([pose[:2], ones_vec], axis=0)
    einsum_str = "ktv,ck->ctv" if len(pose.shape) == 3 else "ktvm,ck->ctvm"
    mat_np = trans_mat.numpy() if isinstance(trans_mat, torch.Tensor) else np.asarray(trans_mat)
    pose_transformed_wo_conf = np.einsum(einsum_str, pose_w_ones, mat_np)
    return np.concatenate([pose_transformed_wo_conf[:2], conf], axis=0)


class PoseTransform:
    """Callable affine transformation for pose sequences."""

    def __init__(
        self,
        sx: float = 1.0,
        sy: float = 1.0,
        tx: float = 0.0,
        ty: float = 0.0,
        rot: float = 0.0,
        shearx: float = 0.0,
        sheary: float = 0.0,
        flip: bool = False,
        trans_mat: torch.Tensor | None = None,
    ) -> None:
        self.trans_mat = (
            trans_mat
            if trans_mat is not None
            else get_aff_trans_mat(sx, sy, tx, ty, rot, shearx, sheary, flip)
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return apply_pose_transform(x, self.trans_mat)


trans_list = [
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=False),
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=True),
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=False, shearx=0.1, sheary=0.1),
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=True, shearx=0.1, sheary=0.1),
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=False, shearx=0, sheary=0.1),
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=True, shearx=0, sheary=0.1),
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=False, shearx=0.1, sheary=0),
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=True, shearx=0.1, sheary=0),
]


def keypoints17_to_coco18(kps: np.ndarray) -> np.ndarray:
    """Convert a 17-keypoint COCO skeleton to an 18-keypoint OpenPose COCO18 skeleton.

    The neck keypoint is computed as the midpoint of left_shoulder (5) and right_shoulder (6),
    and keypoints are reordered to OpenPose 18 order.
    """
    kp_np = np.array(kps, dtype=np.float32)
    neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
    kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
    opp_order = np.array(
        [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3], dtype=np.int_
    )
    return kp_np[..., opp_order, :]


def normalize_pose(pose_data: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Normalize keypoint coordinates following Shopformer's exact formula with zero-std guard.

    Args:
        pose_data: Array of shape (N, T, V, 3) in [X, Y, C].
        **kwargs: Supports `vid_res` ([W, H] or "WxH"), `symm_range` (bool), `traj` (bool).

    Returns:
        Normalized array of shape (N, T, V, 3).
    """
    vid_res = kwargs.get("vid_res", [1920, 1080])
    if isinstance(vid_res, str):
        w_str, h_str = vid_res.split("x")
        res_list = [int(w_str), int(h_str)]
    else:
        res_list = [int(vid_res[0]), int(vid_res[1])]

    symm_range = bool(kwargs.get("symm_range", False))
    pose_data_clean = np.nan_to_num(pose_data.astype(np.float32, copy=True))
    norm_factor = np.array(res_list + [1], dtype=np.float32)
    pose_data_centered = pose_data_clean / norm_factor
    pose_for_traj = np.copy(pose_data_centered)

    if symm_range:
        pose_data_centered[..., :2] = 2.0 * pose_data_centered[..., :2] - 1.0

    pose_data_zero_mean = pose_data_centered
    mean_xy = pose_data_centered[..., :2].mean(axis=(1, 2))[:, None, None, :]
    std_y = pose_data_centered[..., 1].std(axis=(1, 2))[:, None, None, None]
    std_y = np.where(std_y < 1e-6, 1.0, std_y)
    pose_data_zero_mean[..., :2] = (pose_data_centered[..., :2] - mean_xy) / std_y

    if kwargs.get("traj", False):
        c_hip = (pose_for_traj[:, :, 8, :] + pose_for_traj[:, :, 11, :]) / 2.0
        c_hip = c_hip[:, :, np.newaxis, :]
        pose_data_zero_mean = np.concatenate((pose_data_zero_mean, c_hip), axis=2)

    return pose_data_zero_mean


def interpolate_missing_frames(
    frame_to_kps: dict[int, np.ndarray],
    start_frame: int | None = None,
    end_frame: int | None = None,
    max_missing_frames: int = 2,
) -> np.ndarray | None:
    """Linearly interpolate missing frames and zero-confidence joints across time.

    Args:
        frame_to_kps: Mapping of frame_idx -> (17, 3) float32 array in [X, Y, C].
        start_frame: Desired start frame index (defaults to min key in frame_to_kps).
        end_frame: Desired end frame index (defaults to max key in frame_to_kps).
        max_missing_frames: Maximum missing frames allowed when fixed range is given.

    Returns:
        Dense numpy array of shape (T, 17, 3) float32, or None if too many frames are missing.
    """
    if not frame_to_kps:
        return None

    sorted_keys = sorted(frame_to_kps.keys())
    s_frame = sorted_keys[0] if start_frame is None else int(start_frame)
    e_frame = sorted_keys[-1] if end_frame is None else int(end_frame)
    if e_frame < s_frame:
        return None

    total_len = e_frame - s_frame + 1
    present_in_span = [k for k in sorted_keys if s_frame <= k <= e_frame]
    if len(present_in_span) == 0:
        return None
    if start_frame is not None and end_frame is not None:
        if total_len - len(present_in_span) > max_missing_frames:
            return None

    sample_kp = frame_to_kps[present_in_span[0]]
    num_kp, channels = sample_kp.shape
    dense = np.zeros((total_len, num_kp, channels), dtype=np.float32)

    obs_indices = np.array([k - s_frame for k in present_in_span], dtype=np.float64)
    obs_values = np.stack([frame_to_kps[k] for k in present_in_span], axis=0).astype(np.float64)
    target_indices = np.arange(total_len, dtype=np.float64)

    for kp_idx in range(num_kp):
        kp_conf = obs_values[:, kp_idx, 2]
        valid_mask = kp_conf > 0.0
        if np.any(valid_mask) and not np.all(valid_mask):
            valid_t = obs_indices[valid_mask]
            for c_idx in range(channels):
                dense[:, kp_idx, c_idx] = np.interp(
                    target_indices, valid_t, obs_values[valid_mask, kp_idx, c_idx]
                ).astype(np.float32)
        else:
            for c_idx in range(channels):
                dense[:, kp_idx, c_idx] = np.interp(
                    target_indices, obs_indices, obs_values[:, kp_idx, c_idx]
                ).astype(np.float32)

    return dense


def smooth_pose_sequence(pose_seq: np.ndarray, window_size: int = 8) -> np.ndarray:
    """Smooth keypoint (X, Y) coordinates temporally using an 8-frame moving window."""
    if pose_seq.shape[0] <= 1 or window_size <= 1:
        return pose_seq.astype(np.float32, copy=True)

    out = pose_seq.astype(np.float32, copy=True)
    eff_win = min(int(window_size), out.shape[0])
    out[..., :2] = uniform_filter1d(out[..., :2], size=eff_win, axis=0, mode="nearest")
    return out


def coco17_to_openpose18(pose_seq: np.ndarray) -> np.ndarray:
    """Convert (..., 17, 3) COCO17 keypoints to (..., 18, 3) OpenPose COCO18 layout."""
    return keypoints17_to_coco18(pose_seq).astype(np.float32)


def normalize_shopformer_pose(
    pose_seq_18: np.ndarray,
    vid_res: tuple[int, int] | list[int] = (1920, 1080),
    symm_range: bool = False,
) -> np.ndarray:
    """Normalize (T, 18, 3) or (N, T, 18, 3) pose sequence using exact Shopformer formula."""
    single_input = pose_seq_18.ndim == 3
    batch = pose_seq_18[None, ...] if single_input else pose_seq_18
    normed = normalize_pose(
        batch.astype(np.float32, copy=True),
        vid_res=list(vid_res),
        symm_range=symm_range,
    )
    normed_ctv = np.transpose(normed, (0, 3, 1, 2))[:, 0:2, :, :].astype(np.float32)
    if single_input:
        return normed_ctv[0]
    return normed_ctv


class PosePreprocessor:
    """End-to-end Step 5 preprocessor."""

    def __init__(
        self,
        vid_res: tuple[int, int] = (1920, 1080),
        window_size: int = 12,
        smoothing_window: int = 8,
        max_missing_frames: int = 2,
        symm_range: bool = False,
    ) -> None:
        self.vid_res = (int(vid_res[0]), int(vid_res[1]))
        self.window_size = int(window_size)
        self.smoothing_window = int(smoothing_window)
        self.max_missing_frames = int(max_missing_frames)
        self.symm_range = bool(symm_range)

    def process_window(
        self,
        frame_to_kps: dict[int, np.ndarray],
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> np.ndarray | None:
        """Interpolate, smooth (window=8), convert 17->18, and normalize to (2, 12, 18)."""
        dense_17 = interpolate_missing_frames(
            frame_to_kps,
            start_frame=start_frame,
            end_frame=end_frame,
            max_missing_frames=self.max_missing_frames,
        )
        if dense_17 is None or dense_17.shape[0] != self.window_size:
            return None

        smoothed_17 = smooth_pose_sequence(dense_17, window_size=self.smoothing_window)
        pose_18 = coco17_to_openpose18(smoothed_17)
        normed = normalize_shopformer_pose(
            pose_18, vid_res=self.vid_res, symm_range=self.symm_range
        )
        return normed


    @classmethod
    def from_bundle_dict(cls, prep_dict: Mapping[str, Any]) -> PosePreprocessor:
        """Create PosePreprocessor directly from a ModelBundle's `preprocess.json` dict."""
        vid_res_raw = prep_dict.get("vid_res", [1920, 1080])
        return cls(
            vid_res=(int(vid_res_raw[0]), int(vid_res_raw[1])),
            window_size=int(prep_dict.get("window_size", 12)),
            smoothing_window=int(prep_dict.get("smoothing_window", 8)),
            max_missing_frames=int(prep_dict.get("max_missing_frames", 2)),
            symm_range=bool(prep_dict.get("symm_range", False)),
        )


def load_preprocessor_from_config(
    config_or_path: Path | str | AppConfig | Mapping[str, Any],
) -> PosePreprocessor:
    """Create PosePreprocessor from application configuration."""
    cfg = load_app_config(config_or_path)
    prep = cfg.preprocess
    vid = cfg.video
    return PosePreprocessor(
        vid_res=(int(vid["vid_res"][0]), int(vid["vid_res"][1])),
        window_size=int(prep["window_size"]),
        smoothing_window=int(prep["smoothing_window"]),
        max_missing_frames=int(prep["max_missing_frames"]),
        symm_range=bool(prep.get("symm_range", False)),
    )


def main() -> None:
    """CLI smoke test for PosePreprocessor."""
    parser = argparse.ArgumentParser(description="Smoke test PosePreprocessor.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"), help="Path to YAML config."
    )
    args = parser.parse_args()

    preprocessor = load_preprocessor_from_config(args.config)
    sample_dict: dict[int, np.ndarray] = {}
    for f_idx in range(12):
        if f_idx == 4:
            continue
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 0] = np.linspace(500.0, 600.0, 17) + f_idx * 2.0
        kps[:, 1] = np.linspace(300.0, 800.0, 17) + f_idx * 1.5
        kps[:, 2] = 0.95
        sample_dict[f_idx] = kps

    out = preprocessor.process_window(sample_dict, start_frame=0, end_frame=11)
    assert out is not None
    print(
        f"[PosePreprocessor] Output shape: {out.shape} | dtype: {out.dtype} | "
        f"min: {out.min():.3f}, max: {out.max():.3f}"
    )


if __name__ == "__main__":
    main()
