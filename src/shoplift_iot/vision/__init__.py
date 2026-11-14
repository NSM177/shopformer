"""Vision subpackage: YOLOv8-pose + ByteTrack, pose sliding window buffer, preprocessing, and person clips."""

from shoplift_iot.vision.buffer import PoseBuffer, TrackWindow, load_buffer_from_config
from shoplift_iot.vision.person_clips import (
    PersonClipBuffer,
    PersonClipExtractor,
    PersonClipWindow,
    build_videomae_clip_from_crops,
    crop_person_with_margin,
    expand_bbox_xyxy,
    save_person_clip_mp4,
)
from shoplift_iot.vision.pose_track import PersonPose, PoseTracker, load_tracker_from_config
from shoplift_iot.vision.preprocess import (
    PosePreprocessor,
    PoseTransform,
    apply_pose_transform,
    coco17_to_openpose18,
    get_aff_trans_mat,
    interpolate_missing_frames,
    keypoints17_to_coco18,
    load_preprocessor_from_config,
    normalize_pose,
    normalize_shopformer_pose,
    smooth_pose_sequence,
    trans_list,
)

__all__ = [
    "PersonClipBuffer",
    "PersonClipExtractor",
    "PersonClipWindow",
    "PersonPose",
    "PoseBuffer",
    "PosePreprocessor",
    "PoseTracker",
    "PoseTransform",
    "TrackWindow",
    "apply_pose_transform",
    "build_videomae_clip_from_crops",
    "coco17_to_openpose18",
    "crop_person_with_margin",
    "expand_bbox_xyxy",
    "get_aff_trans_mat",
    "interpolate_missing_frames",
    "keypoints17_to_coco18",
    "load_buffer_from_config",
    "load_preprocessor_from_config",
    "load_tracker_from_config",
    "normalize_pose",
    "normalize_shopformer_pose",
    "save_person_clip_mp4",
    "smooth_pose_sequence",
    "trans_list",
]
