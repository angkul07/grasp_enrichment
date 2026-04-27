"""
Stage 1.1 — Kinematic Gap Closure: 3D Hand Mesh Reconstruction

Uses HaMeR to reconstruct 3D MANO hand meshes from egocentric RGB video.
For each video frame:
  1. ViTPose detects whole-body keypoints (using full frame as person bbox, since
     the footage is egocentric and the person is always the camera wearer).
  2. Left/right hand bounding boxes are derived from confident hand keypoints.
  3. HaMeR recovers 3D MANO mesh vertices + camera for each detected hand.
  4. Per-frame results are saved to a parallel HDF5 file.

Reference: third_party/hamer/demo.py (adapted for egocentric video pipeline).
"""

import os
import sys
import argparse
import glob

import cv2
import h5py
import numpy as np
import torch
from loguru import logger
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve third-party paths
# ---------------------------------------------------------------------------
HAMER_ROOT = os.path.abspath('third_party/hamer')
sys.path.insert(0, HAMER_ROOT)

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full


# ---------------------------------------------------------------------------
# Hand Keypoint Extractor  (ViTPose — whole-body)
# ---------------------------------------------------------------------------
class HandKeypointDetector:
    """
    Wraps ViTPose to detect whole-body keypoints and extract per-hand bounding
    boxes.  For egocentric data we treat the full image as the person crop,
    avoiding the need for a separate body detector (Detectron2).
    """

    # Indices in the COCO-WholeBody 133-keypoint layout
    LEFT_HAND_SLICE  = slice(-42, -21)   # 21 keypoints
    RIGHT_HAND_SLICE = slice(-21, None)  # 21 keypoints
    MIN_CONFIDENT_KPS = 3
    KP_CONF_THRESH    = 0.5

    def __init__(self, device: torch.device):
        from vitpose_model import ViTPoseModel
        logger.info("Loading ViTPose whole-body keypoint model…")
        self.cpm = ViTPoseModel(device)

    def detect(self, img_rgb: np.ndarray):
        """
        Detect left / right hand bounding boxes from a single RGB frame.

        Args:
            img_rgb: (H, W, 3) uint8 RGB image.

        Returns:
            boxes:    (N, 4) float32 array of [x1, y1, x2, y2] hand bboxes.
            is_right: (N,)   float32 array — 1.0 for right hand, 0.0 for left.
                      Returns empty arrays when no confident hands are found.
        """
        h, w = img_rgb.shape[:2]

        # Treat the full frame as a single person detection.
        # Format expected by ViTPose: list of (N_persons, 5) arrays [x1,y1,x2,y2,score]
        full_frame_det = [np.array([[0, 0, w, h, 1.0]], dtype=np.float32)]

        vitposes_out = self.cpm.predict_pose(img_rgb, full_frame_det)

        bboxes = []
        is_right = []

        for vitposes in vitposes_out:
            keypoints = vitposes['keypoints']  # (133, 3) — x, y, conf
            for kps_slice, side_flag in [
                (self.LEFT_HAND_SLICE,  0),
                (self.RIGHT_HAND_SLICE, 1),
            ]:
                kps = keypoints[kps_slice]           # (21, 3)
                valid = kps[:, 2] > self.KP_CONF_THRESH
                if valid.sum() > self.MIN_CONFIDENT_KPS:
                    bbox = [
                        kps[valid, 0].min(),
                        kps[valid, 1].min(),
                        kps[valid, 0].max(),
                        kps[valid, 1].max(),
                    ]
                    bboxes.append(bbox)
                    is_right.append(side_flag)

        if len(bboxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        return np.array(bboxes, dtype=np.float32), np.array(is_right, dtype=np.float32)


# ---------------------------------------------------------------------------
# Stage 1 Pipeline
# ---------------------------------------------------------------------------
class Stage1Pipeline:
    """Full per-video pipeline: ViTPose hands → HaMeR mesh → HDF5."""

    HAMER_BATCH_SIZE     = 8
    RESCALE_FACTOR       = 2.0   # Bbox padding factor (same as demo.py default)

    def __init__(self, output_dir: str, checkpoint: str = DEFAULT_CHECKPOINT):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing Stage 1 Pipeline on device: {self.device}")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._load_models(checkpoint)

    # -------------------------------- init ---------------------------------
    def _load_models(self, checkpoint: str):
        # 1. HaMeR
        logger.info("Downloading / verifying HaMeR checkpoint data…")
        download_models(CACHE_DIR_HAMER)

        logger.info(f"Loading HaMeR model from {checkpoint}")
        self.model, self.model_cfg = load_hamer(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 2. Hand keypoint detector (ViTPose)
        self.hand_detector = HandKeypointDetector(self.device)

    # ------------------------------ per-frame ------------------------------
    def _run_hamer_on_frame(self, img_cv2: np.ndarray, boxes: np.ndarray, is_right: np.ndarray):
        """
        Run HaMeR inference on the detected hand crops of a single frame.

        Returns:
            verts_list:  list of (778, 3) float32 arrays  — MANO mesh vertices
            cam_t_list:  list of (3,)    float32 arrays  — full-image camera translation
            right_list:  list of float                    — 1.0 = right, 0.0 = left
        """
        dataset = ViTDetDataset(
            self.model_cfg, img_cv2, boxes, is_right,
            rescale_factor=self.RESCALE_FACTOR,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.HAMER_BATCH_SIZE, shuffle=False, num_workers=0,
        )

        verts_list = []
        cam_t_list = []
        right_list = []

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            # Camera: correct x-flip for left hands, then map crop cam → full image
            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            box_center = batch['box_center'].float()
            box_size   = batch['box_size'].float()
            img_size   = batch['img_size'].float()
            scaled_focal = (
                self.model_cfg.EXTRA.FOCAL_LENGTH
                / self.model_cfg.MODEL.IMAGE_SIZE
                * img_size.max()
            )
            cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal,
            ).detach().cpu().numpy()

            # Collect per-hand results
            bs = batch['img'].shape[0]
            for i in range(bs):
                verts = out['pred_vertices'][i].detach().cpu().numpy()       # (778, 3)
                is_r  = batch['right'][i].cpu().item()
                # Mirror x-axis for left hands so both are in a consistent frame
                verts[:, 0] = (2 * is_r - 1) * verts[:, 0]

                verts_list.append(verts)
                cam_t_list.append(cam_t_full[i])
                right_list.append(float(is_r))

        return verts_list, cam_t_list, right_list

    # ------------------------------ per-video ------------------------------
    def process_video(self, video_path: str):
        basename = os.path.basename(video_path)
        out_path = os.path.join(self.output_dir, basename.replace('.mp4', '_stage1.hdf5'))

        logger.info(f"Processing video: {video_path}")
        logger.info(f"Output HDF5: {out_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            logger.warning(f"Video {video_path} has 0 frames. Skipping.")
            cap.release()
            return

        # Accumulators — variable-length per frame; we will store them as
        # top-level groups keyed by frame index.
        all_verts    = []   # list[list[ndarray(778,3)]]
        all_cam_t    = []   # list[list[ndarray(3,)]]
        all_is_right = []   # list[list[float]]
        frames_with_hands = 0

        pbar = tqdm(total=frame_count, desc=f"[HaMeR] {basename}")
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Step 1: detect hand bounding boxes via ViTPose
            boxes, is_right = self.hand_detector.detect(img_rgb)

            if len(boxes) == 0:
                all_verts.append([])
                all_cam_t.append([])
                all_is_right.append([])
                logger.debug(f"Frame {frame_idx}: no hands detected")
            else:
                # Step 2: run HaMeR on detected hands
                verts, cam_t, right = self._run_hamer_on_frame(frame, boxes, is_right)
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_is_right.append(right)
                frames_with_hands += 1
                logger.debug(f"Frame {frame_idx}: {len(verts)} hand(s) recovered")

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        # ---------------------- write HDF5 ----------------------
        logger.info(f"Saving {frame_idx} frames to HDF5 ({frames_with_hands} with hands)…")
        with h5py.File(out_path, 'w') as hf:
            hf.attrs['source_video'] = basename
            hf.attrs['total_frames'] = frame_idx
            hf.attrs['frames_with_hands'] = frames_with_hands

            for fidx in range(frame_idx):
                grp = hf.create_group(f'frame_{fidx:06d}')
                n_hands = len(all_verts[fidx])
                grp.attrs['n_hands'] = n_hands
                if n_hands > 0:
                    grp.create_dataset(
                        'vertices', data=np.stack(all_verts[fidx]),
                        compression='gzip',
                    )  # (N_hands, 778, 3)
                    grp.create_dataset(
                        'cam_t', data=np.stack(all_cam_t[fidx]),
                        compression='gzip',
                    )  # (N_hands, 3)
                    grp.create_dataset(
                        'is_right', data=np.array(all_is_right[fidx]),
                    )  # (N_hands,)

        logger.success(f"Done → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Stage 1.1 — 3D Hand Mesh Reconstruction via HaMeR + ViTPose"
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Directory containing egocentric .mp4 files")
    parser.add_argument('--output_dir', type=str, default='output',
                        help="Directory for output HDF5 files")
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to HaMeR checkpoint (.ckpt)")
    parser.add_argument('--limit', type=int, default=2,
                        help="Max videos to process (0 = all)")
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    logger.add('logs/stage1.log', rotation='500 MB')
    logger.info("=== Starting Stage 1.1 Pipeline ===")

    pipeline = Stage1Pipeline(output_dir=args.output_dir, checkpoint=args.checkpoint)

    video_files = sorted(glob.glob(os.path.join(args.input_dir, '*.mp4')))
    logger.info(f"Found {len(video_files)} video(s) in {args.input_dir}")

    if args.limit > 0:
        video_files = video_files[:args.limit]
        logger.info(f"Limiting to {args.limit} video(s)")

    for vf in video_files:
        pipeline.process_video(vf)

    logger.info("=== Stage 1.1 Pipeline Complete ===")


if __name__ == '__main__':
    main()
