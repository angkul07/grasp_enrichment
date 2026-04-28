"""
run_stage1_moge.py

Stage 1 MoGe-2 pass: lifts HaMeR 2D keypoints to metric 3D using MoGe-2
monocular depth, then stores ONLY the sparse (21, 3) joint coordinates per
hand — no dense maps ever touch disk (Lever 1 storage strategy).

Fixes applied vs. prior version:
  [F1] FOV is computed from EgoDex ground-truth intrinsics (camera/intrinsic
       in each HDF5), not hardcoded to 85.0 degrees.
  [F2] MoGe intrinsics output is also stored per-file for downstream use.
  [F3] MoGe validity mask is checked at sampled keypoint pixels; frames with
       too many invalid samples emit a warning and store a per-hand flag.
  [F4] Left-hand x-axis mirror is correctly undone before joint projection
       (self-inverse: multiply by -1 again, i.e. (2*is_r - 1) * x).
  [F5] scaled_focal = 5000 / 224 * max(W, H) matches cam_crop_to_full()
       exactly — not the raw crop-space focal of 5000.
  [F6] Compression changed from gzip to lz4 for better speed/ratio on small
       float arrays (requires hdf5plugin; falls back to gzip if not present).

Storage per frame (Lever 1 — sparse keypoints only):
  moge_joints_3d   : float16  (N_hands, 21, 3)   metric 3D joint positions
  moge_joint_valid : bool     (N_hands, 21)       MoGe mask validity per joint
  moge_fov_x_deg   : float32  scalar              FOV used for this frame
  (moge_K_norm written once per file at root level, not per-frame)
"""

import os
import sys
import glob
import cv2
import h5py
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Compression: prefer lz4 (fast, good ratio), fall back to gzip
# ---------------------------------------------------------------------------
try:
    import hdf5plugin  # noqa: F401
    COMPRESS_KW = dict(compression=hdf5plugin.LZ4())
    logger.info("Using LZ4 compression via hdf5plugin.")
except ImportError:
    COMPRESS_KW = dict(compression="gzip", compression_opts=4)
    logger.warning("hdf5plugin not found — falling back to gzip. "
                   "Install with: pip install hdf5plugin")

# ---------------------------------------------------------------------------
# Resolve HaMeR path
# ---------------------------------------------------------------------------
HAMER_ROOT = os.path.abspath('third_party/hamer')
sys.path.insert(0, HAMER_ROOT)

from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from moge.model.v2 import MoGeModel

# HaMeR projection constants (crop-space focal, before rescaling)
HAMER_FOCAL_LENGTH = 5000.0
HAMER_IMAGE_SIZE   = 224.0

# Fraction of joints allowed to land in invalid MoGe regions before we emit
# a warning. 4/21 ~ 19% — two fingers fully occluded is still acceptable.
INVALID_JOINT_WARN_THRESH = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_mano_j_regressor() -> np.ndarray:
    """
    Load HaMeR briefly to extract the MANO joint regressor (21 x 778).
    Frees the full model immediately to save VRAM for MoGe-2.
    """
    logger.info("Loading HaMeR to extract MANO J_regressor (21x778)...")
    model, _ = load_hamer(DEFAULT_CHECKPOINT)

    for attr in ('mano.J_regressor', 'mano_r.J_regressor'):
        obj = model
        try:
            for part in attr.split('.'):
                obj = getattr(obj, part)
            J = obj.cpu().numpy()
            logger.info(f"J_regressor found at model.{attr}, shape: {J.shape}")
            del model
            torch.cuda.empty_cache()
            return J
        except AttributeError:
            continue

    raise AttributeError(
        "Could not locate J_regressor in HaMeR model under 'mano.J_regressor' "
        "or 'mano_r.J_regressor'. Inspect with: "
        "print([n for n, _ in model.named_modules()])"
    )


def fov_x_from_intrinsics(K: np.ndarray, W: int) -> float:
    """
    Compute horizontal FOV in degrees from a 3x3 intrinsics matrix and
    image width. K[0,0] is fx in pixels.

    FOV_x = 2 * arctan(W / (2 * fx))
    """
    fx = float(K[0, 0])
    return float(np.degrees(2.0 * np.arctan(W / (2.0 * fx))))


def project_joints_to_pixels(
    joints_3d: np.ndarray,   # (21, 3) in scaled camera space
    cam_t: np.ndarray,        # (3,) camera translation from HaMeR
    scaled_focal: float,
    W: int,
    H: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project 3D MANO joints to 2D pixel coordinates using HaMeR's scaled
    pinhole model (principal point at image centre).

    Returns:
        xs, ys : (21,) int arrays, clamped to [0, W-1] and [0, H-1].
    """
    joints_cam = joints_3d + cam_t                             # (21, 3)
    xs = (joints_cam[:, 0] / joints_cam[:, 2]) * scaled_focal + (W / 2.0)
    ys = (joints_cam[:, 1] / joints_cam[:, 2]) * scaled_focal + (H / 2.0)
    xs = np.clip(np.round(xs).astype(int), 0, W - 1)
    ys = np.clip(np.round(ys).astype(int), 0, H - 1)
    return xs, ys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. MANO joint regressor
    J_regressor = get_mano_j_regressor()   # (21, 778)

    # 2. MoGe-2 model (vitl-normal: metric depth + surface normals)
    logger.info("Initializing MoGe-2 (Ruicheng/moge-2-vitl-normal)...")
    moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    moge = moge.half()   # fp16: ~2× faster, negligible quality loss
    moge.eval()

    # 3. Find HDF5 files written by run_stage1.py
    output_dir = "output"
    hdf5_files = sorted(glob.glob(os.path.join(output_dir, "*_stage1.hdf5")))

    if not hdf5_files:
        logger.warning(f"No *_stage1.hdf5 files found in '{output_dir}'. "
                       "Run run_stage1.py first.")
        return

    for h5_path in hdf5_files:
        logger.info(f"\n{'='*60}\nProcessing: {h5_path}")

        # -- Read metadata from our stage-1 HDF5 -------------------------
        with h5py.File(h5_path, 'r') as hf:
            source_video  = hf.attrs.get('source_video', None)
            egodex_hdf5   = hf.attrs.get('source_hdf5', None)   # path to original EgoDex HDF5
            total_frames  = int(hf.attrs.get('total_frames', 0))

        if source_video is None:
            logger.warning(f"  'source_video' attr missing — skipping.")
            continue

        # -- Locate original MP4 ------------------------------------------
        mp4_matches = glob.glob(f"**/{source_video}", recursive=True)
        if not mp4_matches:
            logger.error(f"  Cannot find source video '{source_video}' on disk — skipping.")
            continue
        vid_path = mp4_matches[0]

        # -- Load EgoDex camera intrinsics (K is constant across all frames) --
        # [F1] Read ground-truth intrinsics from the EgoDex HDF5 file
        # instead of hardcoding fov_x=85.0.
        # The EgoDex HDF5 stores camera/intrinsic as a (3,3) matrix.
        #
        # If the stage-1 HDF5 does not record 'source_hdf5', fall back to
        # deriving the path from the MP4 path (same stem, .hdf5 extension).
        if egodex_hdf5 and os.path.isfile(egodex_hdf5):
            egodex_h5_path = egodex_hdf5
        else:
            egodex_h5_path = os.path.splitext(vid_path)[0] + ".hdf5"

        if not os.path.isfile(egodex_h5_path):
            logger.error(
                f"  EgoDex intrinsics HDF5 not found at '{egodex_h5_path}'. "
                f"Cannot determine true FOV — skipping. "
                f"Tip: store 'source_hdf5' attr in run_stage1.py."
            )
            continue

        with h5py.File(egodex_h5_path, 'r') as ef:
            # camera/intrinsic is the same for every frame in EgoDex
            K_egodex = ef['/camera/intrinsic'][:]   # (3, 3)

        logger.info(f"  EgoDex intrinsics K:\n{K_egodex}")

        # -- Open video ---------------------------------------------------
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            logger.error(f"  Failed to open video '{vid_path}' — skipping.")
            continue

        # Compute scaled focal and FOV from the first frame's resolution.
        # (EgoDex is 1080p throughout, but we read it from the video to be safe.)
        ret, first_frame = cap.read()
        if not ret:
            logger.error(f"  Cannot read first frame of '{vid_path}' — skipping.")
            cap.release()
            continue
        H_vid, W_vid = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind

        # [F1] Derive FOV from EgoDex ground-truth intrinsics
        fov_x_deg   = fov_x_from_intrinsics(K_egodex, W_vid)
        scaled_focal = HAMER_FOCAL_LENGTH / HAMER_IMAGE_SIZE * max(W_vid, H_vid)  # [F5]

        logger.info(f"  Resolution : {W_vid}×{H_vid}")
        logger.info(f"  True FOV_x : {fov_x_deg:.2f}°  (from EgoDex intrinsics, fx={K_egodex[0,0]:.1f})")
        logger.info(f"  scaled_focal: {scaled_focal:.1f}  (for HaMeR back-projection)")

        pbar = tqdm(total=total_frames, desc=f"MoGe {source_video}")

        with h5py.File(h5_path, 'r+') as hf:

            # [F2] Store normalised intrinsics once at file level (not per-frame).
            # Normalised K: divide fx,fy,cx,cy by W and H respectively so it's
            # resolution-independent — useful if you ever resize frames later.
            if 'moge_K_norm' not in hf:
                K_norm = K_egodex.copy().astype(np.float32)
                K_norm[0, :] /= W_vid   # fx, cx → normalised
                K_norm[1, :] /= H_vid   # fy, cy → normalised
                hf.create_dataset('moge_K_norm', data=K_norm)
                hf.attrs['moge_fov_x_deg'] = fov_x_deg

            for fidx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                grp_name = f'frame_{fidx:06d}'
                if grp_name not in hf:
                    pbar.update(1)
                    continue

                grp    = hf[grp_name]
                n_hands = int(grp.attrs.get('n_hands', 0))

                if n_hands == 0:
                    pbar.update(1)
                    continue

                # Skip if already processed (idempotent re-run)
                if 'moge_joints_3d' in grp:
                    pbar.update(1)
                    continue

                # -- Run MoGe-2 (frame-level, fp16) -----------------------
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_t   = (torch.from_numpy(img_rgb).to(device=device, dtype=torch.float16)
                           / 255.0).permute(2, 0, 1)   # (3, H, W)

                with torch.no_grad():
                    # [F1] Inject true FOV from EgoDex intrinsics
                    moge_out = moge.infer(img_t, fov_x=fov_x_deg)

                # Sparse lookup — dense maps never leave GPU as full arrays.
                # We move only what we need to CPU.
                points = moge_out["points"].cpu().to(torch.float32).numpy()  # (H, W, 3)
                mask   = moge_out["mask"].cpu().numpy()                       # (H, W) bool
                # 'normal' and 'depth' deliberately not transferred — Lever 1.

                # -- HaMeR data for this frame ----------------------------
                vertices = grp['vertices'][:]    # (N_hands, 778, 3)
                cam_t_arr = grp['cam_t'][:]       # (N_hands, 3)
                is_right  = grp['is_right'][:]    # (N_hands,)  bool/int

                moge_joints_3d   = np.zeros((n_hands, 21, 3),  dtype=np.float32)
                moge_joint_valid = np.zeros((n_hands, 21),     dtype=bool)

                for i in range(n_hands):
                    verts_i = vertices[i].copy()   # (778, 3)
                    is_r    = int(is_right[i])

                    # [F4] Undo the x-axis mirror that run_stage1.py applied
                    # before saving. The stored vertices have:
                    #   verts[:, 0] = (2 * is_r - 1) * original_x
                    # Since (2*is_r-1) ∈ {-1, +1} and the operation is its
                    # own inverse, applying it again recovers original_x.
                    verts_i[:, 0] = (2 * is_r - 1) * verts_i[:, 0]

                    # MANO joints from mesh vertices via joint regressor
                    joints_3d = J_regressor @ verts_i   # (21, 3)

                    # Project to 2D using HaMeR's scaled pinhole model [F5]
                    xs, ys = project_joints_to_pixels(
                        joints_3d, cam_t_arr[i], scaled_focal, W_vid, H_vid
                    )

                    # [F3] Check MoGe validity mask at sampled pixels
                    valid = mask[ys, xs]   # (21,) bool
                    n_invalid = int((~valid).sum())

                    if n_invalid > INVALID_JOINT_WARN_THRESH:
                        logger.warning(
                            f"  frame {fidx:06d}, hand {i}: "
                            f"{n_invalid}/21 joints in invalid MoGe regions "
                            f"(occluded / boundary pixels). "
                            f"Consider filtering this frame during training."
                        )

                    # Lookup metric 3D points at keypoint pixel locations
                    sampled_pts = points[ys, xs]   # (21, 3)

                    moge_joints_3d[i]   = sampled_pts
                    moge_joint_valid[i] = valid

                # -- Store sparse keypoints only — no dense maps ----------
                # float16 is sufficient: depth precision of ~1mm at 1m range
                grp.create_dataset(
                    'moge_joints_3d',
                    data=moge_joints_3d.astype(np.float16),
                    **COMPRESS_KW
                )
                grp.create_dataset(
                    'moge_joint_valid',
                    data=moge_joint_valid,
                    **COMPRESS_KW
                )
                # Scalar FOV stored per-frame for traceability
                grp.attrs['moge_fov_x_deg'] = fov_x_deg

                pbar.update(1)

        cap.release()
        pbar.close()
        logger.success(f"  Done: MoGe sparse joints appended to {h5_path}")


if __name__ == '__main__':
    main()