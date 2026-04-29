"""
run_stage1_objects.py — Stage 1.2: Object Reconstruction

Reconstructs 3D geometry of the **grasped object** from egocentric video.
Goal: produce an object point cloud in HaMeR's camera space so ContactOpt
can resolve hand–object interpenetration (Step 1.3).

Pipeline per video:
  1. Parse grasped object name from EgoDex HDF5 `object` attribute.
  2. Run Grounding DINO on frame 0 → bounding box for the grasped object.
  3. Use SAM2 VideoPredictor to track the object across all frames.
  4. Run MoGe-2 per frame → dense depth map.
  5. Compute per-frame scale factor aligning MoGe depth to HaMeR depth.
  6. Back-project object mask pixels into HaMeR camera space.
  7. Subsample to max 2048 points and store in existing *_stage1.hdf5.

Coordinate system:
  Everything is stored in HaMeR's camera space (the same space as the hand
  mesh vertices + cam_t). MoGe-2 depth is scale-aligned per-frame using the
  hand depth from cam_t as reference.

Dependencies:
  - SAM2           (pip install sam-2, or from third_party/Grounded-SAM-2)
  - Grounding DINO (from third_party/Grounded-SAM-2 or pip)
  - MoGe-2         (third_party/ or pip install moge)
  - hdf5plugin     (optional, for LZ4 compression)
"""

import os
import sys
import re
import glob
import shutil
import argparse
import tempfile

import cv2
import h5py
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Optional LZ4 compression (same as run_stage1_moge.py)
# ---------------------------------------------------------------------------
try:
    import hdf5plugin  # noqa: F401
    COMPRESS_KW = dict(compression=hdf5plugin.LZ4())
    logger.info("Using LZ4 compression via hdf5plugin.")
except ImportError:
    COMPRESS_KW = dict(compression="gzip", compression_opts=4)
    logger.warning("hdf5plugin not found — falling back to gzip.")

# ---------------------------------------------------------------------------
# Third-party imports (will fail gracefully with helpful messages)
# ---------------------------------------------------------------------------
def _import_grounding_dino():
    """Import Grounding DINO inference utilities."""
    try:
        from groundingdino.util.inference import load_model, predict
        return load_model, predict
    except ImportError:
        # Try from Grounded-SAM-2 repo
        gsam2_root = os.path.abspath("third_party/Grounded-SAM-2")
        if os.path.isdir(gsam2_root):
            sys.path.insert(0, gsam2_root)
            from groundingdino.util.inference import load_model, predict
            return load_model, predict
        raise ImportError(
            "Cannot import groundingdino. Install via:\n"
            "  pip install groundingdino-py\n"
            "OR clone: git clone https://github.com/IDEA-Research/Grounded-SAM-2 third_party/Grounded-SAM-2"
        )


def _import_sam2():
    """Import SAM2 video predictor."""
    try:
        from sam2.build_sam import build_sam2_video_predictor
        return build_sam2_video_predictor
    except ImportError:
        sam2_root = os.path.abspath("third_party/Grounded-SAM-2")
        if os.path.isdir(sam2_root):
            sys.path.insert(0, sam2_root)
            from sam2.build_sam import build_sam2_video_predictor
            return build_sam2_video_predictor
        raise ImportError(
            "Cannot import sam2. Install via:\n"
            "  pip install sam-2\n"
            "OR clone: git clone https://github.com/IDEA-Research/Grounded-SAM-2 third_party/Grounded-SAM-2"
        )


# HaMeR projection constants
HAMER_FOCAL_LENGTH = 5000.0
HAMER_IMAGE_SIZE = 224.0

# Max points per object per frame after subsampling
MAX_OBJ_POINTS = 2048

# Grounding DINO detection thresholds
GD_BOX_THRESHOLD = 0.30
GD_TEXT_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_grasped_object(egodex_h5_path: str) -> str:
    """
    Extract the grasped object name from the EgoDex HDF5 `object` attribute.
    Format: "object:stapler, color:black" → "stapler"
    Falls back to first entry in `llm_objects` if `object` attr is missing.
    """
    with h5py.File(egodex_h5_path, "r") as f:
        obj_attr = f.attrs.get("object", "")
        llm_objects = list(f.attrs.get("llm_objects", []))

    # Try parsing structured format: "object:NAME, color:COLOR"
    if obj_attr:
        match = re.search(r"object:\s*([^,]+)", str(obj_attr))
        if match:
            return match.group(1).strip()

    # Fallback: first entry in llm_objects (usually the grasped item)
    if llm_objects:
        name = llm_objects[0]
        # llm_objects entries may be numpy strings
        return str(name).strip()

    return "object"  # last-resort generic prompt


def extract_frames_to_dir(video_path: str, output_dir: str) -> int:
    """
    Extract all frames from an MP4 as JPEG files (SAM2 requirement).
    Returns the number of extracted frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(output_dir, f"{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_idx += 1

    cap.release()
    return frame_idx


def compute_depth_scale_factor(
    moge_depth: np.ndarray,
    cam_t_arr: np.ndarray,
    vertices: np.ndarray,
    scaled_focal: float,
    W: int,
    H: int,
) -> float:
    """
    Compute the scale factor aligning MoGe-2 depth to HaMeR depth.
    Uses the median scale across all 778 hand vertices for extreme robustness.
    """
    scales = []
    
    for i in range(cam_t_arr.shape[0]):
        # Vertices are already unmirrored in the HDF5, just add translation
        verts_cam = vertices[i] + cam_t_arr[i]
        
        # Project all 778 vertices to 2D image space
        xs = (verts_cam[:, 0] / verts_cam[:, 2]) * scaled_focal + W / 2.0
        ys = (verts_cam[:, 1] / verts_cam[:, 2]) * scaled_focal + H / 2.0
        
        valid_mask = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        xs, ys = xs[valid_mask].astype(int), ys[valid_mask].astype(int)
        
        if len(xs) == 0:
            continue
            
        moge_zs = moge_depth[ys, xs]
        hamer_zs = verts_cam[valid_mask, 2]
        
        # Filter valid MoGe depths
        valid_depth = moge_zs > 1e-3
        if valid_depth.any():
            hand_scales = hamer_zs[valid_depth] / moge_zs[valid_depth]
            scales.append(np.median(hand_scales))

    if not scales:
        logger.warning("Could not compute depth scale from hands. Defaulting to 1.0.")
        return 1.0

    return float(np.median(scales))


def backproject_mask_to_hamer_space(
    mask: np.ndarray,
    moge_depth: np.ndarray,
    scale_factor: float,
    scaled_focal: float,
    W: int,
    H: int,
    hand_z_center: float = None
) -> np.ndarray:
    """
    Back-project object mask pixels, filtering out background bleed.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    Z = moge_depth[ys, xs].astype(np.float64) * scale_factor
    valid = Z > 1e-3
    xs, ys, Z = xs[valid], ys[valid], Z[valid]

    if len(Z) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    X = (xs.astype(np.float64) - W / 2.0) * Z / scaled_focal
    Y = (ys.astype(np.float64) - H / 2.0) * Z / scaled_focal

    points = np.stack([X, Y, Z], axis=-1).astype(np.float32)

    # --- FILTERING OUTLIERS (Crucial for ContactOpt) ---
    
    # 1. Depth filtering (removes table/background bleed)
    if hand_z_center is not None:
        # Keep object points within 30cm of the hand's Z-depth
        depth_mask = np.abs(points[:, 2] - hand_z_center) < 0.3
        points = points[depth_mask]
    else:
        # Fallback if no hands are in frame
        if len(points) > 0:
            med_z = np.median(points[:, 2])
            depth_mask = np.abs(points[:, 2] - med_z) < 0.2
            points = points[depth_mask]

    # 2. Spatial filtering (removes isolated floating pixels)
    if len(points) > 10:
        center = np.median(points, axis=0)
        dist = np.linalg.norm(points - center, axis=1)
        # Keep points within a 40cm radius sphere
        points = points[dist < 0.4]

    return points


def farthest_point_subsample(points: np.ndarray, n: int) -> np.ndarray:
    """
    Farthest-point subsampling: greedily selects n points that are
    maximally spread out. O(n * N) — fast enough for N < 100K.
    """
    if len(points) <= n:
        return points

    N = len(points)
    selected = np.zeros(n, dtype=int)
    distances = np.full(N, np.inf)

    # Start from a random point
    selected[0] = np.random.randint(N)

    for i in range(1, n):
        last = selected[i - 1]
        dist_to_last = np.sum((points - points[last]) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_last)
        selected[i] = np.argmax(distances)

    return points[selected]


def fov_x_from_intrinsics(K: np.ndarray, W: int) -> float:
    """Compute horizontal FOV in degrees from intrinsics matrix."""
    fx = float(K[0, 0])
    return float(np.degrees(2.0 * np.arctan(W / (2.0 * fx))))


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def process_video(
    video_path: str,
    egodex_h5_path: str,
    stage1_h5_path: str,
    gd_model,
    gd_predict_fn,
    sam2_predictor,
    moge_model,
    device: torch.device,
):
    """Process a single video: detect, track, reconstruct, store."""

    logger.info(f"\n{'='*60}\nProcessing: {stage1_h5_path}")

    # -- 1. Parse grasped object name --
    object_name = parse_grasped_object(egodex_h5_path)
    logger.info(f"  Grasped object: '{object_name}'")

    # -- Read metadata from stage1 HDF5 --
    with h5py.File(stage1_h5_path, "r") as hf:
        total_frames = int(hf.attrs.get("total_frames", 0))

    if total_frames == 0:
        logger.warning("  No frames in stage1 HDF5 — skipping.")
        return

    # -- Read camera intrinsics for FOV --
    with h5py.File(egodex_h5_path, "r") as ef:
        K_egodex = ef["/camera/intrinsic"][:]  # (3, 3)

    # -- Open video for resolution --
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"  Cannot open video '{video_path}' — skipping.")
        return

    ret, first_frame = cap.read()
    if not ret:
        logger.error(f"  Cannot read first frame — skipping.")
        cap.release()
        return

    H_vid, W_vid = first_frame.shape[:2]
    cap.release()

    fov_x_deg = fov_x_from_intrinsics(K_egodex, W_vid)
    scaled_focal = HAMER_FOCAL_LENGTH / HAMER_IMAGE_SIZE * max(W_vid, H_vid)

    logger.info(f"  Resolution: {W_vid}x{H_vid}")
    logger.info(f"  FOV_x: {fov_x_deg:.2f}°, scaled_focal: {scaled_focal:.1f}")

    # -- 2. Extract frames to temp dir (SAM2 needs JPEG directory) --
    frame_dir = tempfile.mkdtemp(
        prefix="sam2_frames_",
        dir=os.path.dirname(stage1_h5_path) or "."
    )
    logger.info(f"  Extracting {total_frames} frames to {frame_dir}")
    n_extracted = extract_frames_to_dir(video_path, frame_dir)
    logger.info(f"  Extracted {n_extracted} frames")

    try:
        # -- 3. Grounding DINO: detect object on frame 0 --
        # from PIL import Image as PILImage

        # frame0_path = os.path.join(frame_dir, "000000.jpg")
        # frame0_pil = PILImage.open(frame0_path).convert("RGB")

 
        # boxes, logits, phrases = gd_predict_fn(
        #     model=gd_model,
        #     image=frame0_pil,
        #     caption=object_name,
        #     box_threshold=GD_BOX_THRESHOLD,
        #     text_threshold=GD_TEXT_THRESHOLD,
        # )

        from groundingdino.util.inference import load_image

        frame0_path = os.path.join(frame_dir, "000000.jpg")
        image_source, image_tensor = load_image(frame0_path)

        logger.info(f"  Running Grounding DINO with prompt: '{object_name}'")
        boxes, logits, phrases = gd_predict_fn(
            model=gd_model,
            image=image_tensor,
            caption=object_name,
            box_threshold=GD_BOX_THRESHOLD,
            text_threshold=GD_TEXT_THRESHOLD,
        )

        if len(boxes) == 0:
            logger.warning(
                f"  Grounding DINO found no '{object_name}' in frame 0. "
                f"Trying with broader prompt..."
            )
            # Retry with the full description as a fallback
            with h5py.File(egodex_h5_path, "r") as ef:
                desc = str(ef.attrs.get("description", object_name))

            boxes, logits, phrases = gd_predict_fn(
                model=gd_model,
                image=frame0_pil,
                caption=desc,
                box_threshold=GD_BOX_THRESHOLD - 0.05,
                text_threshold=GD_TEXT_THRESHOLD - 0.05,
            )

        if len(boxes) == 0:
            logger.error(f"  No object detected even with fallback prompt — skipping.")
            return

        # Take the highest-confidence detection
        best_idx = logits.argmax()
        best_box = boxes[best_idx]  # [cx, cy, w, h] normalised → convert to [x1, y1, x2, y2]
        logger.info(
            f"  Detected '{phrases[best_idx]}' (conf={logits[best_idx]:.3f}), "
            f"box={best_box.tolist()}"
        )

        # Convert from normalised [cx, cy, w, h] to absolute [x1, y1, x2, y2]
        # Grounding DINO returns [cx, cy, w, h] normalised to [0, 1]
        box_abs = best_box.clone()
        box_abs[0] = (best_box[0] - best_box[2] / 2) * W_vid  # x1
        box_abs[1] = (best_box[1] - best_box[3] / 2) * H_vid  # y1
        box_abs[2] = (best_box[0] + best_box[2] / 2) * W_vid  # x2
        box_abs[3] = (best_box[1] + best_box[3] / 2) * H_vid  # y2

        # -- 4. SAM2: track object across all frames --
        logger.info("  Initializing SAM2 VideoPredictor...")
        inference_state = sam2_predictor.init_state(video_path=frame_dir)

        # Add bounding box prompt on frame 0
        sam2_predictor.add_new_points_or_box(
            inference_state,
            frame_idx=0,
            obj_id=1,
            box=box_abs.cpu().numpy(),
        )

        # Propagate forward
        logger.info("  Propagating SAM2 masks...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(
            inference_state
        ):
            # out_mask_logits: (N_objects, 1, H, W) float
            mask = (out_mask_logits[0] > 0.0).squeeze().cpu().numpy()  # (H, W) bool
            video_segments[out_frame_idx] = mask

        logger.info(f"  SAM2 tracked object across {len(video_segments)} frames")

        # Reset SAM2 state for next video
        sam2_predictor.reset_state(inference_state)

        # -- 5 & 6. MoGe-2 depth + back-projection, frame by frame --
        logger.info("  Running MoGe-2 depth extraction + back-projection...")

        cap = cv2.VideoCapture(video_path)
        pbar = tqdm(total=total_frames, desc=f"Objects {os.path.basename(video_path)}")

        with h5py.File(stage1_h5_path, "r+") as hf:
            # Store object metadata at file level
            hf.attrs["object_name"] = object_name

            for fidx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                grp_name = f"frame_{fidx:06d}"
                if grp_name not in hf:
                    pbar.update(1)
                    continue

                grp = hf[grp_name]

                # Skip if already processed (idempotent)
                if "obj_points_3d" in grp:
                    pbar.update(1)
                    continue

                # Get SAM2 mask for this frame
                mask = video_segments.get(fidx)
                if mask is None or not mask.any():
                    grp.attrs["obj_mask_valid"] = False
                    pbar.update(1)
                    continue

                # -- Run MoGe-2 for dense depth --
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_t = (
                    torch.from_numpy(img_rgb)
                    .to(device=device, dtype=torch.float16)
                    / 255.0
                ).permute(2, 0, 1)  # (3, H, W)

                with torch.no_grad():
                    moge_out = moge_model.infer(img_t, fov_x=fov_x_deg)

                depth = moge_out["depth"].cpu().to(torch.float32).numpy()  # (H, W)

                # -- Compute per-frame scale factor --
                n_hands = int(grp.attrs.get("n_hands", 0))
                if n_hands > 0:
                    vertices = grp["vertices"][:]
                    cam_t_arr = grp["cam_t"][:]

                    scale_factor = compute_depth_scale_factor(
                        depth, cam_t_arr, vertices,
                        scaled_focal, W_vid, H_vid,
                    )
                    # Get the median depth of the hands for filtering
                    hand_z_center = float(np.median(cam_t_arr[:, 2]))
                else:
                    scale_factor = 1.0
                    hand_z_center = None

                # -- Back-project object mask to HaMeR camera space --
                obj_points = backproject_mask_to_hamer_space(
                    mask, depth, scale_factor,
                    scaled_focal, W_vid, H_vid, hand_z_center
                )

                # -- Subsample to MAX_OBJ_POINTS --
                if len(obj_points) > MAX_OBJ_POINTS:
                    obj_points = farthest_point_subsample(obj_points, MAX_OBJ_POINTS)

                # -- Store results --
                grp.create_dataset(
                    "obj_mask",
                    data=mask,
                    **COMPRESS_KW,
                )
                grp.create_dataset(
                    "obj_points_3d",
                    data=obj_points.astype(np.float16),
                    **COMPRESS_KW,
                )
                grp.attrs["obj_mask_valid"] = True
                grp.attrs["obj_n_points"] = len(obj_points)
                grp.attrs["obj_depth_scale"] = scale_factor

                pbar.update(1)

        cap.release()
        pbar.close()
        logger.success(f"  Done: object data appended to {stage1_h5_path}")

    finally:
        # Clean up temp frame directory
        shutil.rmtree(frame_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1.2 — Object Reconstruction (Grounding DINO + SAM2 + MoGe-2)"
    )
    parser.add_argument(
        "--input_dir", type=str, default="basic_pick_place",
        help="Directory containing EgoDex .mp4 + .hdf5 files"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Directory containing *_stage1.hdf5 files from run_stage1.py"
    )
    parser.add_argument(
        "--limit", type=int, default=2,
        help="Max videos to process (0 = all)"
    )
    # -- Model paths --
    parser.add_argument(
        "--gd_config", type=str,
        default="third_party/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="Path to Grounding DINO config"
    )
    parser.add_argument(
        "--gd_checkpoint", type=str,
        default="third_party/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth",
        help="Path to Grounding DINO checkpoint"
    )
    parser.add_argument(
        "--sam2_config", type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 model config"
    )
    parser.add_argument(
        "--sam2_checkpoint", type=str,
        default="third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
        help="Path to SAM2 checkpoint"
    )
    parser.add_argument(
        "--moge_model", type=str,
        default="Ruicheng/moge-2-vitl-normal",
        help="MoGe-2 model ID (HuggingFace)"
    )

    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/stage1_objects.log", rotation="500 MB")
    logger.info("=== Starting Stage 1.2 — Object Reconstruction ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # -- Load models --
    logger.info("Loading Grounding DINO...")
    gd_load_model, gd_predict = _import_grounding_dino()
    gd_model = gd_load_model(args.gd_config, args.gd_checkpoint)

    logger.info("Loading SAM2 VideoPredictor...")
    build_sam2_video_predictor = _import_sam2()
    sam2_predictor = build_sam2_video_predictor(
        args.sam2_config, args.sam2_checkpoint, device=device
    )

    logger.info("Loading MoGe-2...")
    from moge.model.v2 import MoGeModel
    moge_model = MoGeModel.from_pretrained(args.moge_model).to(device)
    moge_model = moge_model.half()
    moge_model.eval()

    # -- Find stage1 HDF5 files --
    stage1_files = sorted(glob.glob(os.path.join(args.output_dir, "*_stage1.hdf5")))

    if not stage1_files:
        logger.warning(f"No *_stage1.hdf5 files found in '{args.output_dir}'. "
                       "Run run_stage1.py first.")
        return

    if args.limit > 0:
        stage1_files = stage1_files[:args.limit]

    logger.info(f"Processing {len(stage1_files)} file(s)")

    for h5_path in stage1_files:
        # Derive paths
        basename = os.path.basename(h5_path).replace("_stage1.hdf5", "")
        video_path = os.path.join(args.input_dir, f"{basename}.mp4")
        egodex_h5 = os.path.join(args.input_dir, f"{basename}.hdf5")

        if not os.path.isfile(video_path):
            logger.warning(f"  Video not found: {video_path} — skipping.")
            continue
        if not os.path.isfile(egodex_h5):
            logger.warning(f"  EgoDex HDF5 not found: {egodex_h5} — skipping.")
            continue

        try:
            process_video(
                video_path=video_path,
                egodex_h5_path=egodex_h5,
                stage1_h5_path=h5_path,
                gd_model=gd_model,
                gd_predict_fn=gd_predict,
                sam2_predictor=sam2_predictor,
                moge_model=moge_model,
                device=device,
            )
        except Exception as e:
            logger.error(f"  Failed on {h5_path}: {e}")
            logger.exception(e)

    logger.info("=== Stage 1.2 Pipeline Complete ===")


if __name__ == "__main__":
    main()
