"""
run_stage1_contactopt.py — Stage 1.3: Contact Retargeting via ContactOpt

Reads *_stage1.hdf5 files produced by Stage 1.1 (HaMeR) and Stage 1.2
(object point clouds), runs ContactOpt to fix hand-object interpenetration
and floating-hand artifacts, and writes optimized MANO parameters + vertices
back into the same HDF5.

Two problems solved before ContactOpt can run:

  Problem 1 — MANO pose format mismatch:
    HaMeR stores hand_pose as (N_hands, 15, 3, 3) rotation matrices.
    ContactOpt expects 15 PCA components.
    Fix: rot_mat → axis_angle (pytorch3d) → PCA-15 (fit_pca_to_axang)

  Problem 2 — Depth scale / absolute scale:
    MoGe-2 depth aligned to HaMeR's camera space puts objects at ~16–22m.
    ContactOpt's DeepContact was trained on ContactPose (~0.18m hands).
    Fix: normalize both hand mesh and object point cloud so the hand spans
    ~0.18m (MANO canonical size), run ContactOpt, then invert the scale.

  Problem 3 — No object mesh (only point clouds):
    ContactOpt needs a triangle mesh, not a point cloud.
    Fix: Poisson surface reconstruction via Open3D on obj_points_3d.

Evaluation metrics computed per-frame and saved to HDF5:
  - interpenetration_depth_before / _after  (mean mm of penetration)
  - contact_coverage_before / _after        (fraction of finger verts in contact)
  - contactopt_converged                    (bool)

Usage:
    python run_stage1_contactopt.py --output_dir output/
    
    # Visualize a single frame (requires open3d):
    python run_stage1_contactopt.py --output_dir output/ --vis_frame 0_stage1.hdf5:42
"""

import os
import sys
import glob
import argparse
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

# ── ContactOpt repo path ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTACTOPT_ROOT = os.path.join(BASE_DIR, "third_party", "contactopt")
if not os.path.isdir(CONTACTOPT_ROOT):
    logger.error(
        f"ContactOpt not found at {CONTACTOPT_ROOT}.\n"
        "Clone it: git clone https://github.com/facebookresearch/ContactOpt "
        "third_party/contactopt"
    )
    sys.exit(1)
sys.path.insert(0, CONTACTOPT_ROOT)

# ── MANO canonical hand size (wrist→middle fingertip, metres) ───────────────
MANO_CANONICAL_SPAN_M = 0.18

# ContactOpt contact threshold — vertices within this distance are "in contact"
CONTACT_THRESH_M = 0.005   # 5 mm, in metric space (after normalisation)

# Only run ContactOpt on frames inside a temporal window around peak contact.
# Set to None to run on ALL frames with valid hand+object data.
# If you know your grasping phase (e.g. frames 20–80), set this per-video.
# For now we use a simple heuristic: frames where hand-object distance < threshold.
GRASP_DETECTION_PERCENTILE = 30   # run on frames in bottom-N% of hand-obj distance


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  MANO pose conversion utilities
# ═══════════════════════════════════════════════════════════════════════════════

def rotmat_to_axisangle(rotmats: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrices to axis-angle vectors.

    Args:
        rotmats: (..., 3, 3)
    Returns:
        axisangle: (..., 3)
    """
    try:
        from pytorch3d.transforms import matrix_to_axis_angle
        t = torch.from_numpy(rotmats).float()
        aa = matrix_to_axis_angle(t)
        return aa.numpy()
    except ImportError:
        pass

    # Fallback: scipy
    from scipy.spatial.transform import Rotation
    orig_shape = rotmats.shape[:-2]
    flat = rotmats.reshape(-1, 3, 3)
    aa = Rotation.from_matrix(flat).as_rotvec()   # (N, 3)
    return aa.reshape(*orig_shape, 3)


# def rotmats_to_pca15(
#     global_orient_rotmat: np.ndarray,   # (N_hands, 1, 3, 3)
#     hand_pose_rotmat: np.ndarray,       # (N_hands, 15, 3, 3)
# ) -> tuple:
#     """
#     Convert HaMeR rotation-matrix MANO params to ContactOpt's PCA-15 format.

#     Returns:
#         global_orient_aa:  (N_hands, 3)        axis-angle
#         hand_pose_pca:     (N_hands, 15)        PCA components
#     """
#     # Import ContactOpt's PCA converter
#     from contactopt.util import fit_pca_to_axang

#     N = global_orient_rotmat.shape[0]

#     # global_orient: (N, 1, 3, 3) → (N, 3)
#     go_aa = rotmat_to_axisangle(global_orient_rotmat[:, 0])   # (N, 3)

#     # hand_pose: (N, 15, 3, 3) → (N, 15, 3) → (N, 45)
#     hp_aa = rotmat_to_axisangle(hand_pose_rotmat)             # (N, 15, 3)
#     hp_aa_flat = hp_aa.reshape(N, 45)

#     # → (N, 15) PCA
#     hp_pca = fit_pca_to_axang(hp_aa_flat)                     # (N, 15)

#     return go_aa, hp_pca

def rotmats_to_pca15(global_orient_rotmat, hand_pose_rotmat, betas):
    from contactopt.util import fit_pca_to_axang

    N = global_orient_rotmat.shape[0]

    go_aa = rotmat_to_axisangle(global_orient_rotmat[:,0])

    hp_aa = rotmat_to_axisangle(hand_pose_rotmat)
    hp_aa_flat = hp_aa.reshape(N,45)

    hp_pca = fit_pca_to_axang(hp_aa_flat, betas)

    return go_aa, hp_pca


def pca15_to_rotmats(
    global_orient_aa: np.ndarray,   # (N_hands, 3)
    hand_pose_pca: np.ndarray,      # (N_hands, 15)
) -> tuple:
    """
    Inverse of rotmats_to_pca15.  Converts ContactOpt output back to the
    rotation-matrix format stored in HDF5.

    Returns:
        global_orient_rotmat: (N_hands, 1, 3, 3)
        hand_pose_rotmat:     (N_hands, 15, 3, 3)
    """
    from contactopt.util import pca_to_axang   # inverse of fit_pca_to_axang

    N = global_orient_aa.shape[0]

    # PCA-15 → axis-angle-45
    hp_aa_flat = pca_to_axang(hand_pose_pca)                  # (N, 45)
    hp_aa = hp_aa_flat.reshape(N, 15, 3)                      # (N, 15, 3)

    try:
        from pytorch3d.transforms import axis_angle_to_matrix
        go_rm = axis_angle_to_matrix(
            torch.from_numpy(global_orient_aa).float()
        ).numpy()                                              # (N, 3, 3)
        hp_rm = axis_angle_to_matrix(
            torch.from_numpy(hp_aa).float()
        ).numpy()                                              # (N, 15, 3, 3)
    except ImportError:
        from scipy.spatial.transform import Rotation
        go_rm = Rotation.from_rotvec(global_orient_aa).as_matrix()
        hp_rm = Rotation.from_rotvec(hp_aa.reshape(-1, 3)).as_matrix().reshape(N, 15, 3, 3)

    return go_rm[:, None, :, :], hp_rm   # (N,1,3,3), (N,15,3,3)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Scale normalisation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hand_scale_factor(vertices: np.ndarray) -> float:
    """
    Compute scale factor so hand spans ~MANO_CANONICAL_SPAN_M.
    Uses the bounding-box diagonal of the hand mesh as a proxy for hand size.

    Args:
        vertices: (N_hands, 778, 3)
    Returns:
        scale_factor: scalar — multiply coords by this to get metric scale
    """
    # Use the first hand (index 0)
    verts = vertices[0]   # (778, 3)
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    current_span = float(np.linalg.norm(bbox_max - bbox_min))
    if current_span < 1e-6:
        logger.warning("Hand mesh has near-zero span — defaulting scale to 1.0")
        return 1.0
    return MANO_CANONICAL_SPAN_M / current_span


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Object mesh from point cloud (Poisson reconstruction)
# ═══════════════════════════════════════════════════════════════════════════════

def pointcloud_to_mesh(points: np.ndarray, depth: int = 7):
    """
    Reconstruct a triangle mesh from a point cloud using Poisson surface
    reconstruction (Open3D).

    Args:
        points:  (N, 3) float32/float16 point cloud
        depth:   Poisson octree depth (6=coarse, 9=fine; 7 is good for objects)

    Returns:
        o3d.geometry.TriangleMesh, or None on failure
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "Open3D is required for Poisson reconstruction.\n"
            "Install: pip install open3d"
        )

    pts = points.astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Estimate normals — required for Poisson
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=20)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.1, linear_fit=False
    )

    # Remove low-density vertices (artefacts at the boundary)
    densities_np = np.asarray(densities)
    thresh = np.percentile(densities_np, 10)
    mesh.remove_vertices_by_mask(densities_np < thresh)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    if len(mesh.vertices) < 50:
        logger.warning("Poisson reconstruction produced <50 vertices — likely failed")
        return None

    return mesh


def mesh_to_arrays(mesh) -> tuple:
    """
    Extract vertices and faces from an Open3D mesh.
    Returns:
        verts: (V, 3) float32
        faces: (F, 3) int32
    """
    import open3d as o3d
    verts = np.asarray(mesh.vertices).astype(np.float32)
    faces = np.asarray(mesh.triangles).astype(np.int32)
    return verts, faces


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  ContactOpt runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_contactopt_on_frame(
    global_orient_aa: np.ndarray,   # (1, 3)   — single hand
    hand_pose_pca:    np.ndarray,   # (1, 15)
    betas:            np.ndarray,   # (1, 10)
    cam_t:            np.ndarray,   # (1, 3)
    obj_verts:        np.ndarray,   # (V, 3)
    obj_faces:        np.ndarray,   # (F, 3)
    is_right:         float,        # 1.0 = right hand
    device:           torch.device,
) -> dict:
    """
    Run ContactOpt on a single hand-object pair.

    Returns dict with:
        'global_orient_aa':   (1, 3)   optimised
        'hand_pose_pca':      (1, 15)  optimised
        'opt_verts':          (778, 3) optimised hand mesh vertices
        'converged':          bool
    """
    from contactopt.run_contactopt import run_contactopt   # main optimiser
    from contactopt.hand_object import HandObject

    # Build a HandObject — ContactOpt's data structure
    # It wraps MANO forward pass + differentiable contact
    ho = HandObject()
    ho.load_from_params(
        pose=torch.from_numpy(hand_pose_pca).float().to(device),          # (1, 15)
        global_orient=torch.from_numpy(global_orient_aa).float().to(device),  # (1, 3)
        betas=torch.from_numpy(betas).float().to(device),                 # (1, 10)
        transl=torch.from_numpy(cam_t).float().to(device),               # (1, 3)
        obj_verts=torch.from_numpy(obj_verts).float().to(device),        # (V, 3)
        obj_faces=torch.from_numpy(obj_faces).long().to(device),         # (F, 3)
        is_right=bool(is_right),
    )

    try:
        result_ho = run_contactopt(ho, device=device)
        opt_pose  = result_ho.hand_pose.detach().cpu().numpy()         # (1, 15)
        opt_go    = result_ho.global_orient.detach().cpu().numpy()     # (1, 3)
        opt_verts = result_ho.hand_verts.detach().cpu().numpy()        # (778, 3)
        converged = True
    except Exception as e:
        logger.warning(f"ContactOpt failed: {e}")
        opt_pose  = hand_pose_pca
        opt_go    = global_orient_aa
        opt_verts = None
        converged = False

    return {
        "global_orient_aa": opt_go,
        "hand_pose_pca":    opt_pose,
        "opt_verts":        opt_verts,
        "converged":        converged,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_interpenetration_depth(hand_verts: np.ndarray, obj_verts: np.ndarray) -> float:
    """
    Mean penetration depth: for each hand vertex inside the object bbox,
    compute distance to the nearest object vertex.
    Returns mean penetration depth in the same units as input (metres after normalisation).
    This is an approximation (true SDF-based penetration needs a watertight mesh),
    but robust and fast.
    """
    # Build a simple occupancy test using convex hull of object
    # Fast approximation: hand vertices whose z is between obj z_min and z_max
    # AND whose distance to nearest obj vertex is below a threshold are "penetrating"
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(obj_verts)
        dists, _ = tree.query(hand_verts, k=1)
        # Vertices "penetrating" = those closer than 1mm (within the object surface)
        penetrating = dists < 0.001
        if penetrating.sum() == 0:
            return 0.0
        return float(dists[penetrating].mean())
    except Exception:
        return 0.0


def compute_contact_coverage(hand_verts: np.ndarray, obj_verts: np.ndarray) -> float:
    """
    Fraction of hand vertices within CONTACT_THRESH_M of any object vertex.
    Higher = better contact (hand is actually touching the object).
    """
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(obj_verts)
        dists, _ = tree.query(hand_verts, k=1)
        return float((dists < CONTACT_THRESH_M).mean())
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Grasping frame detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_grasp_frames(h5_file, total_frames, percentile=30):
    distances = []
    frame_indices = []

    for fidx in range(total_frames):
        key = f"frame_{fidx:06d}"
        if key not in h5_file:
            continue

        grp = h5_file[key]

        if int(grp.attrs.get("n_hands", 0)) == 0:
            continue

        if "obj_points_3d" not in grp:
            continue

        if grp.attrs.get("obj_n_points", 0) < 30:
            continue

        cam_t = grp["cam_t"][:].astype(np.float32)
        hand_centroid = cam_t.mean(axis=0)

        obj_pts = grp["obj_points_3d"][:].astype(np.float32)

        if len(obj_pts) == 0:
            continue

        if not np.isfinite(obj_pts).all():
            continue

        obj_centroid = obj_pts.mean(axis=0)

        if not np.isfinite(obj_centroid).all():
            continue

        dist = float(np.linalg.norm(hand_centroid - obj_centroid))

        if not np.isfinite(dist):
            continue

        distances.append(dist)
        frame_indices.append(fidx)

    if len(distances) == 0:
        return []

    threshold = np.percentile(distances, percentile)


    # logger.info(
    #     f"  Grasp frame detection: {len(grasp_frames)}/{len(frame_indices)} frames "
    #     f"selected (distance threshold: {threshold:.4f})"
    # )

    return [
        fidx for fidx, d in zip(frame_indices, distances)
        if d <= threshold
    ]

    # return grasp_frames


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Per-video pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def process_video_h5(h5_path: str, device: torch.device, force_rerun: bool = False):
    logger.info(f"\n{'='*60}")
    logger.info(f"Stage 1.3 ContactOpt: {h5_path}")

    with h5py.File(h5_path, "r+") as hf:
        total_frames = int(hf.attrs.get("total_frames", 0))
        object_name  = str(hf.attrs.get("object_name", "object"))

        logger.info(f"  Object: {object_name}, Total frames: {total_frames}")

        # Detect grasping frames
        grasp_frames = detect_grasp_frames(hf, total_frames, GRASP_DETECTION_PERCENTILE)
        if not grasp_frames:
            logger.warning("  No valid grasping frames found — skipping.")
            return

        # ── Build object mesh ONCE per video ────────────────────────────
        # Aggregate object points across all grasping frames for a better mesh
        logger.info("  Building object mesh from point cloud...")
        all_obj_pts = []
        for fidx in grasp_frames[:20]:   # use first 20 grasping frames
            key = f"frame_{fidx:06d}"
            if key not in hf or "obj_points_3d" not in hf[key]:
                continue
            pts = hf[key]["obj_points_3d"][:].astype(np.float32)
            all_obj_pts.append(pts)

        if not all_obj_pts:
            logger.error("  No object points available — cannot build mesh.")
            return

        combined_pts = np.concatenate(all_obj_pts, axis=0)

        # Subsample to keep Poisson tractable (max 10K points)
        if len(combined_pts) > 10000:
            idx = np.random.choice(len(combined_pts), 10000, replace=False)
            combined_pts = combined_pts[idx]

        # We need a representative scale factor — get it from the first grasping frame
        first_key = f"frame_{grasp_frames[0]:06d}"
        first_verts = hf[first_key]["vertices"][:].astype(np.float32)
        scale_factor = compute_hand_scale_factor(first_verts)
        logger.info(f"  Scale factor (HaMeR→metric): {scale_factor:.6f}")

        # Normalise object points to metric scale for Poisson + ContactOpt
        combined_pts_metric = combined_pts * scale_factor

        obj_mesh = pointcloud_to_mesh(combined_pts_metric, depth=7)
        if obj_mesh is None:
            logger.error("  Poisson reconstruction failed — skipping.")
            return

        obj_verts_metric, obj_faces = mesh_to_arrays(obj_mesh)
        logger.info(
            f"  Object mesh: {len(obj_verts_metric)} verts, {len(obj_faces)} faces"
        )

        # ── Per-frame ContactOpt ─────────────────────────────────────────
        n_success = 0
        n_skip    = 0

        pbar = tqdm(grasp_frames, desc=f"ContactOpt [{object_name}]")

        for fidx in pbar:
            key = f"frame_{fidx:06d}"
            grp = hf[key]

            # Skip already-processed frames unless forced
            if not force_rerun and grp.attrs.get("contactopt_done", False):
                n_skip += 1
                continue

            n_hands = int(grp.attrs.get("n_hands", 0))
            if n_hands == 0:
                continue

            raw_verts    = grp["vertices"][:].astype(np.float32)       # (N, 778, 3)
            raw_cam_t    = grp["cam_t"][:].astype(np.float32)          # (N, 3)
            raw_go_rm    = grp["global_orient"][:].astype(np.float32)  # (N, 1, 3, 3)
            raw_hp_rm    = grp["hand_pose"][:].astype(np.float32)      # (N, 15, 3, 3)
            raw_betas    = grp["betas"][:].astype(np.float32)          # (N, 10)
            raw_is_right = grp["is_right"][:]                          # (N,)

            # Per-frame metrics storage
            interpen_before_list = []
            contact_before_list  = []
            interpen_after_list  = []
            contact_after_list   = []

            opt_verts_all  = []
            opt_go_rm_all  = []
            opt_hp_rm_all  = []
            converged_all  = []

            for hand_idx in range(n_hands):
                # ── Scale normalise ──────────────────────────────────────
                verts_metric = raw_verts[hand_idx] * scale_factor    # (778, 3)
                cam_t_metric = raw_cam_t[hand_idx] * scale_factor    # (3,)

                # ── Convert pose format ──────────────────────────────────
                betas_single = raw_betas[hand_idx:hand_idx+1]

                go_aa, hp_pca = rotmats_to_pca15(
                    raw_go_rm[hand_idx:hand_idx+1],
                    raw_hp_rm[hand_idx:hand_idx+1],
                    betas_single,
                )

                is_right = float(raw_is_right[hand_idx]) # (1, 10)
                is_right     = float(raw_is_right[hand_idx])

                # ── Metrics BEFORE ───────────────────────────────────────
                interpen_before = compute_interpenetration_depth(verts_metric, obj_verts_metric)
                contact_before  = compute_contact_coverage(verts_metric, obj_verts_metric)
                interpen_before_list.append(interpen_before)
                contact_before_list.append(contact_before)

                # ── Run ContactOpt ───────────────────────────────────────
                result = run_contactopt_on_frame(
                    global_orient_aa=go_aa,
                    hand_pose_pca=hp_pca,
                    betas=betas_single,
                    cam_t=cam_t_metric[None, :],    # (1, 3)
                    obj_verts=obj_verts_metric,
                    obj_faces=obj_faces,
                    is_right=is_right,
                    device=device,
                )

                # ── Metrics AFTER ────────────────────────────────────────
                if result["opt_verts"] is not None:
                    opt_v = result["opt_verts"]   # already in metric space
                    interpen_after = compute_interpenetration_depth(opt_v, obj_verts_metric)
                    contact_after  = compute_contact_coverage(opt_v, obj_verts_metric)
                    # Invert scale normalisation to store back in HaMeR space
                    opt_v_hamer = opt_v / scale_factor
                else:
                    interpen_after = interpen_before
                    contact_after  = contact_before
                    opt_v_hamer    = raw_verts[hand_idx]

                interpen_after_list.append(interpen_after)
                contact_after_list.append(contact_after)

                # ── Convert pose back to rot-mat ─────────────────────────
                opt_go_rm, opt_hp_rm = pca15_to_rotmats(
                    result["global_orient_aa"],   # (1, 3)
                    result["hand_pose_pca"],       # (1, 15)
                )
                # opt_go_rm: (1, 1, 3, 3), opt_hp_rm: (1, 15, 3, 3)

                opt_verts_all.append(opt_v_hamer)
                opt_go_rm_all.append(opt_go_rm[0])    # (1, 3, 3)
                opt_hp_rm_all.append(opt_hp_rm[0])    # (15, 3, 3)
                converged_all.append(result["converged"])

            # ── Write results to HDF5 ────────────────────────────────────
            # Store optimised params alongside originals (don't overwrite — keep originals)
            def _replace_or_create(grp, name, data):
                if name in grp:
                    del grp[name]
                grp.create_dataset(name, data=data, compression="gzip")

            opt_verts_arr = np.stack(opt_verts_all)       # (N, 778, 3)
            opt_go_arr    = np.stack(opt_go_rm_all)       # (N, 1, 3, 3)
            opt_hp_arr    = np.stack(opt_hp_rm_all)       # (N, 15, 3, 3)

            _replace_or_create(grp, "opt_vertices",     opt_verts_arr)
            _replace_or_create(grp, "opt_global_orient", opt_go_arr)
            _replace_or_create(grp, "opt_hand_pose",    opt_hp_arr)

            # Evaluation metrics
            grp.attrs["contactopt_done"]             = True
            grp.attrs["contactopt_converged"]        = all(converged_all)
            grp.attrs["interpen_before_mm"]          = float(np.mean(interpen_before_list)) * 1000
            grp.attrs["interpen_after_mm"]           = float(np.mean(interpen_after_list))  * 1000
            grp.attrs["contact_coverage_before"]     = float(np.mean(contact_before_list))
            grp.attrs["contact_coverage_after"]      = float(np.mean(contact_after_list))
            grp.attrs["scale_factor"]                = scale_factor

            n_success += 1

        logger.success(
            f"  Done: {n_success} frames optimised, {n_skip} skipped (already done)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Summary report across all frames
# ═══════════════════════════════════════════════════════════════════════════════

def print_evaluation_summary(h5_path: str):
    """Print before/after ContactOpt metrics for a processed HDF5 file."""
    with h5py.File(h5_path, "r") as hf:
        total = int(hf.attrs.get("total_frames", 0))
        object_name = str(hf.attrs.get("object_name", "?"))

        before_ip, after_ip = [], []
        before_cc, after_cc = [], []
        converged_count = 0
        n_done = 0

        for fidx in range(total):
            key = f"frame_{fidx:06d}"
            if key not in hf:
                continue
            grp = hf[key]
            if not grp.attrs.get("contactopt_done", False):
                continue

            n_done += 1
            before_ip.append(grp.attrs.get("interpen_before_mm", 0))
            after_ip.append(grp.attrs.get("interpen_after_mm",  0))
            before_cc.append(grp.attrs.get("contact_coverage_before", 0))
            after_cc.append(grp.attrs.get("contact_coverage_after",  0))
            if grp.attrs.get("contactopt_converged", False):
                converged_count += 1

        print(f"\n── ContactOpt Evaluation: {os.path.basename(h5_path)} [{object_name}] ──")
        print(f"  Frames processed:        {n_done}")
        print(f"  Converged:               {converged_count}/{n_done}")
        if n_done > 0:
            print(f"  Interpenetration depth:  "
                  f"{np.mean(before_ip):.2f} mm  →  {np.mean(after_ip):.2f} mm  "
                  f"({'↓' if np.mean(after_ip) < np.mean(before_ip) else '↑'} "
                  f"{abs(np.mean(after_ip) - np.mean(before_ip)):.2f} mm)")
            print(f"  Contact coverage:        "
                  f"{np.mean(before_cc)*100:.1f}%  →  {np.mean(after_cc)*100:.1f}%  "
                  f"({'↑' if np.mean(after_cc) > np.mean(before_cc) else '↓'} "
                  f"{abs(np.mean(after_cc) - np.mean(before_cc))*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Optional visualisation (single frame)
# ═══════════════════════════════════════════════════════════════════════════════

def visualise_frame(h5_path: str, fidx: int):
    """
    Open3D visualisation of before/after ContactOpt for a single frame.
    Shows: original hand (red), optimised hand (green), object (blue).
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.error("open3d required for visualisation: pip install open3d")
        return

    with h5py.File(h5_path, "r") as hf:
        key = f"frame_{fidx:06d}"
        if key not in hf:
            logger.error(f"Frame {fidx} not found")
            return
        grp = hf[key]

        if "opt_vertices" not in grp:
            logger.error("Frame not yet processed by ContactOpt")
            return

        orig_verts = grp["vertices"][0].astype(np.float32)     # (778, 3)
        opt_verts  = grp["opt_vertices"][0].astype(np.float32) # (778, 3)
        obj_pts    = grp["obj_points_3d"][:].astype(np.float32)

        scale = float(grp.attrs.get("scale_factor", 1.0))

    # Scale to metric for display
    orig_verts_m = orig_verts * scale
    opt_verts_m  = opt_verts  * scale
    obj_pts_m    = obj_pts    * scale

    def make_pcd(pts, color):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(color)
        return pcd

    orig_pcd = make_pcd(orig_verts_m, [1.0, 0.2, 0.2])   # red  = before
    opt_pcd  = make_pcd(opt_verts_m,  [0.2, 0.9, 0.2])   # green = after
    obj_pcd  = make_pcd(obj_pts_m,    [0.2, 0.4, 1.0])   # blue = object

    print(f"\nVisualising frame {fidx}:")
    print("  RED   = original HaMeR hand (before ContactOpt)")
    print("  GREEN = optimised hand (after ContactOpt)")
    print("  BLUE  = object point cloud")
    print("  Press Q to quit")
    o3d.visualization.draw_geometries([orig_pcd, opt_pcd, obj_pcd])


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1.3 — Contact Retargeting via ContactOpt"
    )
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory containing *_stage1.hdf5 files")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max videos to process (0 = all)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if contactopt_done=True")
    parser.add_argument("--vis_frame", type=str, default=None,
                        help="Visualise a single frame: 'filename.hdf5:frame_idx'")
    parser.add_argument("--summary_only", action="store_true",
                        help="Print evaluation summary without running ContactOpt")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/stage1_contactopt.log", rotation="100 MB")

    # ── Visualisation mode ───────────────────────────────────────────────────
    if args.vis_frame:
        parts = args.vis_frame.split(":")
        h5_path = parts[0]
        fidx = int(parts[1]) if len(parts) > 1 else 0
        visualise_frame(h5_path, fidx)
        return

    # ── Find HDF5 files ──────────────────────────────────────────────────────
    h5_files = sorted(glob.glob(os.path.join(args.output_dir, "*_stage1.hdf5")))
    if not h5_files:
        logger.error(f"No *_stage1.hdf5 files found in '{args.output_dir}'")
        sys.exit(1)
    if args.limit > 0:
        h5_files = h5_files[:args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Processing {len(h5_files)} file(s)")

    # ── Summary-only mode ────────────────────────────────────────────────────
    if args.summary_only:
        for h5_path in h5_files:
            print_evaluation_summary(h5_path)
        return

    # ── Main loop ────────────────────────────────────────────────────────────
    for h5_path in h5_files:
        try:
            process_video_h5(h5_path, device=device, force_rerun=args.force)
            print_evaluation_summary(h5_path)
        except Exception as e:
            logger.error(f"Failed on {h5_path}: {e}")
            logger.exception(e)

    logger.info("=== Stage 1.3 Complete ===")


if __name__ == "__main__":
    main()