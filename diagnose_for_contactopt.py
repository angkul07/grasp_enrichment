"""
diagnose_for_contactopt.py
--------------------------
Inspects *_stage1.hdf5 files produced by run_stage1.py (Stage 1.1) and
run_stage1_objects.py (Stage 1.2) and reports everything needed for
ContactOpt (Stage 1.3).

Run:
    python diagnose_for_contactopt.py --output_dir output/

Then paste the printed report into the chat.
"""

import os
import sys
import glob
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

# ── optional: rich pretty-printing ──────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None

SEP = "=" * 70


def fmt(val):
    if val is True:  return "✅  YES"
    if val is False: return "❌  NO"
    return str(val)


# ──────────────────────────────────────────────────────────────────────────────
# Per-file analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyse_file(h5_path: str) -> dict:
    report = {
        "path": h5_path,
        "file_exists": os.path.isfile(h5_path),
        "file_size_mb": round(os.path.getsize(h5_path) / 1e6, 2) if os.path.isfile(h5_path) else 0,
    }

    if not report["file_exists"]:
        report["error"] = "File not found"
        return report

    with h5py.File(h5_path, "r") as hf:

        # ── Top-level attributes ──────────────────────────────────────────
        report["total_frames"]      = int(hf.attrs.get("total_frames", -1))
        report["frames_with_hands"] = int(hf.attrs.get("frames_with_hands", -1))
        report["object_name"]       = str(hf.attrs.get("object_name", "<missing>"))
        report["source_video"]      = str(hf.attrs.get("source_video", "<missing>"))

        # ── Walk frames ──────────────────────────────────────────────────
        frame_keys = sorted([k for k in hf.keys() if k.startswith("frame_")])
        n_frames = len(frame_keys)

        # Counters
        n_has_hands       = 0
        n_has_obj_pts     = 0
        n_has_global_orient = 0
        n_has_hand_pose   = 0
        n_has_betas       = 0
        n_has_cam_t       = 0
        n_has_vertices    = 0
        n_contactopt_done = 0
        n_obj_mask_valid  = 0

        # Shape samples (from first frame that has data)
        sample_shapes = {}
        sampled = False

        for key in frame_keys:
            grp = hf[key]
            n_hands = int(grp.attrs.get("n_hands", 0))

            has_v  = "vertices"      in grp
            has_ct = "cam_t"         in grp
            has_go = "global_orient" in grp
            has_hp = "hand_pose"     in grp
            has_bt = "betas"         in grp
            has_op = "obj_points_3d" in grp
            has_om = grp.attrs.get("obj_mask_valid", False)
            done   = bool(grp.attrs.get("contactopt_done", False))

            if n_hands > 0:
                n_has_hands += 1
            if has_v:  n_has_vertices     += 1
            if has_ct: n_has_cam_t        += 1
            if has_go: n_has_global_orient+= 1
            if has_hp: n_has_hand_pose    += 1
            if has_bt: n_has_betas        += 1
            if has_op: n_has_obj_pts      += 1
            if has_om: n_obj_mask_valid   += 1
            if done:   n_contactopt_done  += 1

            # Grab shapes from first hand-bearing frame
            if not sampled and n_hands > 0:
                for ds_name in ["vertices", "cam_t", "global_orient", "hand_pose", "betas"]:
                    if ds_name in grp:
                        sample_shapes[ds_name] = tuple(grp[ds_name].shape)
                if "obj_points_3d" in grp:
                    sample_shapes["obj_points_3d"] = tuple(grp["obj_points_3d"].shape)
                sampled = True

        report["n_frame_groups"]        = n_frames
        report["n_has_hands"]           = n_has_hands
        report["n_has_vertices"]        = n_has_vertices
        report["n_has_cam_t"]           = n_has_cam_t
        report["n_has_global_orient"]   = n_has_global_orient
        report["n_has_hand_pose"]       = n_has_hand_pose
        report["n_has_betas"]           = n_has_betas
        report["n_has_obj_points"]      = n_has_obj_pts
        report["n_obj_mask_valid"]      = n_obj_mask_valid
        report["n_contactopt_done"]     = n_contactopt_done
        report["sample_shapes"]         = sample_shapes

        # ── Readiness flags ──────────────────────────────────────────────

        # Stage 1.1 completeness
        report["has_mano_params"]   = n_has_global_orient > 0 and n_has_hand_pose > 0 and n_has_betas > 0
        report["has_vertices"]      = n_has_vertices > 0
        report["has_cam_t"]         = n_has_cam_t > 0

        # Stage 1.2 completeness
        report["has_object_points"] = n_has_obj_pts > 0
        report["object_name_found"] = report["object_name"] != "<missing>"

        # MANO pose format — ContactOpt needs 15 PCA components, not 45 axis-angle joints
        # HaMeR outputs hand_pose as (N_hands, 15, 3, 3) rotation matrices or (N_hands, 45)
        # We detect which format is present.
        hp_shape = sample_shapes.get("hand_pose")
        go_shape = sample_shapes.get("global_orient")
        report["hand_pose_shape"]    = str(hp_shape)
        report["global_orient_shape"]= str(go_shape)

        if hp_shape is not None:
            # HaMeR rotation-matrix format: (N, 15, 3, 3) → 15 joints as rot mats
            # HaMeR axis-angle format:       (N, 45)        → 15 joints * 3
            # ContactOpt PCA format:         scalar 15 components
            if len(hp_shape) == 4 and hp_shape[-2:] == (3, 3):
                report["mano_pose_format"]      = "rotation_matrix (N, 15, 3, 3)"
                report["needs_pca_conversion"]  = True
                report["conversion_note"]       = (
                    "HaMeR gave rotation matrices. Must convert: "
                    "rot_mat → axis-angle (matrix_to_axis_angle) → PCA (fit_pca_to_axang)"
                )
            elif len(hp_shape) == 2 and hp_shape[-1] == 45:
                report["mano_pose_format"]      = "axis_angle (N, 45)"
                report["needs_pca_conversion"]  = True
                report["conversion_note"]       = (
                    "HaMeR gave axis-angle. Must convert: "
                    "axis-angle → PCA (fit_pca_to_axang from ContactOpt utils)"
                )
            elif len(hp_shape) == 2 and hp_shape[-1] == 15:
                report["mano_pose_format"]      = "PCA (N, 15) — already correct!"
                report["needs_pca_conversion"]  = False
                report["conversion_note"]       = "No conversion needed."
            else:
                report["mano_pose_format"]      = f"unknown ({hp_shape})"
                report["needs_pca_conversion"]  = "unknown"
                report["conversion_note"]       = "Inspect manually."
        else:
            report["mano_pose_format"]      = "N/A — no hand_pose found"
            report["needs_pca_conversion"]  = "N/A"
            report["conversion_note"]       = "Stage 1.1 data missing."

        # Object point cloud vs mesh — ContactOpt needs a MESH, not a point cloud
        report["has_obj_mesh_note"] = (
            "⚠️  Stage 1.2 stores POINT CLOUDS (obj_points_3d). "
            "ContactOpt needs a TRIANGLE MESH (OBJ/PLY). "
            "You need either: (a) the Meshy-reconstructed mesh from Stage 1.2, "
            "or (b) Poisson surface reconstruction from the point cloud."
        )

        # Overlap: frames that have BOTH hand data AND object points (needed for ContactOpt)
        frames_overlap = 0
        for key in frame_keys:
            grp = hf[key]
            if int(grp.attrs.get("n_hands", 0)) > 0 and "obj_points_3d" in grp:
                frames_overlap += 1
        report["frames_with_both_hand_and_obj"] = frames_overlap

        # Sample a frame and check obj_points stats
        obj_stats = {}
        for key in frame_keys:
            grp = hf[key]
            if "obj_points_3d" in grp and int(grp.attrs.get("n_hands", 0)) > 0:
                pts = grp["obj_points_3d"][:]
                obj_stats["n_points"]    = pts.shape[0]
                obj_stats["dtype"]       = str(pts.dtype)
                obj_stats["x_range"]     = [round(float(pts[:,0].min()), 4), round(float(pts[:,0].max()), 4)]
                obj_stats["y_range"]     = [round(float(pts[:,1].min()), 4), round(float(pts[:,1].max()), 4)]
                obj_stats["z_range"]     = [round(float(pts[:,2].min()), 4), round(float(pts[:,2].max()), 4)]
                # Check cam_t z for comparison (are object points near the hand?)
                if "cam_t" in grp:
                    hand_z = float(np.median(grp["cam_t"][:, 2]))
                    obj_stats["hand_cam_t_z_median"] = round(hand_z, 4)
                    obj_stats["obj_z_vs_hand_ok"] = (
                        abs(float(np.median(pts[:,2])) - hand_z) < 0.5
                    )
                break
        report["obj_point_sample_stats"] = obj_stats

        # ── ContactOpt repo check ────────────────────────────────────────
        # (look for it relative to script or common locations)
        contactopt_root = _find_contactopt()
        report["contactopt_repo_found"]     = contactopt_root is not None
        report["contactopt_root"]           = contactopt_root or "NOT FOUND"
        if contactopt_root:
            demo_path = os.path.join(contactopt_root, "contactopt", "run_user_demo.py")
            util_path = os.path.join(contactopt_root, "contactopt", "util.py")
            report["contactopt_demo_exists"]     = os.path.isfile(demo_path)
            report["contactopt_util_exists"]     = os.path.isfile(util_path)
            # Check if fit_pca_to_axang is present
            if os.path.isfile(util_path):
                with open(util_path) as f:
                    content = f.read()
                report["fit_pca_to_axang_exists"] = "fit_pca_to_axang" in content
            else:
                report["fit_pca_to_axang_exists"] = False
        else:
            report["contactopt_demo_exists"]    = False
            report["contactopt_util_exists"]    = False
            report["fit_pca_to_axang_exists"]   = False

    return report


def _find_contactopt():
    candidates = [
        os.path.join(os.getcwd(), "third_party", "contactopt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "contactopt"),
        os.path.expanduser("~/contactopt"),
        "/opt/contactopt",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ──────────────────────────────────────────────────────────────────────────────

def print_report(report: dict):
    print(SEP)
    print(f"FILE: {report['path']}")
    print(f"  Size: {report['file_size_mb']} MB")
    print(f"  Source video:    {report['source_video']}")
    print(f"  Object name:     {report['object_name']}")
    print(f"  Total frames:    {report['total_frames']}")
    print(f"  Frames w/ hands: {report['frames_with_hands']}")
    print()

    print("── Stage 1.1 (HaMeR) Completeness ─────────────────────────────")
    print(f"  Vertices stored:      {fmt(report['has_vertices'])}  ({report['n_has_vertices']} frames)")
    print(f"  cam_t stored:         {fmt(report['has_cam_t'])}  ({report['n_has_cam_t']} frames)")
    print(f"  global_orient stored: {fmt(report.get('n_has_global_orient', 0) > 0)}  ({report.get('n_has_global_orient', 0)} frames)")
    print(f"  hand_pose stored:     {fmt(report.get('n_has_hand_pose', 0) > 0)}  ({report.get('n_has_hand_pose', 0)} frames)")
    print(f"  betas stored:         {fmt(report.get('n_has_betas', 0) > 0)}  ({report.get('n_has_betas', 0)} frames)")
    print()
    print(f"  Sample shapes:")
    for k, v in report.get("sample_shapes", {}).items():
        print(f"    {k:20s}: {v}")
    print()
    print(f"  MANO pose format:     {report.get('mano_pose_format', 'N/A')}")
    print(f"  Needs PCA conversion: {fmt(report.get('needs_pca_conversion', 'N/A'))}")
    print(f"  Conversion note:      {report.get('conversion_note', '')}")
    print()

    print("── Stage 1.2 (Object Reconstruction) Completeness ──────────────")
    print(f"  Object name attr:     {fmt(report['object_name_found'])}")
    print(f"  obj_points_3d stored: {fmt(report['has_object_points'])}  ({report['n_has_obj_points']} frames)")
    print(f"  obj_mask_valid:       {report['n_obj_mask_valid']} frames")
    print()
    stats = report.get("obj_point_sample_stats", {})
    if stats:
        print(f"  Sample obj point cloud (first valid frame):")
        print(f"    n_points:   {stats.get('n_points', 'N/A')}")
        print(f"    dtype:      {stats.get('dtype', 'N/A')}")
        print(f"    x range:    {stats.get('x_range', 'N/A')}")
        print(f"    y range:    {stats.get('y_range', 'N/A')}")
        print(f"    z range:    {stats.get('z_range', 'N/A')}")
        if "hand_cam_t_z_median" in stats:
            print(f"    hand z:     {stats['hand_cam_t_z_median']}  (obj z should be close)")
            print(f"    z aligned:  {fmt(stats.get('obj_z_vs_hand_ok', False))}")
    print()
    print(f"  ⚠️  MESH REQUIREMENT: {report['has_obj_mesh_note']}")
    print()

    print("── ContactOpt Readiness ─────────────────────────────────────────")
    print(f"  Frames with hand+obj overlap: {report['frames_with_both_hand_and_obj']}")
    print(f"  ContactOpt repo found:   {fmt(report['contactopt_repo_found'])}")
    print(f"  ContactOpt root:         {report['contactopt_root']}")
    print(f"  run_user_demo.py exists: {fmt(report.get('contactopt_demo_exists', False))}")
    print(f"  util.py exists:          {fmt(report.get('contactopt_util_exists', False))}")
    print(f"  fit_pca_to_axang exists: {fmt(report.get('fit_pca_to_axang_exists', False))}")
    print(f"  contactopt_done frames:  {report['n_contactopt_done']}")
    print()

    print("── Action Items ─────────────────────────────────────────────────")
    items = []
    if not report["has_vertices"]:
        items.append("❌ Re-run Stage 1.1 — no vertices found")
    if not report.get("n_has_global_orient", 0) > 0:
        items.append("❌ Re-run Stage 1.1 — MANO params (global_orient/hand_pose/betas) missing")
    if not report["has_object_points"]:
        items.append("❌ Re-run Stage 1.2 — no obj_points_3d found")
    if report.get("needs_pca_conversion") is True:
        items.append(f"⚠️  Convert MANO pose to PCA-15 before ContactOpt: {report.get('conversion_note')}")
    if not report["contactopt_repo_found"]:
        items.append("❌ Clone ContactOpt: git clone https://github.com/facebookresearch/ContactOpt third_party/contactopt")
    if not report.get("fit_pca_to_axang_exists", False) and report["contactopt_repo_found"]:
        items.append("⚠️  fit_pca_to_axang not found in util.py — check ContactOpt version")
    if report["frames_with_both_hand_and_obj"] == 0:
        items.append("❌ No frames have both hand data AND object points — overlap required for ContactOpt")
    items.append("⚠️  You have point clouds, not meshes. Need triangle mesh for ContactOpt's DeepContact.")
    items.append("    → Option A: Use Meshy .obj from Stage 1.2 (recommended)")
    items.append("    → Option B: Poisson surface reconstruction from obj_points_3d")

    if not items:
        items.append("✅ All checks passed — ready for ContactOpt")
    for item in items:
        print(f"  {item}")

    print(SEP)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diagnose HDF5 files for ContactOpt readiness")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory containing *_stage1.hdf5 files")
    parser.add_argument("--json", action="store_true",
                        help="Also dump full report as JSON (for pasting)")
    args = parser.parse_args()

    h5_files = sorted(glob.glob(os.path.join(args.output_dir, "*.hdf5")))
    if not h5_files:
        print(f"No .hdf5 files found in '{args.output_dir}'. Check --output_dir.")
        sys.exit(1)

    print(f"\nFound {len(h5_files)} HDF5 file(s) in '{args.output_dir}'")

    all_reports = []
    for h5_path in h5_files:
        try:
            r = analyse_file(h5_path)
            print_report(r)
            all_reports.append(r)
        except Exception as e:
            print(f"ERROR analysing {h5_path}: {e}")
            import traceback
            traceback.print_exc()

    if args.json or len(all_reports) > 0:
        print("\n\n── JSON DUMP (paste this into chat) ─────────────────────────────")
        # Make serialisable
        def make_serial(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return obj

        clean = []
        for r in all_reports:
            clean.append({k: make_serial(v) if not isinstance(v, dict) else
                          {kk: make_serial(vv) for kk, vv in v.items()}
                          for k, v in r.items()})
        print(json.dumps(clean, indent=2))


if __name__ == "__main__":
    main()