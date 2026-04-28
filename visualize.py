"""
visualize.py — Stage 1 HDF5 Visualization

Renders 3D scatter plots of HaMeR hand meshes (and optionally object point
clouds) from Stage 1 HDF5 output files.

Key design: each hand gets its own zoomed subplot so finger articulation is
visible. Without per-hand zoom, the ~2-unit Z depth gap between hands causes
the ~0.15-unit finger detail to collapse into blobs.

Modes:
  hamer  — HaMeR mesh vertices only (per-hand zoom subplots)
  moge   — MoGe-2 sparse joints only (per-hand zoom)
  both   — HaMeR + MoGe side-by-side per hand

Outputs per file:
  first_frame.png — static render of frame 0
  <basename>.mp4  — animated render of all frames
"""

import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Optional: LZ4-compressed datasets
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

HAND_COLORS = ["red", "blue", "green", "orange"]
HAND_LABELS = ["Left", "Right", "Hand 2", "Hand 3"]


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def zoom_axes_to_points(ax, pts, pad_factor=1.3):
    """
    Zoom a 3D axis to tightly fit the given points with padding.
    Uses equal aspect ratio so hand shape isn't distorted.
    """
    if pts.size == 0:
        return

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins).max()

    # Ensure minimum span so we don't get a degenerate view
    span = max(span, 0.01)
    radius = span / 2.0 * pad_factor

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def load_hamer_frame(grp):
    """
    Load HaMeR vertices in camera space (vertices + cam_t).

    Returns:
        hands: list of (778, 3) arrays — one per detected hand, in camera space
        is_right: list of bool — handedness per detection
    """
    n_hands = int(grp.attrs.get("n_hands", 0))
    if n_hands == 0:
        return [], []

    vertices = grp["vertices"][:]   # (N, 778, 3)
    cam_t = grp["cam_t"][:]         # (N, 3)
    is_right_arr = grp["is_right"][:]   # (N,)

    hands = []
    is_right = []
    for i in range(n_hands):
        verts_cam = vertices[i] + cam_t[i]  # (778, 3) in camera space
        hands.append(verts_cam)
        is_right.append(bool(is_right_arr[i] > 0.5))

    return hands, is_right


def load_moge_frame(grp):
    """Load MoGe-2 sparse joints if present."""
    if "moge_joints_3d" not in grp:
        return [], None

    joints = grp["moge_joints_3d"][:].astype(np.float32)   # (N, 21, 3)
    valid = grp["moge_joint_valid"][:] if "moge_joint_valid" in grp else None
    return [joints[i] for i in range(len(joints))], valid


def load_object_frame(grp):
    """Load object point cloud if present (from run_stage1_objects.py)."""
    if "obj_points_3d" not in grp:
        return None
    pts = grp["obj_points_3d"][:].astype(np.float32)   # (N, 3)
    if not grp.attrs.get("obj_mask_valid", True) or pts.shape[0] == 0:
        return None
    return pts


def plot_single_hand(ax, pts, color, label, point_size=2, alpha=0.6):
    """Plot a single hand's point cloud on a 3D axis with per-hand zoom."""
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=point_size, alpha=alpha, c=color
    )
    zoom_axes_to_points(ax, pts)
    ax.set_title(label, fontsize=10, fontweight="bold", color=color)
    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)
    ax.tick_params(labelsize=6)


def plot_moge_hand(ax, joints, valid, color, label):
    """Plot MoGe joints for a single hand."""
    if valid is not None:
        mask = valid
        # Valid joints
        ax.scatter(
            joints[mask, 0], joints[mask, 1], joints[mask, 2],
            s=40, marker="o", c=color, alpha=0.8
        )
        # Invalid joints
        if (~mask).any():
            ax.scatter(
                joints[~mask, 0], joints[~mask, 1], joints[~mask, 2],
                s=20, marker="x", c="gray", alpha=0.5
            )
    else:
        ax.scatter(
            joints[:, 0], joints[:, 1], joints[:, 2],
            s=40, marker="o", c=color, alpha=0.8
        )

    zoom_axes_to_points(ax, joints)
    valid_count = int(valid.sum()) if valid is not None else len(joints)
    ax.set_title(f"{label} ({valid_count}/{len(joints)} valid)", fontsize=9, color=color)
    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)
    ax.tick_params(labelsize=6)


# --------------------------------------------------
# Render a single frame to axes
# --------------------------------------------------
def render_frame_to_fig(fig, grp, mode, frame_idx):
    """
    Render one frame's data onto the given figure.
    Creates per-hand subplots so each hand is zoomed independently.
    """
    fig.clear()

    hands_hamer, is_right = load_hamer_frame(grp)
    hands_moge, moge_valid = load_moge_frame(grp)
    obj_pts = load_object_frame(grp)

    n_hands = max(len(hands_hamer), len(hands_moge))

    if n_hands == 0:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"Frame {frame_idx}\nNo hands detected",
                ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return

    # Determine subplot layout
    if mode == "both":
        # 2 rows × n_hands cols: row 0 = HaMeR, row 1 = MoGe
        n_cols = max(n_hands, 1)
        n_rows = 2
    else:
        # 1 row × n_hands cols (+ 1 for object if present)
        n_cols = n_hands + (1 if obj_pts is not None else 0)
        n_rows = 1

    subplot_idx = 1

    if mode in ("hamer", "both"):
        for i, pts in enumerate(hands_hamer):
            ax = fig.add_subplot(n_rows, n_cols, subplot_idx, projection="3d")
            side = "Right" if (i < len(is_right) and is_right[i]) else "Left"
            color = HAND_COLORS[i % len(HAND_COLORS)]

            plot_single_hand(ax, pts, color, f"HaMeR {side} (frame {frame_idx})")

            # Overlay object if in same coordinate space (hamer mode only)
            if mode == "hamer" and obj_pts is not None and i == 0:
                ax.scatter(
                    obj_pts[:, 0], obj_pts[:, 1], obj_pts[:, 2],
                    s=3, alpha=0.3, c="green", marker="^"
                )
                # Re-zoom to include object
                combined = np.concatenate([pts, obj_pts], axis=0)
                zoom_axes_to_points(ax, combined)

            subplot_idx += 1

        # Extra subplot for object alone (if hamer mode and object exists)
        if mode == "hamer" and obj_pts is not None:
            ax = fig.add_subplot(n_rows, n_cols, subplot_idx, projection="3d")
            ax.scatter(
                obj_pts[:, 0], obj_pts[:, 1], obj_pts[:, 2],
                s=3, alpha=0.6, c="green", marker="^"
            )
            zoom_axes_to_points(ax, obj_pts)
            ax.set_title(f"Object ({len(obj_pts)} pts)", fontsize=9, color="green")
            ax.set_xlabel("X", fontsize=7)
            ax.set_ylabel("Y", fontsize=7)
            ax.set_zlabel("Z", fontsize=7)
            ax.tick_params(labelsize=6)
            subplot_idx += 1

    if mode in ("moge", "both"):
        if mode == "both":
            # Start second row
            subplot_idx = n_cols + 1

        for i in range(len(hands_moge)):
            ax = fig.add_subplot(n_rows, n_cols, subplot_idx, projection="3d")
            valid_i = moge_valid[i] if moge_valid is not None else None
            side = "Right" if (i < len(is_right) and is_right[i]) else "Left"
            color = HAND_COLORS[i % len(HAND_COLORS)]

            plot_moge_hand(ax, hands_moge[i], valid_i, color, f"MoGe {side}")
            subplot_idx += 1

    fig.suptitle(f"Frame {frame_idx}", fontsize=12, fontweight="bold")
    fig.tight_layout()


# --------------------------------------------------
# Save first frame PNG
# --------------------------------------------------
def save_first_frame_png(hf, out_path, mode):
    grp = hf["frame_000000"]

    # Figure size adapts to content
    hands_hamer, _ = load_hamer_frame(grp)
    n_hands = max(len(hands_hamer), 1)
    width = 7 * n_hands
    height = 7 if mode != "both" else 12

    fig = plt.figure(figsize=(width, height))
    render_frame_to_fig(fig, grp, mode, 0)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


# --------------------------------------------------
# Save MP4 animation
# --------------------------------------------------
def save_mp4(hf, total_frames, out_path, mode):
    fig = plt.figure(figsize=(14, 7))

    def update(frame_idx):
        frame_key = f"frame_{frame_idx:06d}"
        grp = hf[frame_key]
        render_frame_to_fig(fig, grp, mode, frame_idx)

    anim = FuncAnimation(fig, update, frames=total_frames, interval=80)
    writer = FFMpegWriter(fps=12, bitrate=2400)
    anim.save(out_path, writer=writer, dpi=120)
    plt.close()


# --------------------------------------------------
# Process one file
# --------------------------------------------------
def process_file(hdf5_path, out_root, mode):
    base = os.path.splitext(os.path.basename(hdf5_path))[0]

    save_dir = os.path.join(out_root, base)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Processing {base}")

    with h5py.File(hdf5_path, "r") as hf:
        total_frames = int(hf.attrs["total_frames"])

        png_path = os.path.join(save_dir, "first_frame.png")
        mp4_path = os.path.join(save_dir, f"{base}.mp4")

        save_first_frame_png(hf, png_path, mode)
        save_mp4(hf, total_frames, mp4_path, mode)

    print(f"Done -> {save_dir}")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize Stage 1 HDF5 output (HaMeR hands + optional MoGe/objects)"
    )
    parser.add_argument("--input_dir", default="output")
    parser.add_argument("--out_dir", default="viz_output")
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = all files, otherwise process first N files")
    parser.add_argument(
        "--mode",
        default="hamer",
        choices=["hamer", "moge", "both"],
        help="What to visualize: hamer (default), moge, or both"
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".hdf5")
    ])

    if args.limit > 0:
        files = files[:args.limit]

    print(f"Found {len(files)} files to process (mode={args.mode})")

    for fp in files:
        process_file(fp, args.out_dir, args.mode)

    print("All done.")


if __name__ == "__main__":
    main()