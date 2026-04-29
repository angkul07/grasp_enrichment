"""
visualize_obj.py — Hand + Object Visualization

Visualizes HaMeR hand meshes and object point clouds in HaMeR camera space.
Both are stored in the same coordinate frame, so they should align directly.

Key design:
  - Subplot 1: Combined overview (hands + object) — useful for seeing spatial
    relationship, but individual details may be small
  - Subplots 2+: Per-hand zoomed view with object overlay — shows finger
    articulation clearly alongside nearby object points

Usage:
  # Single frame PNG
  python visualize_obj.py --file output/0_stage1.hdf5 --frame 0

  # Animated MP4
  python visualize_obj.py --file output/0_stage1.hdf5 --animate

  # Batch all files
  python visualize_obj.py --batch --input_dir output --out_dir viz_output
"""

import os
import argparse
import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def zoom_axes(ax, pts, pad=1.3):
    """Set equal-aspect 3D axes tightly around the given points."""
    if pts.size == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    span = max((maxs - mins).max(), 0.01)
    r = span / 2.0 * pad
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)


def load_hands(grp):
    """Load HaMeR hand vertices in camera space. Returns list of (778,3) arrays."""
    n = int(grp.attrs.get("n_hands", 0))
    if n == 0:
        return [], []
    verts = grp["vertices"][:]   # (N, 778, 3)
    cam_t = grp["cam_t"][:]      # (N, 3)
    is_right = grp["is_right"][:]
    hands = [verts[i] + cam_t[i] for i in range(n)]
    sides = ["Right" if is_right[i] > 0.5 else "Left" for i in range(n)]
    return hands, sides


def load_object(grp):
    """Load object point cloud if available."""
    if "obj_points_3d" not in grp:
        return None
    valid = grp.attrs.get("obj_mask_valid", True)
    if not valid:
        return None
    pts = grp["obj_points_3d"][:].astype(np.float32)
    if pts.shape[0] == 0:
        return None
    return pts


COLORS = ["red", "blue", "green", "orange"]


# --------------------------------------------------
# Render a single frame
# --------------------------------------------------
def render_frame(fig, grp, frame_idx, object_name="", is_anim=False):
    """Render one frame with overview + per-hand zoomed subplots."""
    fig.clear()

    hands, sides = load_hands(grp)
    obj_pts = load_object(grp)
    
    # SPEED OPTIMIZATION: Subsample points drastically if making a video.
    # Matplotlib 3D scatter is very slow with thousands of points per frame.
    if is_anim:
        hands = [h[::2] for h in hands]  # Reduce hand points by half
        if obj_pts is not None and len(obj_pts) > 500:
            # Deterministic subsample to ~500 points to prevent flickering
            step = len(obj_pts) // 500 + 1
            obj_pts = obj_pts[::step]

    n_hands = len(hands)

    if n_hands == 0 and obj_pts is None:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"Frame {frame_idx}\nNo data",
                ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return

    # Layout: [overview] [hand0_zoom] [hand1_zoom] ...
    n_cols = 1 + n_hands
    axes =[]

    # -- Overview subplot --
    ax_ov = fig.add_subplot(1, n_cols, 1, projection="3d")
    all_pts =[]

    for i, hand in enumerate(hands):
        ax_ov.scatter(hand[:, 0], hand[:, 1], hand[:, 2],
                      s=0.5 if not is_anim else 1, alpha=0.3, c=COLORS[i % len(COLORS)])
        all_pts.append(hand)

    if obj_pts is not None:
        ax_ov.scatter(obj_pts[:, 0], obj_pts[:, 1], obj_pts[:, 2],
                      s=2 if not is_anim else 4, alpha=0.6, c="green", marker="^")
        all_pts.append(obj_pts)

    if all_pts:
        zoom_axes(ax_ov, np.concatenate(all_pts))

    ax_ov.set_title("Overview", fontsize=9, fontweight="bold")
    ax_ov.set_xlabel("X", fontsize=7)
    ax_ov.set_ylabel("Y", fontsize=7)
    ax_ov.set_zlabel("Z", fontsize=7)
    ax_ov.tick_params(labelsize=5)

    # -- Per-hand zoomed subplots --
    for i, hand in enumerate(hands):
        ax = fig.add_subplot(1, n_cols, 2 + i, projection="3d")
        color = COLORS[i % len(COLORS)]

        # Hand vertices
        ax.scatter(hand[:, 0], hand[:, 1], hand[:, 2],
                   s=2 if not is_anim else 4, alpha=0.6, c=color, label=f"{sides[i]} hand")

        # Object overlay (only points near this hand's Z range for clarity)
        if obj_pts is not None:
            hand_z_min = hand[:, 2].min()
            hand_z_max = hand[:, 2].max()
            z_margin = (hand_z_max - hand_z_min) * 5  # generous margin

            nearby = obj_pts[
                (obj_pts[:, 2] > hand_z_min - z_margin) &
                (obj_pts[:, 2] < hand_z_max + z_margin)
            ]

            if len(nearby) > 0:
                ax.scatter(nearby[:, 0], nearby[:, 1], nearby[:, 2],
                           s=3 if not is_anim else 6, alpha=0.5, c="green", marker="^",
                           label=f"Object ({len(nearby)} pts)")

                # Zoom to include both hand and nearby object
                combined = np.concatenate([hand, nearby])
                zoom_axes(ax, combined)
            else:
                zoom_axes(ax, hand)
        else:
            zoom_axes(ax, hand)

        ax.set_title(f"{sides[i]} Hand", fontsize=9,
                     fontweight="bold", color=color)
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=5)
        ax.legend(fontsize=6, loc="upper left")

    title = f"Frame {frame_idx}"
    if object_name:
        title += f"  |  Object: {object_name}"
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()


# --------------------------------------------------
# Single frame to PNG
# --------------------------------------------------
def save_frame_png(hf, frame_idx, out_path):
    grp_name = f"frame_{frame_idx:06d}"
    if grp_name not in hf:
        print(f"Frame {grp_name} not found!")
        return

    object_name = str(hf.attrs.get("object_name", ""))

    # Width depends on number of hands
    n_hands = int(hf[grp_name].attrs.get("n_hands", 0))
    width = 7 * (1 + max(n_hands, 1))

    fig = plt.figure(figsize=(width, 7))
    render_frame(fig, hf[grp_name], frame_idx, object_name, is_anim=False)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# --------------------------------------------------
# Animate to MP4
# --------------------------------------------------
def save_animation(hf, out_path):
    total = int(hf.attrs.get("total_frames", 0))
    object_name = str(hf.attrs.get("object_name", ""))

    fig = plt.figure(figsize=(21, 7))

    def update(fidx):
        grp_name = f"frame_{fidx:06d}"
        if grp_name in hf:
            # Pass is_anim=True to trigger the speed optimizations
            render_frame(fig, hf[grp_name], fidx, object_name, is_anim=True)

    anim = FuncAnimation(fig, update, frames=total, interval=100)
    writer = FFMpegWriter(fps=10, bitrate=2400)
    anim.save(out_path, writer=writer, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")


# --------------------------------------------------
# Batch processing
# --------------------------------------------------
def process_file(hdf5_path, out_dir):
    base = os.path.splitext(os.path.basename(hdf5_path))[0]
    save_dir = os.path.join(out_dir, base)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Processing {base}...")

    with h5py.File(hdf5_path, "r") as hf:
        png_path = os.path.join(save_dir, "hand_object_frame0.png")
        mp4_path = os.path.join(save_dir, f"{base}_hand_object.mp4")

        save_frame_png(hf, 0, png_path)
        save_animation(hf, mp4_path)

    print(f"Done -> {save_dir}")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize hand meshes + object point clouds from Stage 1 HDF5"
    )
    parser.add_argument("--file", type=str, help="Single HDF5 file to visualize")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (for single mode)")
    parser.add_argument("--animate", action="store_true", help="Generate MP4 animation")
    parser.add_argument("--batch", action="store_true", help="Process all files in input_dir")
    parser.add_argument("--input_dir", default="output", help="Directory with *_stage1.hdf5 files")
    parser.add_argument("--out_dir", default="viz_output", help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Max files for batch mode (0=all)")

    args = parser.parse_args()

    if args.batch:
        os.makedirs(args.out_dir, exist_ok=True)
        files = sorted([
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".hdf5")
        ])
        if args.limit > 0:
            files = files[:args.limit]
        print(f"Batch processing {len(files)} files...")
        for fp in files:
            process_file(fp, args.out_dir)
        print("All done.")

    elif args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return

        # FIXED: Ensure single file mode saves correctly inside out_dir
        base = os.path.splitext(os.path.basename(args.file))[0]
        save_dir = os.path.join(args.out_dir, base)
        os.makedirs(save_dir, exist_ok=True)

        with h5py.File(args.file, "r") as hf:
            if args.animate:
                out = os.path.join(save_dir, f"{base}_hand_object.mp4")
                save_animation(hf, out)
            else:
                out = os.path.join(save_dir, f"{base}_frame{args.frame}.png")
                save_frame_png(hf, args.frame, out)
        print(f"Done -> {save_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()