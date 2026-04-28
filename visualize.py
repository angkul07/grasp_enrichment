import os
import argparse
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def set_equal_axes(ax, pts):
    if pts.size == 0:
        return

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0 + 1e-6

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def load_hamer_frame(grp):
    n_hands = int(grp.attrs["n_hands"])

    if n_hands == 0:
        return None, None

    vertices = grp["vertices"][:]   # (N,778,3)
    cam_t = grp["cam_t"][:]         # (N,3)

    verts_world = vertices + cam_t[:, None, :]
    return verts_world, n_hands


def load_moge_frame(grp):
    if "moge_joints_3d" not in grp:
        return None, None

    joints = grp["moge_joints_3d"][:].astype(np.float32)   # (N,J,3)
    valid = grp["moge_joint_valid"][:] if "moge_joint_valid" in grp else None

    return joints, valid


# --------------------------------------------------
# Save first frame PNG
# --------------------------------------------------
def save_first_frame_png(hf, out_path, mode):
    grp = hf["frame_000000"]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["r", "b", "g", "y"]

    all_pts = []

    if mode in ["hamer", "both"]:
        verts_world, n_hands = load_hamer_frame(grp)
        if verts_world is not None:
            for i in range(n_hands):
                pts = verts_world[i]
                all_pts.append(pts)
                ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    s=1,
                    alpha=0.25,
                    c=colors[i % len(colors)]
                )

    if mode in ["moge", "both"]:
        joints, valid = load_moge_frame(grp)
        if joints is not None:
            for i in range(len(joints)):
                pts = joints[i]
                all_pts.append(pts)

                if valid is not None:
                    mask = valid[i]
                else:
                    mask = np.ones(len(pts), dtype=bool)

                ax.scatter(
                    pts[mask, 0], pts[mask, 1], pts[mask, 2],
                    s=28,
                    marker="o",
                    c=colors[i % len(colors)]
                )

    if all_pts:
        set_equal_axes(ax, np.concatenate(all_pts, axis=0))

    ax.set_title("Frame 0")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# --------------------------------------------------
# Save MP4
# --------------------------------------------------
def save_mp4(hf, total_frames, out_path, mode):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["r", "b", "g", "y"]

    def update(frame_idx):
        ax.clear()

        frame_key = f"frame_{frame_idx:06d}"
        grp = hf[frame_key]

        all_pts = []

        if mode in ["hamer", "both"]:
            verts_world, n_hands = load_hamer_frame(grp)
            if verts_world is not None:
                for i in range(n_hands):
                    pts = verts_world[i]
                    all_pts.append(pts)

                    ax.scatter(
                        pts[:, 0], pts[:, 1], pts[:, 2],
                        s=1,
                        alpha=0.25,
                        c=colors[i % len(colors)]
                    )

        if mode in ["moge", "both"]:
            joints, valid = load_moge_frame(grp)
            if joints is not None:
                for i in range(len(joints)):
                    pts = joints[i]
                    all_pts.append(pts)

                    if valid is not None:
                        mask = valid[i]
                    else:
                        mask = np.ones(len(pts), dtype=bool)

                    ax.scatter(
                        pts[mask, 0], pts[mask, 1], pts[mask, 2],
                        s=28,
                        marker="o",
                        c=colors[i % len(colors)]
                    )

        if all_pts:
            set_equal_axes(ax, np.concatenate(all_pts, axis=0))

        ax.set_title(f"Frame {frame_idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    anim = FuncAnimation(fig, update, frames=total_frames, interval=80)

    writer = FFMpegWriter(fps=12, bitrate=1800)
    anim.save(out_path, writer=writer, dpi=150)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="output")
    parser.add_argument("--out_dir", default="viz_output")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--mode",
        default="both",
        choices=["hamer", "moge", "both"],
        help="What to visualize"
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

    print(f"Found {len(files)} files")

    for fp in files:
        process_file(fp, args.out_dir, args.mode)

    print("All done.")


if __name__ == "__main__":
    main()