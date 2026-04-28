import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter


# --------------------------------------------------
# Load frame data
# --------------------------------------------------
def load_frame(grp):
    n_hands = grp.attrs["n_hands"]

    if n_hands == 0:
        return None, None, None

    vertices = grp["vertices"][:]   # (N,778,3)
    cam_t = grp["cam_t"][:]         # (N,3)
    is_right = grp["is_right"][:]   # (N,)

    return vertices, cam_t, is_right


# --------------------------------------------------
# Save first frame PNG only
# --------------------------------------------------
def save_first_frame_png(hf, out_path):
    grp = hf["frame_000000"]

    vertices, cam_t, _ = load_frame(grp)

    if vertices is None:
        print("First frame has no hands, skipping PNG.")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["r", "b", "g", "y"]

    for i in range(len(vertices)):
        verts = vertices[i] + cam_t[i]

        ax.scatter(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            s=2,
            c=colors[i % len(colors)]
        )

    ax.set_title("Frame 0")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# --------------------------------------------------
# Save MP4 animation
# --------------------------------------------------
def save_mp4(hf, total_frames, out_path):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["r", "b", "g", "y"]

    def update(frame_idx):
        ax.clear()

        frame_key = f"frame_{frame_idx:06d}"
        grp = hf[frame_key]

        vertices, cam_t, _ = load_frame(grp)

        ax.set_title(f"Frame {frame_idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if vertices is None:
            return

        for i in range(len(vertices)):
            verts = vertices[i] + cam_t[i]

            ax.scatter(
                verts[:, 0],
                verts[:, 1],
                verts[:, 2],
                s=2,
                c=colors[i % len(colors)]
            )

    anim = FuncAnimation(fig, update, frames=total_frames, interval=80)

    writer = FFMpegWriter(fps=12, bitrate=1800)
    anim.save(out_path, writer=writer, dpi=150)

    plt.close()


# --------------------------------------------------
# Process one file
# --------------------------------------------------
def process_file(hdf5_path, out_root):
    base = os.path.splitext(os.path.basename(hdf5_path))[0]

    save_dir = os.path.join(out_root, base)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Processing {base}")

    with h5py.File(hdf5_path, "r") as hf:
        total_frames = hf.attrs["total_frames"]

        png_path = os.path.join(save_dir, "first_frame.png")
        mp4_path = os.path.join(save_dir, f"{base}.mp4")

        save_first_frame_png(hf, png_path)
        save_mp4(hf, total_frames, mp4_path)

    print(f"Done -> {save_dir}")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="output")
    parser.add_argument("--out_dir", default="viz_output")
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = all files, otherwise process first N files")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".hdf5")
    ])

    if args.limit > 0:
        files = files[:args.limit]

    print(f"Found {len(files)} files to process")

    for fp in files:
        process_file(fp, args.out_dir)

    print("All done.")


if __name__ == "__main__":
    main()