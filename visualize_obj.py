#!/usr/bin/env python3
"""
visualize_obj.py

Visualize:
1. HaMeR hand mesh vertices
2. Object point cloud reconstructed in SAME HaMeR camera space

Reads *_stage1.hdf5 outputs from your Stage 1 / Stage 1.2 pipeline.

Usage:
python visualize_obj.py \
    --file output/0_stage1.hdf5 \
    --frame 0

python visualize_obj.py \
    --file output/0_stage1.hdf5 \
    --animate

Why this matters:
Your object points are already backprojected into HaMeR camera space:
(X,Y,Z) using scaled_focal + aligned depth.

So hand vertices + object cloud should now align directly.
:contentReference[oaicite:0]{index=0}
"""

import argparse
import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def set_equal_axes(ax, pts):
    """
    Equal 3D aspect ratio.
    """
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0 + 1e-6

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def world_hand_vertices(vertices, cam_t, is_right):
    """
    Convert stored HaMeR local mesh -> camera/world coordinates.

    vertices: (N,778,3)
    cam_t:    (N,3)
    is_right: (N,)
    """
    out = []

    for i in range(len(vertices)):
        v = vertices[i].copy()

        # Undo right-hand mirror used in your pipeline
        if int(is_right[i]) == 1:
            v[:, 0] *= -1.0

        v = v + cam_t[i][None, :]
        out.append(v)

    return out


def collect_all_points(hand_meshes, obj_pts):
    pts = []

    for h in hand_meshes:
        if len(h) > 0:
            pts.append(h)

    if obj_pts is not None and len(obj_pts) > 0:
        pts.append(obj_pts)

    if len(pts) == 0:
        return np.zeros((1, 3))

    return np.concatenate(pts, axis=0)


# ---------------------------------------------------------
# Single frame render
# ---------------------------------------------------------
def render_frame(hf, frame_idx):
    grp_name = f"frame_{frame_idx:06d}"

    if grp_name not in hf:
        print("Frame not found:", grp_name)
        return

    grp = hf[grp_name]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    hand_meshes = []

    # ----------------------------
    # Hand mesh
    # ----------------------------
    if "vertices" in grp:
        vertices = grp["vertices"][:]
        cam_t = grp["cam_t"][:]
        is_right = grp["is_right"][:]

        hand_meshes = world_hand_vertices(vertices, cam_t, is_right)

        for i, hand in enumerate(hand_meshes):
            side = "Right" if int(is_right[i]) == 1 else "Left"

            ax.scatter(
                hand[:, 0],
                hand[:, 1],
                hand[:, 2],
                s=1,
                alpha=0.45,
                label=f"{side} Hand",
            )

    # ----------------------------
    # Object point cloud
    # ----------------------------
    obj_pts = None

    if "obj_points_3d" in grp:
        obj_pts = grp["obj_points_3d"][:].astype(np.float32)

        if len(obj_pts) > 0:
            ax.scatter(
                obj_pts[:, 0],
                obj_pts[:, 1],
                obj_pts[:, 2],
                s=6,
                alpha=0.9,
                label="Object",
            )

    # ----------------------------
    # Axis scaling
    # ----------------------------
    all_pts = collect_all_points(hand_meshes, obj_pts)
    set_equal_axes(ax, all_pts)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    title = f"{grp_name}"
    if "object_name" in hf.attrs:
        title += f" | Object: {hf.attrs['object_name']}"

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig("frame0.png", dpi=220, bbox_inches="tight")
    print("saved frame0.png")


# ---------------------------------------------------------
# Animation
# ---------------------------------------------------------
def animate_file(hf):
    frame_keys = sorted(
        [k for k in hf.keys() if k.startswith("frame_")]
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(i):
        ax.cla()

        grp = hf[frame_keys[i]]

        hand_meshes = []

        if "vertices" in grp:
            vertices = grp["vertices"][:]
            cam_t = grp["cam_t"][:]
            is_right = grp["is_right"][:]

            hand_meshes = world_hand_vertices(vertices, cam_t, is_right)

            for j, hand in enumerate(hand_meshes):
                side = "Right" if int(is_right[j]) == 1 else "Left"

                ax.scatter(
                    hand[:, 0], hand[:, 1], hand[:, 2],
                    s=1, alpha=0.45, label=side
                )

        obj_pts = None
        if "obj_points_3d" in grp:
            obj_pts = grp["obj_points_3d"][:].astype(np.float32)

            if len(obj_pts) > 0:
                ax.scatter(
                    obj_pts[:, 0],
                    obj_pts[:, 1],
                    obj_pts[:, 2],
                    s=6,
                    alpha=0.9,
                    label="Object"
                )

        all_pts = collect_all_points(hand_meshes, obj_pts)
        set_equal_axes(ax, all_pts)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(frame_keys[i])
        ax.legend(loc="upper right")

        ani = FuncAnimation(
            fig,
            update,
            frames=len(frame_keys),
            interval=120,
            repeat=True
        )

        plt.tight_layout()

        ani.save("animation_output.gif", writer="pillow", fps=8)
        print("saved animation_output.gif")

        plt.close(fig)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        required=True,
        help="Path to *_stage1.hdf5"
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index"
    )

    parser.add_argument(
        "--animate",
        action="store_true"
    )

    args = parser.parse_args()

    with h5py.File(args.file, "r") as hf:
        if args.animate:
            animate_file(hf)
        else:
            render_frame(hf, args.frame)


if __name__ == "__main__":
    main()