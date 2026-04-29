import os
import sys
import glob
import h5py
import numpy as np
import subprocess
from loguru import logger
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTACTOPT_ROOT = os.path.join(BASE_DIR, "third_party", "contactopt")
sys.path.insert(0, CONTACTOPT_ROOT)


def ensure_contactopt():
    if not os.path.isdir(CONTACTOPT_ROOT):
        logger.error("ContactOpt repo not found.")
        sys.exit(1)

    run_file = os.path.join(CONTACTOPT_ROOT, "contactopt", "run_user_demo.py")
    if not os.path.isfile(run_file):
        logger.error("run_user_demo.py not found in ContactOpt.")
        sys.exit(1)


def export_temp_inputs(frame_group, tmp_dir):
    """
    Save temporary files ContactOpt demo script can consume.
    You may later customize this depending on exact mesh format.
    """

    obj_pts = frame_group["obj_points_3d"][:]

    np.save(os.path.join(tmp_dir, "object_points.npy"), obj_pts)

    mano = {
        "global_orient": frame_group["global_orient"][:],
        "hand_pose": frame_group["hand_pose"][:],
        "betas": frame_group["betas"][:],
        "cam_t": frame_group["cam_t"][:],
    }

    np.savez(os.path.join(tmp_dir, "mano_input.npz"), **mano)


def run_contactopt(tmp_dir):
    cmd = [
        sys.executable,
        "-m",
        "contactopt.run_user_demo"
    ]

    result = subprocess.run(
        cmd,
        cwd=CONTACTOPT_ROOT,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(result.stderr)
        return False

    logger.info(result.stdout)
    return True


def process_h5(h5_path):
    logger.info(f"Processing {h5_path}")

    with h5py.File(h5_path, "r+") as hf:
        total_frames = int(hf.attrs.get("total_frames", 0))

        for fidx in tqdm(range(total_frames), desc="ContactOpt"):
            key = f"frame_{fidx:06d}"

            if key not in hf:
                continue

            grp = hf[key]

            if "obj_points_3d" not in grp:
                continue

            tmp_dir = os.path.join(BASE_DIR, "_tmp_contactopt")
            os.makedirs(tmp_dir, exist_ok=True)

            export_temp_inputs(grp, tmp_dir)

            ok = run_contactopt(tmp_dir)

            if not ok:
                continue

            # Placeholder:
            # later load optimized verts from ContactOpt outputs
            # and save back into HDF5

            grp.attrs["contactopt_done"] = True


def main():
    logger.add("logs/stage1_contactopt.log", rotation="100 MB")

    ensure_contactopt()

    files = sorted(glob.glob("output/*.hdf5"))

    if not files:
        logger.error("No H5 files found in output/")
        return

    for f in files:
        process_h5(f)


if __name__ == "__main__":
    main()