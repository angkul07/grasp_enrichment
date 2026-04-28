# Project Documentation

This file documents the progress, stages, steps, decisions, open questions, results, and suggestions throughout the project lifecycle.

## Overview
- **Objective:** Develop a robust, universal pipeline converting human egocentric videos (RGB only) to executable robot policies.
- **Current Dataset:** EgoDex `basic_pick_place`.
- **References:**
  - [VideoManip](file:///home/angkul/Desktop/robotics/grasp_enrichment/VideoManip.pdf)
  - [HaMeR](file:///home/angkul/Desktop/robotics/grasp_enrichment/hamer.pdf)
  - [MoGe-2](file:///home/angkul/Desktop/robotics/grasp_enrichment/moge-2.pdf)
  - Universal 3D HOI Plan Document.

---

## Stage 0: Environment Setup & Data Preparation (Current)

### Steps Taken
- Created virtual environment using `uv` (.venv).
- Created `rules.md`, `plan.md`, `docs.md`.
- Created `third_party` directory.
- Cloned `HaMeR` and `MoGe-2` into `third_party`.

### Key Decisions
- Adopted strict structure with dedicated files for rules, planning, and documentation.
- Using `uv` for lightning-fast and reproducible environment management.

### Open Questions
- What specific versions of PyTorch and CUDA will be necessary for `HaMeR` and `MoGe-2` on this machine?
- What is the internal file structure of the `basic_pick_place` dataset?

### Results
- Environment and base project structure initialized.

### Suggestions
- Validate the dataset paths and formatting against the expected inputs of `HaMeR` and `MoGe-2` before writing the pipeline scripts.

---

## Stage 1.1: 3D Mesh Reconstruction for Human Hand

### Steps Taken
- Studied `third_party/hamer/demo.py` to understand the real HaMeR inference API.
- Rewrote `run_stage1.py` from scratch using the actual HaMeR pipeline (no more placeholder mocks).
- Updated `install_dependencies.sh` to include the full dependency chain: Detectron2, mmpose/mmcv, ViTPose submodule initialization, and the HaMeR checkpoint download.
- Changed Python version from 3.11 → 3.10 to match HaMeR's documented requirement.

### Pipeline Architecture (from demo.py analysis)
The real HaMeR pipeline has 5 stages per frame:
1. **Person Detection** (Detectron2 ViTDet) — finds person bounding boxes.
2. **Whole-Body Keypoint Detection** (ViTPose) — detects 133 COCO-WholeBody keypoints per person, including 21 left-hand and 21 right-hand keypoints.
3. **Hand BBox Extraction** — derives tight hand bounding boxes from keypoints with confidence > 0.5, requiring at least 3 valid keypoints.
4. **HaMeR Inference** — `ViTDetDataset` wraps the image + hand bboxes + is_right flags into a dataset; `model(batch)` returns `pred_vertices` (MANO 778×3 mesh) and `pred_cam` (weak-perspective camera).
5. **Camera Transform** — `cam_crop_to_full()` converts crop-space weak-perspective camera to full-image camera translation.

### Key Decisions
- **Skipping Body Detection for Egocentric Data:** Since the footage is egocentric (first-person), the camera wearer's body is always present. We pass the full frame as a single person detection to ViTPose, eliminating the need for Detectron2's body detector at inference time.
- **Per-Frame HDF5 Groups:** Output HDF5 uses per-frame groups (`frame_000000/`) instead of flat arrays, since the number of detected hands varies per frame (0, 1, or 2).
- **HDF5 Schema:** Each frame group contains: `vertices` (N_hands, 778, 3), `cam_t` (N_hands, 3), `is_right` (N_hands,).
- **Compute Limitation:** Run successfully executed on a `lightning.ai` cloud instance.

### Open Questions
- None for HaMeR currently. Hand mesh reconstruction works as expected on the cloud environment.

### Results
- `run_stage1.py` now uses the real HaMeR API: `download_models()`, `load_hamer()`, `ViTDetDataset`, `model(batch)`, `cam_crop_to_full()`.
- `install_dependencies.sh` includes full dependency chain including ViTPose submodule + mmpose.
- Successfully executed the `run_stage1.py` pipeline on `lightning.ai`, resulting in 25 `.hdf5` files containing the MANO mesh vertices, camera transformations, and handedness for each frame.

---

## Tooling Module: Remote Data Visualizer

### Steps Taken
- Installed `h5py`, `matplotlib`, `trimesh`, and `pyglet` directly into the local environment to support lightweight UI rendering.
- Built explicit file `visualize_stage1.py` taking `--file` (to pass target `.hdf5`) and `--frame` indices.
- Configured `--style` parsing to adapt between `points` (raw coordinate plotting) and `surface` modeling. 

### Key Decisions
- **Decoupled Environment:** Rendered the visualizer dependency tree to be purposefully separate from the intensive models, meaning it is safe to test locally while bypassing the main HaMeR weights.
- **Topology Approximation:** Set `visualize.py` to attempt to create a structural convex hull wrapping the point cloud if `--style surface` is requested but the topological MANO face file is missing.

### Open Questions
- In the future, do we intend to commit the `MANO_RIGHT.pkl` topology into the repo, or require users to provide it manually due to SMPL open-source licensing restrictions?

### Results
- Successfully executed `visualize.py` to render 3D keypoint scatter plots for the extracted `.hdf5` files.
- Visualizer accurately output `.mp4` animations and `.png` still frames for each processed video inside the `viz_output/` folder, confirming spatial coherence.

---

## Stage 1.1.2: Monocular Geometry Estimation (MoGe-2) [PLANNED]

### Next Steps
- Implement `run_stage1_moge.py` to run MoGe-2 monocular depth extraction independently.
- Iterate over the dataset and extract metric depth maps and 3D point maps.
- Handle massive data footprint of full-resolution float32 point/depth maps by optionally compressing to float16 and storing in HDF5 format.
