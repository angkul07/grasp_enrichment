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

## Stage 1.2: Object Reconstruction

### Objective
Reconstruct the grasped object's 3D surface in HaMeR's camera space, so that
ContactOpt (Step 1.3) can resolve hand–object interpenetration from HaMeR.

### Architecture Decisions

- **MoGe-2 for hands was abandoned** — MoGe-2 samples the scene surface at joint pixel locations,
  which collapses 21 articulated joints into a single depth blob. HaMeR already provides a much
  better 3D hand mesh. MoGe-2 is instead used **only for object depth**.

- **Coordinate system: HaMeR camera space** — Everything (hand mesh, object point cloud) lives
  in HaMeR's camera coordinate system (scaled pinhole: `scaled_focal = 5000/224 * max(H,W)`,
  principal point at image center). This ensures ContactOpt operates on hand + object in a
  single consistent frame.

- **Per-frame depth scale alignment** — MoGe-2 produces metric depth (meters) while HaMeR's
  camera Z values are in an arbitrary scale. A per-frame scale factor is computed by comparing
  HaMeR hand depth (`cam_t[2]`) with MoGe-2 depth at the hand wrist pixel location.

### Pipeline
1. Parse grasped object name from EgoDex `object` attribute (e.g., `"object:stapler, color:black"` → `"stapler"`).
2. Grounding DINO detects the object on frame 0 → bounding box.
3. SAM2 VideoPredictor propagates the mask across all frames.
4. MoGe-2 extracts depth per frame.
5. Per-frame scale alignment + HaMeR-camera back-projection → 3D object point cloud.
6. Farthest-point subsampled to 2048 points, stored in `*_stage1.hdf5`.

### HDF5 Schema (appended to existing files)
```
attrs['object_name']        : str        # grasped object name

frame_XXXXXX/
    obj_mask           : bool (H, W)     # SAM2 binary mask (LZ4 compressed)
    obj_points_3d      : float16 (N, 3)  # 3D point cloud in HaMeR camera space
    attrs['obj_mask_valid']  : bool      # SAM2 tracking confidence flag
    attrs['obj_n_points']    : int       # actual point count
    attrs['obj_depth_scale'] : float     # MoGe→HaMeR scale factor used
```

### Key Files
- `run_stage1_objects.py` — main object reconstruction script
- `install_dependencies.sh` — updated with SAM2 + Grounding DINO installation

### Open Questions
- ContactOpt integration (Step 1.3): will the ~2048-point object surface be dense enough?
  Can refine by increasing MAX_OBJ_POINTS if needed.

### Visualization Fix
- `visualize.py` was rewritten to handle the HaMeR / MoGe coordinate mismatch:
  - `--mode hamer` plots HaMeR vertices with proper equal-aspect scaling
  - `--mode both` uses side-by-side subplots with independent axis scaling
  - Also supports rendering future object point clouds (`obj_points_3d`) alongside hands
