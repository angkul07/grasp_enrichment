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
- Designed `install_dependencies.sh` to install heavy PyTorch and detection libraries via `uv pip` securely within the virtual environment.
- Created `run_stage1.py`, a robust Python pipeline wrapping HaMeR and MoGe-2 execution across an array of Egocentric `.mp4` video files.
- Constructed parallel file architecture: configured the pipeline to target the `output/` directory, saving aligned 3D features into `[videoname]_stage1.hdf5` to isolate processed data from the raw source files.
- Configured CLI control, particularly integrating the `--limit 2` arg for rapid environment testing.

### Key Decisions
- **Compute Limitation Mapping:** Due to the user's local RTX 2050 (4GB VRAM) being insufficient for concurrent MoGe-2 and HaMeR operations, `run_stage1.py` acts as a deployment-ready scaffolding pointing precisely at `lightning.ai` for final inference compute.
- **Loguru Integration:** Used Loguru explicitly to maintain standard production-grade traces to `logs/stage1.log` as opposed to native print statements.

### Open Questions
- On the `lightning.ai` instance, will we need to orchestrate distributed GPU processing if the single GPU processing length over all 200+ videos proves too extensive computationally?

### Results
- Setup script and core architecture developed in `.py` environment, explicitly mapped and prepped for SSH transport.

---

## Tooling Module: Remote Data Visualizer

### Steps Taken
- Installed `h5py`, `matplotlib`, `trimesh`, and `pyglet` directly into the local environment to support lightweight UI rendering.
- Built explicit file `visualize_stage1.py` taking `--file` (to pass target `.hdf5`) and `--frame` indices.
- Configured `--style` parsing to adapt between `points` (raw coordinate plotting) and `surface` modeling. 

### Key Decisions
- **Decoupled Environment:** Rendered the visualizer dependency tree to be purposefully separate from the intensive models, meaning it is safe to test locally while bypassing the main HaMeR weights.
- **Topology Approximation:** Set `visualize_stage1.py` to attempt to create a structural convex hull wrapping the point cloud if `--style surface` is requested but the topological MANO face file is missing.

### Open Questions
- In the future, do we intend to commit the `MANO_RIGHT.pkl` topology into the repo, or require users to provide it manually due to SMPL open-source licensing restrictions?
