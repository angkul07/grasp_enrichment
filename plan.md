# Project Plan

This document outlines the step-by-step plan for the 3D Human-Object Interaction (HOI) pipeline, focused on adapting egocentric human video to robot policies.

## Stage 0: Environment Setup & Data Preparation
- Initialize `uv` virtual environment.
- Collect and structure third-party repositories (`HaMeR`, `MoGe-2`, etc.) in `third_party/`.
- Prepare the EgoDex `basic_pick_place` dataset and assess structure.

## Stage 1: Kinematic Gap Closure
Closing the kinematic gap between human operations and robot execution via 3D mesh reconstruction and retargeting.

- **Step 1.1**: 3D Mesh Reconstruction for Human Hand
  - Integrate HaMeR for 3D hand pose estimation from egocentric RGB footage.
  - Apply MoGe-2 to extract monocular metric depth and align the hand in 3D space.
- **Step 1.2**: Object Reconstruction
  - Determine optimal tools (e.g., SAM2/FoundationPose) for extracting interacting object geometry from the video.
- **Step 1.3**: Contact Retargeting
  - Implement ContactOpt for contact-aware retargeting.
  - Map human hand-object interactions to the robot end-effector URDF, generating initial robot joint trajectories and wrist poses.
- **Step 1.4**: Evaluation & Logging
  - Validate median IK error, interference, and contact plausibility. Log outcomes to `docs.md`.

## Stage 2: Perceptual / Semantic Gap Closure (EgoBridge)
Closing the perceptual gap through latent space alignment.

- **Step 2.1**: EgoBridge Implementation
  - Set up shared encoders for human observations and robot observations.
- **Step 2.2**: Temporal Alignment
  - Use Dynamic Time Warping (DTW) to form pseudo-pairs between robot trajectories and human demonstrations.
- **Step 2.3**: Distribution Alignment
  - Add Optimal Transport (OT) latent alignment loss during training.
- **Step 2.4**: Model Training & Evaluation
  - Train the Robot policy (e.g., DP3) on the aligned representations. Evaluate success rates.

## Stage 3: Future Phases
- Scalability to cross-embodiment retargeting.
- Add force estimation.
