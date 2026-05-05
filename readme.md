# =============================================================================
# PHASE 1 — Foundation & Pose Extraction
# Exercise Form Detection System
# =============================================================================
# GOAL OF THIS PHASE:
#   Open the webcam, run every frame through YOLOv8-Pose, and draw the
#   detected skeleton (keypoints + connecting lines) on screen in real time.
#   This is the "eyes" of the entire system — every later phase reads the
#   keypoint data that this phase produces.
# ==========================

Phase 1 — Foundation & Pose Extraction 
Install YOLOv8, capture webcam feed, extract & visualize keypoints live

Phase 2 — Angle Calculation Engine
Compute joint angles from keypoints, display them on screen in real time

Phase 3 — Rule-Based Form Evaluator
Apply thresholds per exercise, green/red feedback overlay

Phase 4 — Temporal Smoothing & Confidence Gating
Rolling window, keypoint confidence filter, eliminate jitter & false positives

Phase 5 — Dataset Integration
Use your dataset to validate/calibrate thresholds, build a one-class anomaly detector

Phase 6 — Full System Integration & Demo Polish
Exercise selector UI, rep counter, audio feedback, clean presentation mode