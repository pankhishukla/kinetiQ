"""
app.py
======
Main entry point for the Exercise Form Detection system.

Run with:   python app.py

Key bindings:
    1  Bicep Curl      5  Shoulder Press
    2  Squat           6  Lunge
    3  Lateral Raise   7  Plank
    4  Push-Up
    R  Reset rep counter
    D  Toggle debug overlay
    Q  Quit
"""

import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO

# Clean imports from the src package — no more shim files needed.
from src.pose_extractor import (
    KEYPOINT_NAMES, CONF_THRESHOLD,
    SKELETON_CONNECTIONS, draw_skeleton,
)
from src.angle_engine import get_exercise_angles
from src.form_evaluator import (
    FORM_RULES, evaluate_form,
    draw_feedback_overlay,
    COLOR_CORRECT, COLOR_INCORRECT, COLOR_UNKNOWN,
    RepCounter,
)
# ---- NEW: premium renderer ------------------------------------------------
from src.renderer import (
    draw_pose, KeypointSmoother, TrailBuffer,
    draw_confidence_label, ACTIVE_JOINTS,
    Colors,
)

# =============================================================================
# SMOOTHING PARAMETERS
# =============================================================================
# WHY smooth?  Raw per-frame angles jitter by ±5–10 degrees because YOLOv8
# keypoints move slightly each frame even when the person is still.
# EMA dampens this noise; debouncing prevents the feedback label from
# flickering every other frame.

EMA_ALPHA          = 0.75   # 75% new frame weight, 25% history (real-time response)
DEBOUNCE_WINDOW    = 10     # frames kept in vote buffer
DEBOUNCE_THRESHOLD = 0.6    # fraction of window needed to flip status


# =============================================================================
# JOINT SMOOTHER — EMA + debouncing for one joint
# =============================================================================

class JointSmoother:
    """
    Maintains an EMA-smoothed angle AND a majority-vote status window
    for a single joint across frames.

    WHY separate class per joint?
        Each joint has its own noise characteristics and its own history.
        Mixing joints into one smoother would be incorrect.
    """

    def __init__(self, window_size=DEBOUNCE_WINDOW):
        self.ema_angle     = None
        self.status_window = deque(maxlen=window_size)

    def update_angle(self, new_angle):
        if new_angle is None:
            return self.ema_angle
        if self.ema_angle is None:
            self.ema_angle = new_angle
        else:
            self.ema_angle = (EMA_ALPHA * new_angle
                              + (1 - EMA_ALPHA) * self.ema_angle)
        return self.ema_angle

    def update_status(self, new_status):
        self.status_window.append(new_status)
        if len(self.status_window) < 3:
            return new_status
        total           = len(self.status_window)
        correct_count   = self.status_window.count("correct")
        incorrect_count = self.status_window.count("incorrect")
        if incorrect_count / total >= DEBOUNCE_THRESHOLD:
            return "incorrect"
        elif correct_count / total >= DEBOUNCE_THRESHOLD:
            return "correct"
        return "correct" if correct_count >= incorrect_count else "incorrect"

    def reset(self):
        self.ema_angle = None
        self.status_window.clear()


# =============================================================================
# EXERCISE SMOOTHER — registry of JointSmoothers; resets on exercise switch
# =============================================================================

class ExerciseSmoother:
    """
    Manages a collection of JointSmoothers, one per joint in the current
    exercise.  Automatically resets all smoothers when the user switches
    exercise so stale history from the previous exercise doesn't bleed in.
    """

    def __init__(self):
        self.smoothers        = {}
        self.current_exercise = None

    def set_exercise(self, exercise_name):
        if exercise_name != self.current_exercise:
            self.smoothers        = {}
            self.current_exercise = exercise_name

    def _get(self, joint_name):
        if joint_name not in self.smoothers:
            self.smoothers[joint_name] = JointSmoother()
        return self.smoothers[joint_name]

    def smooth(self, angles_dict, evaluations_dict):
        """Apply EMA + debouncing to each joint's angle and status."""
        smoothed = {}
        for joint_name, eval_data in evaluations_dict.items():
            smoother    = self._get(joint_name)
            smooth_angle = smoother.update_angle(eval_data["angle"])

            # Re-evaluate form on smoothed angle (more stable than raw)
            if smooth_angle is not None and self.current_exercise in FORM_RULES:
                rule = FORM_RULES[self.current_exercise].get(joint_name)
                if rule:
                    if smooth_angle < rule["min"]:
                        raw_status, cue = "incorrect", rule["cue_low"]
                        base_color = COLOR_INCORRECT
                    elif smooth_angle > rule["max"]:
                        raw_status, cue = "incorrect", rule["cue_high"]
                        base_color = COLOR_INCORRECT
                    else:
                        raw_status, cue = "correct", rule["cue_good"]
                        base_color = COLOR_CORRECT
                else:
                    raw_status = eval_data["status"]
                    cue        = eval_data["cue"]
                    base_color = eval_data["color"]
            else:
                raw_status = "unknown"
                cue        = eval_data["cue"]
                base_color = COLOR_UNKNOWN

            debounced   = smoother.update_status(raw_status)
            final_color = (COLOR_CORRECT   if debounced == "correct" else
                           COLOR_INCORRECT if debounced == "incorrect" else
                           COLOR_UNKNOWN)

            smoothed[joint_name] = {
                "status":       debounced,
                "cue":          cue,
                "angle":        smooth_angle,
                "color":        final_color,
                "vertex_xy":    eval_data["vertex_xy"],
                "display_name": eval_data["display_name"],
            }
        return smoothed

    def get_smoothed_angles(self):
        """Return {joint_name: {"angle": smoothed_value}} for rep counting."""
        return {jn: {"angle": s.ema_angle} for jn, s in self.smoothers.items()}


# =============================================================================
# DEBUG OVERLAY
# =============================================================================

def draw_debug(frame, raw_angles, smooth_evals):
    """Show raw vs smoothed angle side-by-side for development/demo."""
    h = frame.shape[0]
    x0, y0 = 10, h - 150
    cv2.rectangle(frame, (x0 - 5, y0 - 20),
                  (x0 + 270, y0 + len(raw_angles) * 22 + 5), (0, 0, 0), -1)
    cv2.putText(frame, "DEBUG: Raw vs Smoothed",
                (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 100), 1, cv2.LINE_AA)
    y = y0 + 15
    for jn in raw_angles:
        ra = raw_angles[jn]["angle"]
        sa = smooth_evals.get(jn, {}).get("angle")
        rs = f"{ra:.1f}" if ra is not None else "N/A"
        ss = f"{sa:.1f}" if sa is not None else "N/A"
        cv2.putText(frame, f"{jn[:12]:12s}  raw:{rs:6s} sm:{ss:6s}",
                    (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (200, 200, 200), 1, cv2.LINE_AA)
        y += 20
    return frame


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    model    = YOLO("models/yolov8n-pose.pt")
    smoother = ExerciseSmoother()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    current_exercise = "bicep_curl"
    smoother.set_exercise(current_exercise)
    rep_counter = RepCounter(current_exercise)

    EXERCISE_KEYS = {
        ord('1'): "bicep_curl",
        ord('2'): "squat",
        ord('3'): "lateral_raise",
        ord('4'): "push_up",
        ord('5'): "shoulder_press",
        ord('6'): "lunge",
        ord('7'): "plank",
    }

    show_debug   = False
    show_conf    = False   # press C to toggle confidence % labels
    prev_time    = time.time()

    # ---- per-person visual helpers (index 0 = first detected person) ----
    kp_smoother  = KeypointSmoother(alpha=0.80)  # high responsiveness, minimal lag
    trail_buffer = TrailBuffer(max_frames=6)

    print("[INFO] Exercise Form Detection — running")
    print("[INFO] 1=Bicep Curl 2=Squat 3=Lat Raise 4=Push-Up")
    print("[INFO] 5=Shoulder Press 6=Lunge 7=Plank")
    print("[INFO] R=Reset reps  D=Debug overlay  C=Confidence labels  Q=Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results    = model(frame, verbose=False, conf=0.5)
        kp_data    = results[0].keypoints
        num_people = 0

        if kp_data is not None and len(kp_data.data) > 0:
            num_people = len(kp_data.data)
            for person_kpts in kp_data.data:
                kpts = person_kpts.cpu().numpy()
                raw_keypoints = [(kpts[i][0], kpts[i][1], kpts[i][2])
                                 for i in range(17)]

                # --- VISUAL SMOOTHING (cosmetic only) ---------------------
                # EMA-smooth the keypoint (x,y) positions before drawing to
                # eliminate per-frame jitter.  Angle math still uses raw.
                smooth_kpts = kp_smoother.update(raw_keypoints)
                trail_buffer.push(smooth_kpts)

                # --- BUILD EVAL COLOR MAP (kp_index → BGR) ----------------
                # Tells the renderer which joints are correct/incorrect so
                # it can override the default anatomy color.
                raw_angles  = get_exercise_angles(raw_keypoints, current_exercise)
                raw_evals   = evaluate_form(raw_angles, current_exercise)
                smooth_evals = smoother.smooth(raw_angles, raw_evals)

                # Map joint name -> COCO index -> BGR color
                eval_color_map = {}
                for joint_name, ev in smooth_evals.items():
                    # Find the COCO index for this joint name
                    if joint_name in KEYPOINT_NAMES:
                        idx = KEYPOINT_NAMES.index(joint_name)
                        eval_color_map[idx] = ev["color"]

                # --- DRAW POSE (premium renderer) -------------------------
                frame = draw_pose(
                    frame,
                    smooth_kpts,
                    conf_threshold      = CONF_THRESHOLD,
                    exercise_name       = current_exercise,
                    eval_colors         = eval_color_map,
                    skeleton_connections= SKELETON_CONNECTIONS,
                    line_thickness      = 3,
                    joint_radius        = 8,
                    show_glow           = True,
                    trail_kpts          = list(trail_buffer.frames())[1:],  # skip current
                )

                if show_conf:
                    frame = draw_confidence_label(frame, smooth_kpts, CONF_THRESHOLD)

                # Rep counter operates on smoothed ANGLES (less jitter)
                rep_counter.update(smoother.get_smoothed_angles())
                frame = draw_feedback_overlay(
                    frame, smooth_evals, rep_counter.count, current_exercise)

                if show_debug:
                    frame = draw_debug(frame, raw_angles, smooth_evals)

        # HUD
        cv2.putText(frame,
                    f"Exercise: {current_exercise.replace('_', ' ').title()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(frame,
                    f"EMA={EMA_ALPHA}  Win={DEBOUNCE_WINDOW}  Thr={DEBOUNCE_THRESHOLD}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (180, 180, 180), 1, cv2.LINE_AA)
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f} | People: {num_people}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

        debug_lbl = "Debug: ON (D)" if show_debug else "Debug: OFF (D)"
        cv2.putText(frame, debug_lbl,
                    (10, frame.shape[0] - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1,
                    cv2.LINE_AA)

        cv2.imshow("Exercise Form Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in EXERCISE_KEYS:
            current_exercise = EXERCISE_KEYS[key]
            smoother.set_exercise(current_exercise)
            rep_counter  = RepCounter(current_exercise)
            kp_smoother.reset()    # clear visual history on exercise switch
            trail_buffer.reset()   # clear trail ghosts
            print(f"[INFO] Switched to: {current_exercise}")
        elif key == ord('r'):
            rep_counter.reset()
            print("[INFO] Rep counter reset")
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('c'):
            show_conf = not show_conf
            print(f"[INFO] Confidence labels: {'ON' if show_conf else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")


if __name__ == "__main__":
    main()
