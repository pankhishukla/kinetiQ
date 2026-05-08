"""
src/pose_extractor.py
=====================
Foundation module: webcam capture, YOLOv8-Pose inference, skeleton drawing.

WHY this exists as a separate module:
    Every other part of the system (angle engine, form evaluator, app) needs
    the same constants (keypoint indices, skeleton layout, confidence threshold)
    and the same drawing helpers. Centralising them here means changing a
    constant once fixes it everywhere.
"""

import cv2
import time
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# MODEL  (loaded once at import time so every importer shares one instance)
# ---------------------------------------------------------------------------
# WHY load at import? Loading YOLOv8 takes ~1–2 seconds.  If each module
# loaded it independently we'd pay that cost multiple times.  Sharing one
# instance also means only one copy of weights lives in RAM.
model = YOLO("yolov8m-pose.pt")

# ---------------------------------------------------------------------------
# KEYPOINT NAMES  — COCO-17 format
# ---------------------------------------------------------------------------
# WHY store names: using KEYPOINT_NAMES.index("left_elbow") in the angle
# engine is self-documenting.  Magic numbers like '7' are error-prone.
KEYPOINT_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# ---------------------------------------------------------------------------
# SKELETON CONNECTIONS  — which joints to connect with lines
# ---------------------------------------------------------------------------
# WHY omit face? Eyes/ears/nose add visual clutter and carry zero information
# for exercise form analysis.
SKELETON_CONNECTIONS = [
    (5, 6),   # shoulders
    (5, 7),   # L upper arm
    (7, 9),   # L forearm
    (6, 8),   # R upper arm
    (8, 10),  # R forearm
    (5, 11),  # L torso
    (6, 12),  # R torso
    (11, 12), # pelvis
    (11, 13), # L thigh
    (13, 15), # L shin
    (12, 14), # R thigh
    (14, 16), # R shin
]

# ---------------------------------------------------------------------------
# DRAWING CONSTANTS
# ---------------------------------------------------------------------------
KEYPOINT_COLOR    = (0, 255, 255)   # Cyan
SKELETON_COLOR    = (255, 255, 255) # White
KEYPOINT_RADIUS   = 5
SKELETON_THICKNESS = 2

# WHY 0.5? Below this confidence YOLOv8's keypoint location is unreliable.
# Jittery low-confidence points would corrupt angle math downstream.
CONF_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# draw_skeleton
# ---------------------------------------------------------------------------
def draw_skeleton(frame, keypoints):
    """
    Paint joint dots and bone lines onto *frame* in-place.

    Parameters
    ----------
    frame     : ndarray (H, W, 3)  — BGR frame from webcam
    keypoints : list of 17 (x, y, conf) tuples

    Returns
    -------
    frame  (same object, modified in-place for efficiency)
    """
    # Joint dots
    for idx, (x, y, conf) in enumerate(keypoints):
        if conf < CONF_THRESHOLD:
            continue
        cv2.circle(frame, (int(x), int(y)), KEYPOINT_RADIUS,
                   KEYPOINT_COLOR, -1)

    # Bone lines — only when BOTH endpoints are confident
    for (a, b) in SKELETON_CONNECTIONS:
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca < CONF_THRESHOLD or cb < CONF_THRESHOLD:
            continue
        cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                 SKELETON_COLOR, SKELETON_THICKNESS)

    return frame


# ---------------------------------------------------------------------------
# draw_hud
# ---------------------------------------------------------------------------
def draw_hud(frame, fps, num_people):
    """Minimal heads-up display: FPS + person count."""
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                cv2.LINE_AA)
    cv2.putText(frame, f"People detected: {num_people}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# main  — standalone demo (runs only when this file is executed directly)
# ---------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Pose Extractor — press Q to quit")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, verbose=False, conf=0.5)
        kp_data = results[0].keypoints
        num_people = 0

        if kp_data is not None and len(kp_data.data) > 0:
            num_people = len(kp_data.data)
            for person in kp_data.data:
                kpts = person.cpu().numpy()
                keypoints_list = [(kpts[i][0], kpts[i][1], kpts[i][2])
                                  for i in range(17)]
                frame = draw_skeleton(frame, keypoints_list)

        curr_time = time.time()
        frame = draw_hud(frame, 1.0 / (curr_time - prev_time), num_people)
        prev_time = curr_time

        cv2.imshow("Pose Extractor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
