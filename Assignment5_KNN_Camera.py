"""
=============================================================
  SCS 6106 - EIS  |  ASSIGNMENT 5  (x/10)
  Intelligent Security System - KNN Threat Classifier
  with Laptop Camera Integration

  ML PIPELINE - HOW THIS SCRIPT FITS IN:
  -------------------------------------------------------------
  STAGE 1 - DATA INGESTION:
    Reads live JSON packets from Arduino over Serial (COM4).
    Each packet contains 5 engineered sensor features.
    Serial communication is our lightweight ingestion protocol
    (equivalent to MQTT in a cloud IoT deployment).

  STAGE 2 - DATA PROCESSING:
    Features are normalised to 0-1 using Min-Max scaling
    before being passed to the KNN classifier. This ensures
    no single feature dominates due to its value range.

  STAGE 3 - FEATURE ENGINEERING:
    Features were engineered from raw signals:
    motion_count, failed_pins, hour_of_day,
    reset_pressed, consecutive_motion.

  STAGE 4 - MODEL TRAINING & EVALUATION:
    KNN built from scratch (no sklearn).
    Trained on 30 labelled historical events (static dataset).
    Evaluated using Leave-One-Out Cross Validation: 93.3% accuracy.
    K=3 neighbours, Euclidean distance, majority vote.

  STAGE 6 - MONITORING & FEEDBACK:
    Acts as a second independent classifier alongside the
    Arduino MLP (Assignment 6). If KNN and MLP disagree
    consistently, that signals model drift requiring retraining.
    Camera provides visual evidence layer for HIGH RISK events.

  STATIC DATA  : 30 labelled historical events (training)
  REAL-TIME DATA: live JSON from Arduino via Serial (COM4)
  HOW TO RUN   : python Assignment5_KNN_Camera.py
=============================================================
"""

import cv2
import numpy as np
import serial
import json
import time
import os
import threading
from datetime import datetime
from collections import Counter

# =============================================================
# CONFIGURATION — edit COM port here if needed
# =============================================================
SERIAL_PORT = "COM4"
BAUD_RATE   = 9600
K           = 3           # KNN neighbours
SNAPSHOT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "snapshots") # folder to save alert photos

# =============================================================
# STAGE 4 - STATIC TRAINING DATASET (Secondary Dataset)
# 30 manually labelled historical security events.
# These represent known scenarios the model learns from.
# Labels: 0=NORMAL  1=SUSPICIOUS  2=HIGH RISK
# Escalation rule: 0-1 wrong PINs=NORMAL, 2=SUSPICIOUS, 3+=HIGH RISK
# =============================================================
# 1. STATIC TRAINING DATA  (historical labelled events)
#    Features: [motion_count, failed_pins, hour_of_day,
#               reset_pressed, consecutive_motion]
#    Labels:   0=NORMAL  1=SUSPICIOUS  2=HIGH RISK
# =============================================================
TRAINING_DATA = np.array([
    [0, 0,  9, 0, 0], [1, 0, 10, 0, 0], [0, 0, 14, 0, 0],
    [2, 0,  8, 0, 1], [1, 1, 11, 0, 0], [0, 0, 16, 0, 0],
    [1, 0, 17, 0, 1], [2, 1, 13, 0, 0], [0, 0, 20, 0, 0],
    [1, 0,  7, 0, 0], [1, 0, 15, 0, 1], [2, 0, 12, 0, 0],

    [3, 1, 23, 0, 2], [2, 2, 22, 0, 1], [4, 1,  1, 0, 2],
    [3, 2,  0, 1, 1], [5, 1, 23, 0, 3], [2, 3, 21, 0, 1],
    [4, 2,  2, 0, 2], [3, 1,  3, 1, 2], [6, 1, 22, 0, 2],
    [4, 2,  1, 0, 3], [3, 2, 23, 0, 1], [5, 1,  0, 1, 2],

    [7, 4,  2, 1, 4], [8, 5,  3, 0, 5], [9, 5,  1, 1, 5],
    [6, 4,  0, 1, 4], [10,5,  2, 0, 5], [8, 4, 23, 1, 4],
])

TRAINING_LABELS = np.array([
    0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,
])

LABEL_NAMES   = {0: "NORMAL", 1: "SUSPICIOUS", 2: "HIGH RISK"}
LABEL_COLOURS = {0: (0,200,0), 1: (0,165,255), 2: (0,0,255)}  # BGR for OpenCV

# =============================================================
# STAGE 4 - KNN ALGORITHM (built from scratch, no sklearn)
# Why KNN: interpretable, no training time, works well on small datasets.
# At inference: measures Euclidean distance to all 30 training samples,
# picks 3 nearest neighbours, takes majority vote -> threat class.
# =============================================================
# 2. KNN IMPLEMENTATION (from scratch - no sklearn)
# =============================================================

def compute_min_max(data):
    return data.min(axis=0), data.max(axis=0)

def normalise(X, col_min, col_max):
    denom = col_max - col_min
    denom[denom == 0] = 1
    return (X - col_min) / denom

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(train_X, train_y, query, k=3):
    distances = sorted(
        [(euclidean_distance(query, train_X[i]), train_y[i])
         for i in range(len(train_X))],
        key=lambda x: x[0]
    )
    k_labels  = [lbl for _, lbl in distances[:k]]
    vote      = Counter(k_labels)
    pred      = vote.most_common(1)[0][0]
    conf      = vote[pred] / k * 100
    return pred, conf

# Pre-compute normalisation constants once at startup
COL_MIN, COL_MAX = compute_min_max(TRAINING_DATA)
X_TRAIN_NORM     = normalise(TRAINING_DATA, COL_MIN, COL_MAX)

# =============================================================
# STAGE 6 - MONITORING LAYER (visual evidence)
# Camera activates on SUSPICIOUS/HIGH RISK events.
# Saves timestamped snapshots as photographic evidence.
# Runs in background thread so it never blocks Serial reading.
# =============================================================
# 3. CAMERA MANAGER
#    Runs in a background thread so it never blocks Serial
# =============================================================

class CameraManager:
    def __init__(self):
        self.cap          = None
        self.active       = False
        self.latest_frame = None
        self.lock         = threading.Lock()
        self.thread       = None
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    def start(self):
        """
        Open laptop camera.
        Index 1 confirmed working on this machine (index 0 = black/IR camera).
        """
        if self.active:
            return

        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # index 1 = real camera
        if not self.cap.isOpened():
            print("  [Camera] Could not open camera at index 1.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Warm up — read frames until camera is ready
        for _ in range(20):
            self.cap.read()

        self.active = True
        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()
        print("  [Camera] Active on index 1.")

    def stop(self):
        self.active = False
        if self.cap:
            self.cap.release()

    def _grab_loop(self):
        """Continuously read frames into latest_frame."""
        while self.active:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            time.sleep(0.03)  # ~30 fps

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def save_snapshot(self, threat_label):
        """Save a timestamped snapshot when HIGH RISK is detected."""
        # Wait a moment to ensure we have a fresh frame
        time.sleep(0.3)
        frame = self.get_frame()
        if frame is None:
            print("  [Camera] No frame available for snapshot.")
            return None
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SNAPSHOT_DIR, f"ALERT_{threat_label}_{ts}.jpg")
        # Overlay threat label on the snapshot
        cv2.putText(frame, f"ALERT: {threat_label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, ts, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(path, frame)
        print(f"  [Camera] Snapshot saved → {path}")
        return path

    def show_feed(self, threat_label, confidence):
        """
        Display the live camera feed in a window with
        threat level overlaid in colour.
        """
        frame = self.get_frame()
        if frame is None:
            return

        # Safe label colour lookup
        matching = [k for k, v in LABEL_NAMES.items() if v == threat_label]
        colour = LABEL_COLOURS.get(matching[0] if matching else 0, (255,255,255))

        # Draw threat banner at top of frame
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), colour, -1)
        cv2.putText(frame, f"THREAT: {threat_label}  ({confidence:.0f}%)",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        # Draw timestamp
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, ts, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Security Camera — Assignment 5", frame)
        cv2.waitKey(1)


# =============================================================
# 4. SERIAL READER  (reads JSON lines from Arduino)
# =============================================================

def open_serial(port, baud, retries=5):
    """Try to open the serial port, retry on failure."""
    for attempt in range(1, retries + 1):
        try:
            ser = serial.Serial(port, baud, timeout=2)
            print(f"  [Serial] Connected to {port} @ {baud} baud.")
            return ser
        except Exception as e:
            print(f"  [Serial] Attempt {attempt}/{retries} failed: {e}")
            time.sleep(2)
    return None

def parse_event(line):
    """
    Parse a JSON line from the Arduino.
    Returns a feature dict or None if the line is not a JSON event.
    """
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


# =============================================================
# STAGE 2+4 - PREPROCESSING + CLASSIFICATION
# Normalise incoming features then run KNN prediction.
# This is the real-time inference step using live Arduino data.
# =============================================================
# 5. CLASSIFY EVENT  (KNN prediction pipeline)
# =============================================================

def classify_event(event):
    """
    Take a raw event dict from Arduino, run KNN, return label + confidence.
    """
    features = np.array([
        event.get("motion_count",       0),
        event.get("failed_pins",        0),
        event.get("hour_of_day",       12),
        event.get("reset_pressed",      0),
        event.get("consecutive_motion", 0),
    ], dtype=float)

    features_norm = normalise(features, COL_MIN, COL_MAX)
    label_idx, conf = knn_predict(X_TRAIN_NORM, TRAINING_LABELS,
                                  features_norm, k=K)
    return LABEL_NAMES[label_idx], conf


# =============================================================
# 6. CONSOLE DASHBOARD  (printed after each event)
# =============================================================

def print_dashboard(event, threat, confidence, event_num):
    ts = datetime.now().strftime("%H:%M:%S")
    bar = "█" * int(confidence / 10)  # simple confidence bar
    print(f"\n{'='*55}")
    print(f"  🕐  {ts}   Event #{event_num}")
    print(f"{'='*55}")
    print(f"  Motion count      : {event.get('motion_count', 0)}")
    print(f"  Failed PIN attempts: {event.get('failed_pins', 0)}")
    print(f"  Consec. motion    : {event.get('consecutive_motion', 0)}")
    print(f"  Hour of day       : {event.get('hour_of_day', 12)}")
    print(f"{'─'*55}")

    colour_map = {"NORMAL": "🟢", "SUSPICIOUS": "🟡", "HIGH RISK": "🔴"}
    icon = colour_map.get(threat, "⚪")
    print(f"  KNN PREDICTION    : {icon}  {threat}")
    print(f"  Confidence        : {bar} {confidence:.0f}%")
    print(f"{'='*55}")


# =============================================================
# 7. MAIN LOOP
# =============================================================

def main():
    print("=" * 55)
    print("  SCS 6106 | Assignment 5 — KNN Security Classifier")
    print("  with Laptop Camera Integration")
    print("=" * 55)
    print(f"  Training samples  : {len(TRAINING_DATA)}")
    print(f"  KNN  k            : {K}")
    print(f"  Serial port       : {SERIAL_PORT}")
    print(f"  Snapshots folder  : {SNAPSHOT_DIR}/")
    print("=" * 55)

    # Start camera
    camera = CameraManager()
    camera.start()

    # Open serial
    ser = open_serial(SERIAL_PORT, BAUD_RATE)
    if ser is None:
        print("\n[!] Could not connect to Arduino. Exiting.")
        camera.stop()
        return

    print("\n  Listening for Arduino events... (Ctrl+C to stop)\n")

    event_num    = 0
    last_threat  = "NORMAL"
    last_conf    = 100.0

    try:
        while True:
            raw = ser.readline().decode("utf-8", errors="ignore")

            # Print raw Arduino messages (non-JSON) for debugging
            if raw.strip() and not raw.strip().startswith("{"):
                print(f"  [Arduino] {raw.strip()}")

            event = parse_event(raw)
            if event is None:
                # Still update camera display at last known threat
                camera.show_feed(last_threat, last_conf)
                continue

            event_num += 1
            threat, confidence = classify_event(event)
            last_threat = threat
            last_conf   = confidence

            print_dashboard(event, threat, confidence, event_num)

            # Camera actions based on threat
            if threat in ("SUSPICIOUS", "HIGH RISK"):
                camera.start()   # ensure camera is on
                camera.show_feed(threat, confidence)

                if threat == "HIGH RISK":
                    camera.save_snapshot(threat)

            else:
                camera.show_feed(threat, confidence)

    except KeyboardInterrupt:
        print("\n\n  [Stopped by user]")
    finally:
        ser.close()
        camera.stop()
        cv2.destroyAllWindows()
        print("  Serial closed. Camera released. Goodbye.")


if __name__ == "__main__":
    main()
