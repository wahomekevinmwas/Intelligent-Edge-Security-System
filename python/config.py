# python/config.py
# ─────────────────────────────────────────
# Edit these values for your machine before
# running Assignment5_KNN_Camera.py
# ─────────────────────────────────────────

# Serial port where Arduino Uno is connected
# Windows: 'COM3', 'COM4', etc.
# Linux:   '/dev/ttyUSB0', '/dev/ttyACM0'
# macOS:   '/dev/cu.usbmodem1401'
SERIAL_PORT = 'COM3'
BAUD_RATE = 9600

# Camera index (try 0, 1, 2 if camera doesn't open)
CAMERA_INDEX = 1
CAMERA_BACKEND = 'CAP_DSHOW'  # Windows DirectShow — remove on Linux/Mac

# Snapshot output directory (created automatically)
SNAPSHOT_DIR = 'snapshots'

# Flask dashboard
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = False

# KNN parameters
KNN_K = 3

# Drift detection threshold (consecutive disagreements before warning)
DRIFT_THRESHOLD = 3

# De-escalation timeout (milliseconds, must match Arduino firmware)
DEESCALATION_MS = 120000  # 2 minutes
