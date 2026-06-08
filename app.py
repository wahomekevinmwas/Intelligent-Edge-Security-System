"""
=============================================================
  SCS 6106 — EIS  |  Web Dashboard Server
  Intelligent Rental Property Access Control System
  Mwangi Kevin Wahome | SCS 61/49733/2025

  ARCHITECTURE:
    Arduino ──Serial JSON──► Python (KNN + Flask/SocketIO) ──WebSocket──► Browser
                                         │
                                    OpenCV Camera

  HOW TO RUN:
    1. pip install flask flask-socketio eventlet opencv-python pyserial numpy
    2. Set SERIAL_PORT below (or leave as "AUTO" for auto-detection)
    3. python app.py
    4. Open http://localhost:5000 in Chrome or Firefox
    5. Upload Arduino sketch first, then run this script

  COM PORT:
    Change SERIAL_PORT to your Arduino port e.g. "COM3", "COM4",
    "/dev/ttyUSB0" (Linux), "/dev/tty.usbmodem14101" (Mac).
    Set to "AUTO" to scan all ports automatically.
=============================================================
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║                  CONFIGURATION — EDIT HERE                  ║
# ╚══════════════════════════════════════════════════════════════╝

SERIAL_PORT   = "AUTO"    # "COM3" / "COM4" / "AUTO" to auto-detect
BAUD_RATE     = 9600
CAMERA_INDEX  = 0         # 0 = default webcam, 1 = second camera
FLASK_PORT    = 5000      # Browser: http://localhost:5000
K             = 3         # KNN neighbours

# ╔══════════════════════════════════════════════════════════════╗
# ║                    IMPORTS & SETUP                          ║
# ╚══════════════════════════════════════════════════════════════╝

import os, json, time, threading, base64, io
from datetime import datetime
from collections import Counter, deque

import cv2
import numpy as np
import serial
import serial.tools.list_ports
from flask import Flask, render_template_string
from flask_socketio import SocketIO

SNAPSHOT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'eis_scs6106_kevin'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins='*')

# ── Shared state (thread-safe via simple dicts + deque) ───────────────
state = {
    'knn_threat':   'NORMAL',
    'mlp_threat':   'NORMAL',
    'confidence':   100.0,
    'sim_time':     '08:00',
    'motion':       0,
    'failed_pins':  0,
    'consecutive':  0,
    'reset':        0,
    'hour':         8,
    'agree_count':  0,
    'total_count':  0,
    'drift_counter':0,
    'system_state': 'READY',
    'connected':    False,
    'port_used':    '',
}
event_log    = deque(maxlen=80)
threat_history = deque(maxlen=60)   # last 60 events for chart
snapshots    = deque(maxlen=20)     # base64 encoded snapshots

# ╔══════════════════════════════════════════════════════════════╗
# ║              TRAINING DATASET — 72 SAMPLES                  ║
# ╚══════════════════════════════════════════════════════════════╝
# Features: [motion_count, failed_pins, hour_of_day,
#            reset_pressed, consecutive_motion]
# Labels  : 0=NORMAL  1=SUSPICIOUS  2=HIGH RISK

TRAINING_DATA = np.array([
    # NORMAL (24)
    [0,0,9,0,0],[1,0,10,0,0],[0,0,14,0,0],[2,0,8,0,1],[1,1,11,0,0],[0,0,16,0,0],
    [1,0,17,0,1],[2,1,13,0,0],[0,0,20,0,0],[1,0,7,0,0],[1,0,15,0,1],[2,0,12,0,0],
    [0,1,9,0,0],[1,0,18,0,0],[2,0,10,0,2],[0,0,11,1,0],[1,1,14,1,0],[2,0,16,0,1],
    [0,0,8,0,0],[1,0,19,0,0],[2,1,15,0,0],[0,0,13,0,0],[1,0,21,0,1],[2,0,9,1,0],
    # SUSPICIOUS (24)
    [3,1,23,0,2],[2,2,22,0,1],[4,1,1,0,2],[3,2,0,1,1],[5,1,23,0,3],[2,3,21,0,1],
    [4,2,2,0,2],[3,1,3,1,2],[6,1,22,0,2],[4,2,1,0,3],[3,2,23,0,1],[5,1,0,1,2],
    [2,2,20,0,1],[4,1,22,0,3],[3,3,19,0,0],[1,2,23,0,0],[5,2,2,0,2],[3,1,21,0,4],
    [2,2,22,1,1],[4,1,0,0,3],[6,2,23,0,1],[3,2,4,0,2],[5,1,3,1,3],[2,3,20,0,2],
    # HIGH RISK (24)
    [7,4,2,1,4],[8,5,3,0,5],[9,5,1,1,5],[6,4,0,1,4],[10,5,2,0,5],[8,4,23,1,4],
    [7,5,4,0,5],[9,3,3,1,4],[6,5,1,0,5],[8,5,2,1,5],[10,4,0,1,4],[9,5,3,0,5],
    [7,3,2,0,5],[8,4,1,1,5],[6,5,23,0,4],[10,5,4,1,5],[9,4,0,0,5],[7,5,3,1,4],
    [8,3,2,0,5],[6,4,1,0,5],[9,5,0,1,4],[7,4,4,0,5],[10,5,2,1,5],[8,5,1,0,5],
])
TRAINING_LABELS = np.array([0]*24 + [1]*24 + [2]*24)
LABEL_NAMES = {0:'NORMAL', 1:'SUSPICIOUS', 2:'HIGH RISK'}

# ╔══════════════════════════════════════════════════════════════╗
# ║              KNN — FROM SCRATCH (no sklearn)                ║
# ╚══════════════════════════════════════════════════════════════╝

COL_MIN = TRAINING_DATA.min(axis=0)
COL_MAX = TRAINING_DATA.max(axis=0)

def normalise(X):
    d = COL_MAX - COL_MIN
    d[d == 0] = 1
    return (X - COL_MIN) / d

X_TRAIN_NORM = normalise(TRAINING_DATA)

def knn_predict(features):
    q = normalise(np.array(features, dtype=float))
    dists = [
        (float(np.sqrt(np.sum((q - X_TRAIN_NORM[i])**2))), int(TRAINING_LABELS[i]))
        for i in range(len(X_TRAIN_NORM))
    ]
    dists.sort(key=lambda x: x[0])
    labels = [l for _, l in dists[:K]]
    vote   = Counter(labels)
    pred   = vote.most_common(1)[0][0]
    conf   = vote[pred] / K * 100
    return pred, conf

# ╔══════════════════════════════════════════════════════════════╗
# ║                    CAMERA MANAGER                           ║
# ╚══════════════════════════════════════════════════════════════╝

class CameraManager:
    def __init__(self):
        self.cap    = None
        self.active = False
        self.lock   = threading.Lock()
        self.latest = None
        self._try_open()

    def _try_open(self):
        for idx in [CAMERA_INDEX, 0, 1, 2]:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == 'nt' else 0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                for _ in range(8): cap.read()
                self.cap    = cap
                self.active = True
                t = threading.Thread(target=self._loop, daemon=True)
                t.start()
                print(f"  [Camera] Opened at index {idx}")
                return
        print("  [Camera] No camera found — evidence capture disabled")

    def _loop(self):
        while self.active:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest = frame
            time.sleep(0.033)

    def get_frame_b64(self, threat, sim_time):
        """Returns annotated frame as base64 JPEG for browser display."""
        with self.lock:
            frame = self.latest.copy() if self.latest is not None else None
        if frame is None:
            return None
        # Threat colour
        colours = {'NORMAL':(39,80,10),'SUSPICIOUS':(6,56,99),'HIGH RISK':(31,31,121)}
        bgr = colours.get(threat, (80,80,80))
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,64), bgr, -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.putText(frame, f"THREAT: {threat}", (12,42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        ts = f"{sim_time}  |  {datetime.now().strftime('%d %b %Y')}"
        cv2.rectangle(frame, (0,h-32), (w,h), (20,20,20), -1)
        cv2.putText(frame, ts, (10,h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf).decode('utf-8')

    def save_snapshot(self, threat, sim_time):
        with self.lock:
            frame = self.latest.copy() if self.latest is not None else None
        if frame is None:
            return None
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SNAPSHOT_DIR, f"ALERT_{threat.replace(' ','_')}_{ts}.jpg")
        h, w = frame.shape[:2]
        colours = {'NORMAL':(39,80,10),'SUSPICIOUS':(6,56,99),'HIGH RISK':(31,31,121)}
        bgr = colours.get(threat, (80,80,80))
        cv2.rectangle(frame, (0,0), (w,64), bgr, -1)
        cv2.putText(frame, f"ALERT: {threat}", (12,42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(frame, f"{sim_time} | {datetime.now().strftime('%d %b %Y %H:%M:%S')}",
                    (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.imwrite(path, frame)
        # Also return base64 for gallery
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf).decode('utf-8')
        return path, b64

camera = CameraManager()

# ╔══════════════════════════════════════════════════════════════╗
# ║                  SERIAL AUTO-DETECTION                      ║
# ╚══════════════════════════════════════════════════════════════╝

def find_arduino_port():
    """Scans COM ports and returns the first one that looks like an Arduino."""
    ports = list(serial.tools.list_ports.comports())
    print(f"  [Serial] Available ports: {[p.device for p in ports]}")
    # Priority: Arduino in description
    for p in ports:
        desc = (p.description or '').lower()
        if 'arduino' in desc or 'ch340' in desc or 'cp210' in desc or 'ftdi' in desc:
            print(f"  [Serial] Auto-detected: {p.device} ({p.description})")
            return p.device
    # Fallback: first available port
    if ports:
        print(f"  [Serial] Using first available: {ports[0].device}")
        return ports[0].device
    return None

def open_serial():
    port = SERIAL_PORT if SERIAL_PORT != "AUTO" else find_arduino_port()
    if port is None:
        print("  [Serial] No COM ports found.")
        return None
    for attempt in range(1, 6):
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=2)
            print(f"  [Serial] Connected: {port} @ {BAUD_RATE} baud")
            state['connected'] = True
            state['port_used'] = port
            return ser
        except Exception as e:
            print(f"  [Serial] Attempt {attempt}/5 failed: {e}")
            time.sleep(2)
    return None

# ╔══════════════════════════════════════════════════════════════╗
# ║               SERIAL READER + KNN THREAD                    ║
# ╚══════════════════════════════════════════════════════════════╝

def serial_thread():
    ser = open_serial()
    if ser is None:
        socketio.emit('system_log', {
            'msg': 'ERROR: Arduino not found. Check USB connection and port.',
            'level': 'error'
        })
        return

    socketio.emit('system_log', {
        'msg': f"Connected to Arduino on {state['port_used']}",
        'level': 'info'
    })

    last_mlp = 'NORMAL'

    while True:
        try:
            raw = ser.readline().decode('utf-8', errors='ignore').strip()
            if not raw:
                continue

            # MLP result line from Arduino
            if raw.startswith('ML_THREAT:'):
                parts = raw.split(',')
                last_mlp = parts[0].replace('ML_THREAT:', '').strip()
                continue

            # State / event lines
            if raw.startswith('STATE:') or raw.startswith('EVENT:') or raw.startswith('WINDOW:'):
                socketio.emit('system_log', {'msg': raw, 'level': 'info'})
                continue

            # JSON packet
            if not raw.startswith('{'):
                continue

            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Extract features
            features = [
                event.get('motion_count',       0),
                event.get('failed_pins',         0),
                event.get('hour_of_day',        12),
                event.get('reset_pressed',       0),
                event.get('consecutive_motion',  0),
            ]
            h   = event.get('hour_of_day', 8)
            m   = event.get('sim_minute',  0)
            sim = f"{int(h):02d}:{int(m):02d}"

            # KNN classify
            pred_idx, conf = knn_predict(features)
            knn_label = LABEL_NAMES[pred_idx]
            mlp_label = last_mlp if last_mlp in LABEL_NAMES.values() else 'NORMAL'

            # Drift detection
            state['total_count'] += 1
            if knn_label == mlp_label:
                state['agree_count'] += 1
                state['drift_counter'] = 0
            else:
                state['drift_counter'] += 1
                if state['drift_counter'] >= 3:
                    socketio.emit('system_log', {
                        'msg': f'⚠ DRIFT WARNING: KNN/MLP disagreement x{state["drift_counter"]}',
                        'level': 'warning'
                    })

            # Update shared state
            state.update({
                'knn_threat':  knn_label,
                'mlp_threat':  mlp_label,
                'confidence':  conf,
                'sim_time':    sim,
                'motion':      features[0],
                'failed_pins': features[1],
                'hour':        features[2],
                'reset':       features[3],
                'consecutive': features[4],
                'system_state':event.get('state', 'READY'),
            })

            agree_pct = (state['agree_count'] / state['total_count'] * 100) if state['total_count'] > 0 else 100

            # Threat history for chart
            threat_history.append({
                'time':   sim,
                'knn':    pred_idx,
                'mlp':    list(LABEL_NAMES.values()).index(mlp_label) if mlp_label in LABEL_NAMES.values() else 0,
                'label':  knn_label,
            })

            # Camera frame (always stream)
            cam_b64 = camera.get_frame_b64(knn_label, sim)

            # Snapshot on threat
            snap_b64 = None
            snap_name = None
            if knn_label in ('SUSPICIOUS', 'HIGH RISK'):
                result = camera.save_snapshot(knn_label, sim)
                if result:
                    snap_path, snap_b64 = result
                    snap_name = os.path.basename(snap_path)
                    snapshots.appendleft({
                        'name':   snap_name,
                        'threat': knn_label,
                        'time':   sim,
                        'b64':    snap_b64,
                    })

            # Emit to all browser clients
            socketio.emit('update', {
                'knn':        knn_label,
                'mlp':        mlp_label,
                'confidence': round(conf, 1),
                'sim_time':   sim,
                'features': {
                    'motion':      features[0],
                    'failed_pins': features[1],
                    'hour':        features[2],
                    'reset':       features[3],
                    'consecutive': features[4],
                },
                'agree_pct':   round(agree_pct, 1),
                'total_events':state['total_count'],
                'system_state':event.get('state', 'READY'),
                'history':     list(threat_history),
                'camera_b64':  cam_b64,
                'snap_b64':    snap_b64,
                'snap_name':   snap_name,
            })

            # Event log entry
            ts_now = datetime.now().strftime('%H:%M:%S')
            log_entry = {
                'time':    ts_now,
                'sim':     sim,
                'knn':     knn_label,
                'mlp':     mlp_label,
                'motion':  features[0],
                'pins':    features[1],
                'agree':   knn_label == mlp_label,
            }
            event_log.appendleft(log_entry)
            socketio.emit('log_entry', log_entry)

        except serial.SerialException as e:
            state['connected'] = False
            socketio.emit('system_log', {'msg': f'Serial disconnected: {e}', 'level': 'error'})
            time.sleep(3)
            ser = open_serial()
        except Exception as e:
            print(f"  [Thread] Error: {e}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                  CAMERA STREAM THREAD                       ║
# ╚══════════════════════════════════════════════════════════════╝

def camera_stream_thread():
    """Pushes camera frames to browser at ~10fps even when no serial events."""
    while True:
        time.sleep(0.1)
        b64 = camera.get_frame_b64(state['knn_threat'], state['sim_time'])
        if b64:
            socketio.emit('camera_frame', {'b64': b64})


# ╔══════════════════════════════════════════════════════════════╗
# ║                    FLASK ROUTES                             ║
# ╚══════════════════════════════════════════════════════════════╝

@app.route('/')
def index():
    return render_template_string(HTML)

@socketio.on('connect')
def on_connect():
    """Send current state to newly connected browser."""
    socketio.emit('update', {
        'knn':        state['knn_threat'],
        'mlp':        state['mlp_threat'],
        'confidence': state['confidence'],
        'sim_time':   state['sim_time'],
        'features': {
            'motion':      state['motion'],
            'failed_pins': state['failed_pins'],
            'hour':        state['hour'],
            'reset':       state['reset'],
            'consecutive': state['consecutive'],
        },
        'agree_pct':   100.0,
        'total_events':state['total_count'],
        'system_state':state['system_state'],
        'history':     list(threat_history),
        'camera_b64':  None,
        'snap_b64':    None,
        'snap_name':   None,
    })
    # Send existing log
    for entry in list(event_log)[:20]:
        socketio.emit('log_entry', entry)
    # Send existing snapshots
    for snap in list(snapshots):
        socketio.emit('snapshot', snap)
    socketio.emit('system_log', {
        'msg': f"Dashboard connected. Arduino: {'✓ ' + state['port_used'] if state['connected'] else '✗ Not connected'}",
        'level': 'info' if state['connected'] else 'warning'
    })

@socketio.on('request_snapshots')
def send_snapshots():
    for snap in list(snapshots):
        socketio.emit('snapshot', snap)


# ╔══════════════════════════════════════════════════════════════╗
# ║                    HTML DASHBOARD                           ║
# ╚══════════════════════════════════════════════════════════════╝

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Intelligent Security System — SCS 6106</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: Arial, sans-serif; background: #0F1117; color: #E8E8E8; min-height: 100vh; }

  /* ── Header ── */
  header {
    background: #1A1D2E; border-bottom: 2px solid #2E75B6;
    padding: 12px 24px; display: flex; justify-content: space-between; align-items: center;
  }
  header h1 { font-size: 15px; font-weight: 600; color: #fff; }
  header span { font-size: 11px; color: #8899AA; }
  .conn-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
              background: #E24B4A; margin-right: 6px; transition: background 0.3s; }
  .conn-dot.on { background: #1D9E75; }

  /* ── Grid ── */
  .grid {
    display: grid;
    grid-template-columns: 280px 1fr 320px;
    grid-template-rows: auto auto auto;
    gap: 12px; padding: 12px; max-width: 1400px; margin: 0 auto;
  }

  /* ── Cards ── */
  .card {
    background: #1A1D2E; border: 1px solid #2A2D40;
    border-radius: 10px; padding: 14px; overflow: hidden;
  }
  .card-title {
    font-size: 10px; font-weight: 600; color: #6677AA;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 10px; padding-bottom: 6px;
    border-bottom: 1px solid #2A2D40;
  }

  /* ── Threat gauge ── */
  #threat-box {
    grid-column: 1; grid-row: 1;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 180px;
    transition: border-color 0.4s;
  }
  #threat-box.NORMAL    { border-color: #1D9E75; }
  #threat-box.SUSPICIOUS{ border-color: #EF9F27; }
  #threat-box.HIGH-RISK { border-color: #E24B4A; }
  #threat-label {
    font-size: 28px; font-weight: 700; color: #1D9E75;
    transition: color 0.3s; margin-bottom: 6px;
  }
  #threat-label.NORMAL    { color: #1D9E75; }
  #threat-label.SUSPICIOUS{ color: #EF9F27; }
  #threat-label.HIGH-RISK { color: #E24B4A; }
  #conf-bar-wrap { width: 100%; background: #2A2D40; border-radius: 4px; height: 6px; margin: 8px 0; }
  #conf-bar { height: 6px; border-radius: 4px; background: #1D9E75; width: 100%; transition: width 0.4s, background 0.4s; }
  .meta-row { font-size: 11px; color: #8899AA; margin-top: 4px; }
  .meta-val { font-weight: 600; color: #DDD; margin-left: 4px; }
  #sys-state-badge {
    font-size: 11px; font-weight: 600; padding: 3px 10px;
    border-radius: 12px; background: #2A2D40; color: #8899AA;
    margin-top: 10px; display: inline-block;
  }

  /* ── Feature bars ── */
  #features-box { grid-column: 1; grid-row: 2; }
  .feat-row { margin-bottom: 10px; }
  .feat-label { font-size: 11px; color: #9AABB8; margin-bottom: 3px;
                display: flex; justify-content: space-between; }
  .feat-val { font-weight: 600; color: #DDD; }
  .feat-bar-bg { background: #2A2D40; border-radius: 3px; height: 7px; }
  .feat-bar { height: 7px; border-radius: 3px; background: #2E75B6;
              transition: width 0.5s; min-width: 2px; }

  /* ── KNN vs MLP panel ── */
  #agreement-box { grid-column: 1; grid-row: 3; }
  .clf-row { display: flex; gap: 8px; margin-bottom: 8px; }
  .clf-card {
    flex: 1; background: #12152A; border-radius: 8px; padding: 8px 10px;
    text-align: center; border: 1px solid #2A2D40;
  }
  .clf-name { font-size: 9px; font-weight: 600; color: #6677AA; text-transform: uppercase; }
  .clf-val  { font-size: 14px; font-weight: 700; color: #DDD; margin-top: 4px; }
  #agree-indicator { text-align: center; font-size: 24px; font-weight: 700; padding: 0 4px; }
  #agree-pct-row { display: flex; justify-content: space-between; font-size: 11px;
                   color: #8899AA; margin-top: 6px; }
  #drift-warn { font-size: 11px; color: #EF9F27; margin-top: 6px; display: none; }

  /* ── Camera ── */
  #camera-box { grid-column: 2; grid-row: 1 / 3; }
  #cam-img { width: 100%; border-radius: 6px; background: #0A0C14;
             min-height: 260px; display: block; object-fit: cover; }
  #no-cam { padding: 60px 20px; text-align: center; color: #444; font-size: 13px; }

  /* ── Threat chart ── */
  #chart-box { grid-column: 2; grid-row: 3; }
  #threat-chart { max-height: 140px; }

  /* ── Event log ── */
  #log-box { grid-column: 3; grid-row: 1 / 3; }
  #log-list { max-height: 380px; overflow-y: auto; }
  .log-entry {
    font-size: 11px; padding: 5px 6px; border-radius: 4px;
    margin-bottom: 4px; border-left: 3px solid #2A2D40;
    background: #12152A;
  }
  .log-entry.NORMAL    { border-color: #1D9E75; }
  .log-entry.SUSPICIOUS{ border-color: #EF9F27; }
  .log-entry.HIGH-RISK { border-color: #E24B4A; }
  .log-time { color: #6677AA; font-size: 10px; }
  .log-knn  { font-weight: 600; }
  .log-agree{ font-size: 10px; color: #1D9E75; }
  .log-disagree{ font-size: 10px; color: #EF9F27; }

  /* ── Snapshot gallery ── */
  #snap-box { grid-column: 3; grid-row: 3; }
  #snap-gallery { display: flex; flex-wrap: wrap; gap: 6px; max-height: 200px; overflow-y: auto; }
  .snap-thumb { position: relative; cursor: pointer; }
  .snap-thumb img { width: 90px; height: 60px; object-fit: cover; border-radius: 4px;
                    border: 2px solid #E24B4A; }
  .snap-label { position: absolute; bottom: 2px; left: 2px; font-size: 8px;
                font-weight: 600; color: #fff; background: rgba(0,0,0,0.7);
                padding: 1px 4px; border-radius: 2px; }

  /* ── System log ── */
  #syslog-box { grid-column: 1 / 4; padding: 8px 14px; }
  #syslog { font-size: 11px; color: #6677AA; font-family: monospace;
            max-height: 52px; overflow-y: auto; }
  .syslog-info    { color: #6699CC; }
  .syslog-warning { color: #EF9F27; }
  .syslog-error   { color: #E24B4A; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: #12152A; }
  ::-webkit-scrollbar-thumb { background: #2A2D40; border-radius: 2px; }
</style>
</head>
<body>

<header>
  <div>
    <h1>Intelligent Rental Property Access Control System</h1>
    <span>SCS 6106 — EIS &nbsp;|&nbsp; Mwangi Kevin Wahome &nbsp;|&nbsp; SCS 61/49733/2025</span>
  </div>
  <div style="text-align:right;font-size:11px;color:#8899AA">
    <span class="conn-dot" id="conn-dot"></span>
    <span id="conn-text">Connecting...</span>
    &nbsp;&nbsp;
    <span id="clock" style="color:#DDD;font-weight:600"></span>
  </div>
</header>

<div class="grid">

  <!-- Threat gauge -->
  <div class="card" id="threat-box">
    <div class="card-title">Current threat level</div>
    <div id="threat-label">NORMAL</div>
    <div id="conf-bar-wrap"><div id="conf-bar"></div></div>
    <div class="meta-row">KNN confidence: <span class="meta-val" id="conf-val">100%</span></div>
    <div class="meta-row">Simulated time: <span class="meta-val" id="sim-time">--:--</span></div>
    <div class="meta-row">Events seen:    <span class="meta-val" id="evt-count">0</span></div>
    <div id="sys-state-badge">READY</div>
  </div>

  <!-- Feature bars -->
  <div class="card" id="features-box">
    <div class="card-title">Feature vector</div>
    <div class="feat-row">
      <div class="feat-label">Motion count <span class="feat-val" id="f-motion">0</span></div>
      <div class="feat-bar-bg"><div class="feat-bar" id="fb-motion" style="width:0%"></div></div>
    </div>
    <div class="feat-row">
      <div class="feat-label">Failed PINs <span class="feat-val" id="f-pins">0</span></div>
      <div class="feat-bar-bg"><div class="feat-bar" id="fb-pins" style="width:0%;background:#E24B4A"></div></div>
    </div>
    <div class="feat-row">
      <div class="feat-label">Hour of day <span class="feat-val" id="f-hour">8</span></div>
      <div class="feat-bar-bg"><div class="feat-bar" id="fb-hour" style="width:33%;background:#8877DD"></div></div>
    </div>
    <div class="feat-row">
      <div class="feat-label">Consecutive motion <span class="feat-val" id="f-cons">0</span></div>
      <div class="feat-bar-bg"><div class="feat-bar" id="fb-cons" style="width:0%;background:#EF9F27"></div></div>
    </div>
    <div class="feat-row">
      <div class="feat-label">Reset pressed <span class="feat-val" id="f-reset">0</span></div>
      <div class="feat-bar-bg"><div class="feat-bar" id="fb-reset" style="width:0%;background:#1D9E75"></div></div>
    </div>
  </div>

  <!-- KNN vs MLP -->
  <div class="card" id="agreement-box">
    <div class="card-title">Classifier cross-validation</div>
    <div class="clf-row">
      <div class="clf-card">
        <div class="clf-name">KNN (PC)</div>
        <div class="clf-val" id="clf-knn">NORMAL</div>
      </div>
      <div id="agree-indicator" style="color:#1D9E75">=</div>
      <div class="clf-card">
        <div class="clf-name">MLP (Arduino)</div>
        <div class="clf-val" id="clf-mlp">NORMAL</div>
      </div>
    </div>
    <div id="agree-pct-row">
      <span>Agreement</span>
      <span id="agree-pct-val" style="color:#1D9E75;font-weight:600">100%</span>
    </div>
    <div id="agree-bar-bg" style="background:#2A2D40;border-radius:3px;height:6px;margin-top:6px">
      <div id="agree-bar" style="height:6px;border-radius:3px;background:#1D9E75;width:100%;transition:width 0.4s"></div>
    </div>
    <div id="drift-warn">⚠ Model drift detected — retraining recommended</div>
  </div>

  <!-- Camera feed -->
  <div class="card" id="camera-box">
    <div class="card-title">Live camera feed</div>
    <img id="cam-img" src="" alt="Camera feed" onerror="this.style.display='none'">
    <div id="no-cam" style="display:none">No camera feed available</div>
  </div>

  <!-- Threat history chart -->
  <div class="card" id="chart-box">
    <div class="card-title">Threat history (last 60 events)</div>
    <canvas id="threat-chart"></canvas>
  </div>

  <!-- Event log -->
  <div class="card" id="log-box">
    <div class="card-title">Event log</div>
    <div id="log-list"></div>
  </div>

  <!-- Snapshot gallery -->
  <div class="card" id="snap-box">
    <div class="card-title">Snapshot evidence gallery</div>
    <div id="snap-gallery"><span style="font-size:11px;color:#444">Snapshots appear here on HIGH RISK events</span></div>
  </div>

  <!-- System log -->
  <div class="card" id="syslog-box">
    <div class="card-title">System log</div>
    <div id="syslog"></div>
  </div>

</div><!-- /grid -->

<script>
const socket = io();

// ── Real clock ───────────────────────────────────────────────────────
setInterval(() => {
  document.getElementById('clock').textContent = new Date().toLocaleTimeString();
}, 1000);

// ── Chart setup ──────────────────────────────────────────────────────
const ctx = document.getElementById('threat-chart').getContext('2d');
const threatChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'KNN threat',
      data: [],
      borderColor: '#2E75B6',
      backgroundColor: 'rgba(46,117,182,0.1)',
      tension: 0.3, pointRadius: 3, fill: true,
    },{
      label: 'MLP threat',
      data: [],
      borderColor: '#1D9E75',
      backgroundColor: 'rgba(29,158,117,0.05)',
      borderDash: [4,3], tension: 0.3, pointRadius: 2, fill: false,
    }]
  },
  options: {
    responsive: true, animation: false,
    scales: {
      y: { min: -0.1, max: 2.1, ticks: {
            stepSize: 1, color: '#6677AA',
            callback: v => ['NRM','SUS','HI'][v] || ''
          }, grid: { color: '#1A1D2E' } },
      x: { ticks: { color: '#6677AA', maxTicksLimit: 8 }, grid: { color: '#1A1D2E' } }
    },
    plugins: { legend: { labels: { color: '#8899AA', font: { size: 10 } } } }
  }
});

const THREAT_COLOURS = {
  'NORMAL':    { border:'#1D9E75', bar:'#1D9E75', text:'#1D9E75' },
  'SUSPICIOUS':{ border:'#EF9F27', bar:'#EF9F27', text:'#EF9F27' },
  'HIGH RISK': { border:'#E24B4A', bar:'#E24B4A', text:'#E24B4A' },
};

// ── Main update handler ──────────────────────────────────────────────
socket.on('update', d => {
  const threat  = d.knn;
  const col     = THREAT_COLOURS[threat] || THREAT_COLOURS['NORMAL'];
  const cls     = threat.replace(' ','-');

  // Threat gauge
  const tl = document.getElementById('threat-label');
  tl.textContent = threat;
  tl.className   = cls;
  document.getElementById('threat-box').className = 'card ' + cls;

  const confPct = d.confidence;
  document.getElementById('conf-bar').style.width      = confPct + '%';
  document.getElementById('conf-bar').style.background = col.bar;
  document.getElementById('conf-val').textContent      = confPct.toFixed(0) + '%';
  document.getElementById('sim-time').textContent      = d.sim_time;
  document.getElementById('evt-count').textContent     = d.total_events;

  const badge = document.getElementById('sys-state-badge');
  badge.textContent = d.system_state;
  const bcolors = {READY:'#1D9E75',ALERT:'#8877DD',SUSPICIOUS:'#EF9F27',HIGH_RISK:'#E24B4A',LOCKOUT:'#E24B4A'};
  badge.style.background = bcolors[d.system_state] || '#2A2D40';
  badge.style.color = '#fff';

  // Feature bars
  const f = d.features;
  const updateFeat = (id, val, max, pct100) => {
    document.getElementById('f-' + id).textContent = val;
    document.getElementById('fb-' + id).style.width = Math.min(val / max * 100, 100) + '%';
  };
  updateFeat('motion', f.motion,      10,  true);
  updateFeat('pins',   f.failed_pins,  5,  true);
  updateFeat('hour',   f.hour,        23,  true);
  updateFeat('cons',   f.consecutive,  5,  true);
  document.getElementById('f-reset').textContent = f.reset;
  document.getElementById('fb-reset').style.width = (f.reset * 100) + '%';

  // KNN vs MLP
  document.getElementById('clf-knn').textContent = d.knn;
  document.getElementById('clf-mlp').textContent = d.mlp;
  const agree = d.knn === d.mlp;
  const ai = document.getElementById('agree-indicator');
  ai.textContent = agree ? '=' : '≠';
  ai.style.color = agree ? '#1D9E75' : '#E24B4A';
  const pct = d.agree_pct;
  document.getElementById('agree-pct-val').textContent = pct.toFixed(0) + '%';
  document.getElementById('agree-bar').style.width = pct + '%';
  document.getElementById('agree-bar').style.background = pct >= 80 ? '#1D9E75' : '#E24B4A';
  document.getElementById('drift-warn').style.display = pct < 70 ? 'block' : 'none';

  // Chart
  if (d.history && d.history.length) {
    const h = d.history;
    threatChart.data.labels   = h.map(x => x.time);
    threatChart.data.datasets[0].data = h.map(x => x.knn);
    threatChart.data.datasets[1].data = h.map(x => x.mlp);
    threatChart.update('none');
  }

  // Camera frame
  if (d.camera_b64) {
    const img = document.getElementById('cam-img');
    img.src = 'data:image/jpeg;base64,' + d.camera_b64;
    img.style.display = 'block';
  }

  // New snapshot
  if (d.snap_b64 && d.snap_name) {
    addSnapshot(d.snap_b64, d.snap_name, threat, d.sim_time);
  }
});

// ── Camera frame (continuous stream) ─────────────────────────────────
socket.on('camera_frame', d => {
  if (d.b64) {
    const img = document.getElementById('cam-img');
    img.src = 'data:image/jpeg;base64,' + d.b64;
    img.style.display = 'block';
  }
});

// ── Event log ─────────────────────────────────────────────────────────
socket.on('log_entry', e => {
  const cls    = e.knn.replace(' ','-');
  const agree  = e.knn === e.mlp;
  const div    = document.createElement('div');
  div.className= 'log-entry ' + cls;
  div.innerHTML= `<span class="log-time">${e.time} [${e.sim}]</span>
    <span class="log-knn" style="color:${cls==='NORMAL'?'#1D9E75':cls==='SUSPICIOUS'?'#EF9F27':'#E24B4A'};margin:0 6px">${e.knn}</span>
    <span class="${agree?'log-agree':'log-disagree'}">${agree?'✓ agree':'≠ drift'}</span>
    <span style="color:#556;font-size:10px;margin-left:6px">m=${e.motion} p=${e.pins}</span>`;
  const list = document.getElementById('log-list');
  list.insertBefore(div, list.firstChild);
  if (list.children.length > 60) list.removeChild(list.lastChild);
});

// ── Snapshot gallery ──────────────────────────────────────────────────
function addSnapshot(b64, name, threat, simTime) {
  const gallery = document.getElementById('snap-gallery');
  // Remove placeholder text
  const placeholder = gallery.querySelector('span');
  if (placeholder) placeholder.remove();

  const div = document.createElement('div');
  div.className = 'snap-thumb';
  div.title = `${threat} — ${simTime}\n${name}`;
  div.innerHTML = `<img src="data:image/jpeg;base64,${b64}">
    <div class="snap-label">${threat === 'HIGH RISK' ? 'HI' : 'SU'}</div>`;
  div.onclick = () => window.open('data:image/jpeg;base64,' + b64, '_blank');
  gallery.insertBefore(div, gallery.firstChild);
}

socket.on('snapshot', d => {
  if (d.b64) addSnapshot(d.b64, d.name, d.threat, d.time);
});

// ── System log ────────────────────────────────────────────────────────
socket.on('system_log', d => {
  const el  = document.getElementById('syslog');
  const ts  = new Date().toLocaleTimeString();
  const div = document.createElement('div');
  div.className = 'syslog-' + (d.level || 'info');
  div.textContent = `[${ts}] ${d.msg}`;
  el.insertBefore(div, el.firstChild);
  if (el.children.length > 20) el.removeChild(el.lastChild);
});

// ── Connection indicator ──────────────────────────────────────────────
socket.on('connect', () => {
  document.getElementById('conn-dot').classList.add('on');
  document.getElementById('conn-text').textContent = 'Dashboard connected';
});
socket.on('disconnect', () => {
  document.getElementById('conn-dot').classList.remove('on');
  document.getElementById('conn-text').textContent = 'Disconnected';
});
</script>
</body>
</html>
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║                        MAIN                                 ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    print("=" * 60)
    print("  SCS 6106 — Intelligent Security System Dashboard")
    print("  Mwangi Kevin Wahome | SCS 61/49733/2025")
    print("=" * 60)
    print(f"  Serial port  : {SERIAL_PORT}")
    print(f"  Baud rate    : {BAUD_RATE}")
    print(f"  Camera index : {CAMERA_INDEX}")
    print(f"  Snapshots    : {SNAPSHOT_DIR}")
    print(f"  Dashboard    : http://localhost:{FLASK_PORT}")
    print("=" * 60)

    # Start serial + camera threads
    threading.Thread(target=serial_thread,     daemon=True).start()
    threading.Thread(target=camera_stream_thread, daemon=True).start()

    # Start Flask
    socketio.run(app, host='0.0.0.0', port=FLASK_PORT, debug=False)
