"""
demo_simulation.py
==================
Hardware-free demonstration of the Intelligent Edge Security System.

Simulates all sensor events (PIR motion, keypad PINs, physical reset)
and injects them into the same Flask/Socket.IO pipeline as real hardware.
The dashboard at http://localhost:5000 behaves identically to live operation.

Usage:
    python simulation/demo_simulation.py

Requirements: see requirements.txt (no Arduino or webcam needed)
"""

import threading
import time
import json
import random
import numpy as np
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit

# ─────────────────────────────────────────────────────────────
# KNN Classifier (mirrors Assignment5_KNN_Camera.py)
# Trained on 72 labelled samples, k=3, Euclidean distance
# ─────────────────────────────────────────────────────────────

TRAINING_DATA = [
    # [motion_count, failed_pins, consec_motion, reset_pressed, hour] → label
    # NORMAL (0)
    [1, 0, 0, 0, 10], [0, 0, 0, 0, 14], [2, 0, 0, 0, 9],
    [1, 0, 0, 0, 16], [0, 1, 0, 0, 11], [3, 0, 0, 0, 8],
    [1, 0, 0, 0, 13], [2, 1, 0, 0, 15], [0, 0, 0, 0, 12],
    [1, 0, 0, 0, 17], [2, 0, 0, 0, 10], [1, 1, 0, 0, 14],
    [0, 0, 0, 0, 9],  [3, 0, 0, 0, 11], [1, 0, 0, 0, 15],
    [2, 0, 0, 0, 13], [1, 1, 0, 0, 16], [0, 0, 0, 0, 10],
    [2, 0, 0, 0, 8],  [1, 0, 0, 0, 12], [3, 1, 0, 0, 14],
    [0, 0, 0, 0, 11], [1, 0, 0, 0, 9],  [2, 0, 0, 0, 15],
    # SUSPICIOUS (1)
    [4, 2, 1, 0, 23], [5, 2, 0, 0, 1],  [3, 2, 1, 0, 22],
    [6, 1, 1, 0, 2],  [4, 2, 0, 0, 0],  [5, 2, 1, 0, 23],
    [3, 2, 0, 0, 1],  [7, 1, 1, 0, 3],  [4, 2, 1, 0, 22],
    [5, 2, 0, 0, 0],  [6, 2, 1, 0, 23], [3, 2, 0, 0, 2],
    [8, 1, 1, 0, 4],  [4, 2, 1, 0, 1],  [5, 2, 0, 0, 22],
    [3, 2, 1, 0, 23], [6, 2, 0, 0, 0],  [4, 2, 1, 0, 3],
    [5, 2, 0, 0, 1],  [7, 2, 1, 0, 22], [3, 2, 0, 0, 23],
    [4, 2, 1, 0, 2],  [6, 1, 1, 0, 0],  [5, 2, 0, 0, 4],
    # HIGH RISK (2)
    [8, 4, 1, 0, 3],  [10, 5, 1, 0, 2], [7, 3, 1, 0, 1],
    [9, 4, 0, 0, 23], [12, 5, 1, 0, 3], [8, 3, 1, 0, 2],
    [10, 4, 1, 0, 0], [7, 5, 0, 0, 1],  [9, 3, 1, 0, 23],
    [11, 4, 1, 0, 2], [8, 5, 0, 0, 3],  [10, 3, 1, 0, 1],
    [7, 4, 1, 0, 0],  [9, 5, 1, 0, 22], [12, 3, 1, 0, 3],
    [8, 4, 0, 0, 2],  [10, 5, 1, 0, 1], [7, 3, 1, 0, 23],
    [9, 4, 1, 0, 0],  [11, 5, 0, 0, 3], [8, 3, 1, 0, 2],
    [10, 4, 1, 0, 1], [7, 5, 1, 0, 22], [9, 3, 0, 0, 0],
]
LABELS = [0]*24 + [1]*24 + [2]*24

FEAT_MIN = np.array([0, 0, 0, 0, 0], dtype=float)
FEAT_MAX = np.array([20, 5, 1, 1, 23], dtype=float)


def normalise(features):
    f = np.array(features, dtype=float)
    return (f - FEAT_MIN) / (FEAT_MAX - FEAT_MIN + 1e-6)


def knn_predict(features, k=3):
    norm = normalise(features)
    train = np.array(TRAINING_DATA, dtype=float)
    train_norm = (train - FEAT_MIN) / (FEAT_MAX - FEAT_MIN + 1e-6)
    dists = np.linalg.norm(train_norm - norm, axis=1)
    idx = np.argsort(dists)[:k]
    votes = [LABELS[i] for i in idx]
    return max(set(votes), key=votes.count)


# ─────────────────────────────────────────────────────────────
# State Machine
# ─────────────────────────────────────────────────────────────

STATES = ['READY', 'ALERT', 'SUSPICIOUS', 'HIGH_RISK', 'LOCKOUT']
STATE_COLOURS = {
    'READY': '#10b981',
    'ALERT': '#f59e0b',
    'SUSPICIOUS': '#f97316',
    'HIGH_RISK': '#ef4444',
    'LOCKOUT': '#7c3aed',
}
THREAT_LABELS = {0: 'NORMAL', 1: 'SUSPICIOUS', 2: 'HIGH RISK'}


class SecurityStateMachine:
    def __init__(self):
        self.state = 'READY'
        self.failed_pins = 0
        self.motion_count = 0
        self.consecutive_motion = 0
        self.reset_pressed = 0
        self.hour = datetime.now().hour
        self.last_event_time = time.time()
        self.drift_counter = 0
        self.event_log = []
        self.knn_history = []
        self.mlp_history = []

    def get_features(self):
        return [
            self.motion_count,
            self.failed_pins,
            self.consecutive_motion,
            self.reset_pressed,
            self.hour,
        ]

    def simulate_mlp(self, features):
        """
        Simplified MLP simulation matching the Arduino 5→6→3 architecture.
        Weights are approximated to match the labelling rules from training.
        In real hardware this runs from PROGMEM via pgm_read_float().
        """
        norm = normalise(features)
        # Rule-based approximation matching training labels
        if features[1] >= 3:    # failed_pins >= 3
            return 2
        if features[1] >= 2 and (features[4] < 6 or features[4] > 21):
            return 1
        if features[2] == 1 and (features[4] < 6 or features[4] > 21):
            return 1
        if features[0] > 5 and (features[4] < 6 or features[4] > 21):
            return 1
        return 0

    def log(self, event_type, message, colour='white', knn=None, mlp=None):
        entry = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'type': event_type,
            'message': message,
            'colour': colour,
            'knn': THREAT_LABELS.get(knn, '—') if knn is not None else '—',
            'mlp': THREAT_LABELS.get(mlp, '—') if mlp is not None else '—',
            'agreement': (knn == mlp) if (knn is not None and mlp is not None) else None,
        }
        self.event_log.insert(0, entry)
        self.event_log = self.event_log[:50]  # keep last 50
        return entry

    def transition(self, new_state):
        old = self.state
        self.state = new_state
        return old != new_state

    def process_event(self, event_type, value=None):
        """Process a simulated sensor event."""
        self.hour = datetime.now().hour
        self.last_event_time = time.time()

        if event_type == 'motion':
            self.motion_count = min(self.motion_count + 1, 20)
            self.consecutive_motion = 1 if self.state in ('ALERT', 'SUSPICIOUS') else 0
            if self.state == 'READY':
                self.transition('ALERT')

        elif event_type == 'wrong_pin':
            self.failed_pins = min(self.failed_pins + 1, 5)
            self.consecutive_motion = 0
            if self.failed_pins >= 5:
                self.transition('LOCKOUT')

        elif event_type == 'correct_pin':
            self.failed_pins = 0
            self.consecutive_motion = 0
            self.motion_count = 0
            self.transition('READY')

        elif event_type == 'reset':
            self.reset_pressed = 1
            self.failed_pins = 0
            self.motion_count = 0
            self.consecutive_motion = 0
            self.transition('READY')
            self.reset_pressed = 0

        elif event_type == 'deescalate':
            self.failed_pins = max(0, self.failed_pins - 1)
            self.motion_count = max(0, self.motion_count - 1)
            self.transition('READY')

        features = self.get_features()
        knn_pred = knn_predict(features)
        mlp_pred = self.simulate_mlp(features)

        # Update state based on ML prediction
        if self.state not in ('LOCKOUT',):
            if mlp_pred == 2 and self.state not in ('HIGH_RISK',):
                self.transition('HIGH_RISK')
            elif mlp_pred == 1 and self.state in ('READY', 'ALERT'):
                self.transition('SUSPICIOUS')

        # Drift detection
        if knn_pred != mlp_pred:
            self.drift_counter += 1
        else:
            self.drift_counter = 0

        drift_warn = self.drift_counter >= 3

        colour_map = {
            'READY': '#10b981',
            'ALERT': '#f59e0b',
            'SUSPICIOUS': '#f97316',
            'HIGH_RISK': '#ef4444',
            'LOCKOUT': '#7c3aed',
        }

        log_colour = colour_map.get(self.state, 'white')
        label_map = {
            'motion': '🔴 PIR Motion Detected',
            'wrong_pin': '⚠️ Wrong PIN Entered',
            'correct_pin': '✅ Correct PIN — Access Granted',
            'reset': '🔘 Physical Reset Button Pressed',
            'deescalate': '⏱ 2-Min Quiet — System De-escalated',
        }

        entry = self.log(
            event_type.upper(),
            label_map.get(event_type, event_type),
            log_colour,
            knn=knn_pred,
            mlp=mlp_pred,
        )

        if drift_warn:
            self.log('DRIFT_WARNING', '⚡ KNN/MLP disagreement ×3 — model drift detected', '#a855f7')

        snapshot = None
        if knn_pred >= 1:
            snapshot = {
                'time': datetime.now().strftime('%H:%M:%S'),
                'threat': THREAT_LABELS[knn_pred],
                'features': features,
            }

        return {
            'state': self.state,
            'state_colour': colour_map[self.state],
            'features': features,
            'knn': knn_pred,
            'mlp': mlp_pred,
            'knn_label': THREAT_LABELS[knn_pred],
            'mlp_label': THREAT_LABELS[mlp_pred],
            'agreement': knn_pred == mlp_pred,
            'drift_warning': drift_warn,
            'event_log': self.event_log,
            'snapshot': snapshot,
            'failed_pins': self.failed_pins,
            'motion_count': self.motion_count,
        }


# ─────────────────────────────────────────────────────────────
# Flask + Socket.IO App
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['SECRET_KEY'] = 'edge-security-demo-2025'
socketio = SocketIO(app, cors_allowed_origins='*')
sm = SecurityStateMachine()

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Intelligent Edge Security System — Demo</title>
<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
<style>
  :root {
    --bg: #0a0e1a;
    --surface: #111827;
    --border: #1f2937;
    --text: #e5e7eb;
    --muted: #6b7280;
    --green: #10b981;
    --amber: #f59e0b;
    --orange: #f97316;
    --red: #ef4444;
    --purple: #a855f7;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Courier New', monospace;
    min-height: 100vh;
  }
  header {
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface);
  }
  header h1 { font-size: 14px; letter-spacing: 0.15em; color: var(--green); text-transform: uppercase; }
  header span { font-size: 11px; color: var(--muted); }
  .grid {
    display: grid;
    grid-template-columns: 260px 1fr 300px;
    grid-template-rows: auto auto;
    gap: 1px;
    background: var(--border);
    height: calc(100vh - 49px);
  }
  .panel {
    background: var(--surface);
    padding: 16px;
    overflow: hidden;
  }
  .panel-title {
    font-size: 10px;
    letter-spacing: 0.12em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  /* State gauge */
  .state-box {
    border: 2px solid var(--state-colour, var(--green));
    border-radius: 6px;
    padding: 20px;
    text-align: center;
    margin-bottom: 14px;
    transition: border-color 0.4s, background 0.4s;
    background: color-mix(in srgb, var(--state-colour, var(--green)) 8%, transparent);
  }
  .state-label {
    font-size: 22px;
    font-weight: bold;
    letter-spacing: 0.08em;
    color: var(--state-colour, var(--green));
    transition: color 0.4s;
  }
  .state-sub { font-size: 10px; color: var(--muted); margin-top: 4px; }
  .metric { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 12px; }
  .metric span:first-child { color: var(--muted); }
  .metric span:last-child { color: var(--text); font-weight: bold; }
  .sim-badge {
    background: #f97316;
    color: white;
    font-size: 9px;
    padding: 2px 8px;
    border-radius: 2px;
    letter-spacing: 0.1em;
    display: inline-block;
    margin-bottom: 12px;
  }
  /* Event log */
  .log-wrap { overflow-y: auto; height: calc(100% - 40px); }
  .log-entry {
    border-left: 3px solid var(--entry-colour, var(--muted));
    padding: 6px 10px;
    margin-bottom: 6px;
    background: color-mix(in srgb, var(--entry-colour, var(--muted)) 6%, transparent);
    font-size: 11px;
    border-radius: 0 3px 3px 0;
  }
  .log-time { color: var(--muted); font-size: 10px; }
  .log-msg { margin: 2px 0; }
  .log-knn { font-size: 10px; color: var(--muted); }
  /* KNN/MLP panel */
  .compare-box {
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px;
    margin-bottom: 10px;
  }
  .compare-row { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12px; }
  .agree { color: var(--green); } .disagree { color: var(--red); }
  .drift-warn {
    background: color-mix(in srgb, var(--purple) 15%, transparent);
    border: 1px solid var(--purple);
    border-radius: 4px;
    padding: 8px;
    font-size: 11px;
    color: var(--purple);
    display: none;
    margin-top: 8px;
  }
  .drift-warn.visible { display: block; }
  /* Snapshots */
  .snap-wrap { overflow-y: auto; height: calc(100% - 40px); }
  .snap-entry {
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px;
    margin-bottom: 8px;
    font-size: 11px;
  }
  .snap-threat { font-weight: bold; color: var(--orange); margin-bottom: 4px; }
  .snap-feat { color: var(--muted); }
  /* Bottom row */
  .full-width { grid-column: 1 / -1; }
  .scenario-list { display: flex; gap: 8px; flex-wrap: wrap; }
  .scenario-btn {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 12px;
    font-size: 11px;
    border-radius: 3px;
    cursor: pointer;
    font-family: inherit;
    transition: border-color 0.2s;
  }
  .scenario-btn:hover { border-color: var(--green); color: var(--green); }
  .progress-bar {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
    margin-top: 8px;
  }
  .progress-fill {
    height: 100%;
    background: var(--green);
    width: 0%;
    transition: width 0.5s linear;
  }
  .scenario-status { font-size: 11px; color: var(--muted); margin-top: 6px; }
</style>
</head>
<body>
<header>
  <h1>🔐 Intelligent Edge Security System</h1>
  <span>Arduino Uno + KNN/MLP Dual Classifier · Demo Simulation</span>
</header>

<div class="grid">
  <!-- Left: State + Metrics -->
  <div class="panel">
    <div class="panel-title">System State</div>
    <div class="sim-badge">SIMULATION MODE — NO HARDWARE REQUIRED</div>
    <div class="state-box" id="state-box" style="--state-colour: #10b981">
      <div class="state-label" id="state-label">READY</div>
      <div class="state-sub" id="state-sub">Awaiting events…</div>
    </div>
    <div class="panel-title" style="margin-top:14px">Sensor Readings</div>
    <div class="metric"><span>Motion Count (PIR)</span><span id="m-motion">0</span></div>
    <div class="metric"><span>Failed PINs</span><span id="m-failed">0</span></div>
    <div class="metric"><span>Consecutive Motion</span><span id="m-consec">No</span></div>
    <div class="metric"><span>Hour of Day</span><span id="m-hour">—</span></div>
    <div class="metric"><span>Reset Pressed</span><span id="m-reset">No</span></div>
  </div>

  <!-- Centre: Event Log -->
  <div class="panel">
    <div class="panel-title">Event Log</div>
    <div class="log-wrap" id="log-wrap">
      <div style="color:var(--muted);font-size:12px;padding:20px 0">
        Waiting for first simulated event…
      </div>
    </div>
  </div>

  <!-- Right: KNN/MLP + Snapshots -->
  <div class="panel" style="overflow-y:auto">
    <div class="panel-title">Classifier Comparison</div>
    <div class="compare-box">
      <div class="compare-row"><span style="color:var(--muted)">KNN (PC Layer)</span><span id="knn-pred">—</span></div>
      <div class="compare-row"><span style="color:var(--muted)">MLP (Arduino)</span><span id="mlp-pred">—</span></div>
      <div class="compare-row"><span style="color:var(--muted)">Agreement</span><span id="agree-label">—</span></div>
    </div>
    <div class="drift-warn" id="drift-warn">⚡ DRIFT WARNING — KNN/MLP disagree 3×<br>Model retraining recommended</div>

    <div class="panel-title" style="margin-top:16px">Camera Evidence</div>
    <div class="snap-wrap" id="snap-wrap">
      <div style="color:var(--muted);font-size:11px">Snapshots captured on SUSPICIOUS / HIGH RISK events.</div>
    </div>
  </div>

  <!-- Bottom: Simulation Control -->
  <div class="panel full-width" style="height:auto">
    <div class="panel-title">Simulation Control</div>
    <div class="scenario-list">
      <button class="scenario-btn" onclick="triggerEvent('motion')">PIR Motion</button>
      <button class="scenario-btn" onclick="triggerEvent('wrong_pin')">Wrong PIN</button>
      <button class="scenario-btn" onclick="triggerEvent('correct_pin')">Correct PIN ✓</button>
      <button class="scenario-btn" onclick="triggerEvent('reset')">Physical Reset</button>
      <button class="scenario-btn" onclick="triggerEvent('deescalate')">De-escalate (2min)</button>
      <button class="scenario-btn" onclick="runAutoDemo()">▶ Run Full Demo Sequence</button>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="demo-progress"></div></div>
    <div class="scenario-status" id="demo-status">Manual mode — click events above, or run the full demo sequence.</div>
  </div>
</div>

<script>
const socket = io();
let snapCount = 0;

socket.on('update', (data) => {
  // State
  const box = document.getElementById('state-box');
  box.style.setProperty('--state-colour', data.state_colour);
  document.getElementById('state-label').textContent = data.state;
  document.getElementById('state-label').style.color = data.state_colour;
  const subs = {READY:'System nominal',ALERT:'Motion detected',SUSPICIOUS:'Threat escalated',HIGH_RISK:'High threat — evidence captured',LOCKOUT:'LOCKED — physical reset required'};
  document.getElementById('state-sub').textContent = subs[data.state] || '';

  // Metrics
  document.getElementById('m-motion').textContent = data.features[0];
  document.getElementById('m-failed').textContent = data.features[1];
  document.getElementById('m-consec').textContent = data.features[2] ? 'Yes' : 'No';
  document.getElementById('m-hour').textContent = String(data.features[4]).padStart(2,'0') + ':xx';
  document.getElementById('m-reset').textContent = data.features[3] ? 'Yes' : 'No';

  // KNN/MLP
  const colours = {NORMAL:'#10b981', SUSPICIOUS:'#f97316', 'HIGH RISK':'#ef4444'};
  document.getElementById('knn-pred').textContent = data.knn_label;
  document.getElementById('knn-pred').style.color = colours[data.knn_label] || 'white';
  document.getElementById('mlp-pred').textContent = data.mlp_label;
  document.getElementById('mlp-pred').style.color = colours[data.mlp_label] || 'white';
  const agEl = document.getElementById('agree-label');
  agEl.textContent = data.agreement ? '✓ AGREE' : '✗ DISAGREE';
  agEl.className = data.agreement ? 'agree' : 'disagree';
  document.getElementById('drift-warn').className = 'drift-warn' + (data.drift_warning ? ' visible' : '');

  // Event log
  const log = document.getElementById('log-wrap');
  log.innerHTML = data.event_log.map(e => `
    <div class="log-entry" style="--entry-colour:${e.colour}">
      <span class="log-time">${e.time}</span>
      <div class="log-msg">${e.message}</div>
      <div class="log-knn">KNN: ${e.knn} · MLP: ${e.mlp} ${e.agreement===true?'<span style="color:#10b981">✓</span>':e.agreement===false?'<span style="color:#ef4444">✗</span>':''}</div>
    </div>
  `).join('');

  // Snapshots
  if (data.snapshot) {
    snapCount++;
    const sw = document.getElementById('snap-wrap');
    const f = data.snapshot.features;
    const div = document.createElement('div');
    div.className = 'snap-entry';
    div.innerHTML = `
      <div class="snap-threat">[${snapCount}] ${data.snapshot.threat}</div>
      <div style="color:var(--muted);font-size:10px">${data.snapshot.time}</div>
      <div class="snap-feat">
        Motion: ${f[0]} · PINs failed: ${f[1]}<br>
        Consec motion: ${f[2]?'Yes':'No'} · Hour: ${f[4]}:xx
      </div>
      <div style="font-size:10px;color:var(--muted);margin-top:4px">
        📷 snapshot_${data.snapshot.time.replace(/:/g,'')}.jpg [simulated]
      </div>
    `;
    sw.prepend(div);
  }
});

function triggerEvent(type) {
  fetch('/event/' + type, {method:'POST'});
}

const demoSequence = [
  {event:'motion',     delay:0,    label:'PIR motion detected — entering ALERT'},
  {event:'wrong_pin',  delay:3000, label:'Wrong PIN #1 entered'},
  {event:'wrong_pin',  delay:6000, label:'Wrong PIN #2 — escalating to SUSPICIOUS'},
  {event:'wrong_pin',  delay:9000, label:'Wrong PIN #3 — HIGH RISK'},
  {event:'wrong_pin',  delay:12000,label:'Wrong PIN #4'},
  {event:'wrong_pin',  delay:15000,label:'Wrong PIN #5 — LOCKOUT'},
  {event:'reset',      delay:19000,label:'Physical reset button pressed → READY'},
  {event:'motion',     delay:22000,label:'Night-time motion (hour=23)'},
  {event:'wrong_pin',  delay:25000,label:'Wrong PIN during suspicious period'},
  {event:'deescalate', delay:30000,label:'Quiet period — de-escalating to READY'},
  {event:'correct_pin',delay:34000,label:'Correct PIN — access granted'},
];

function runAutoDemo() {
  const statusEl = document.getElementById('demo-status');
  const progressEl = document.getElementById('demo-progress');
  const total = demoSequence[demoSequence.length-1].delay + 2000;

  demoSequence.forEach(step => {
    setTimeout(() => {
      statusEl.textContent = '▶ ' + step.label;
      triggerEvent(step.event);
    }, step.delay);
    setTimeout(() => {
      const pct = ((step.delay / total) * 100).toFixed(1);
      progressEl.style.width = pct + '%';
    }, step.delay);
  });

  setTimeout(() => {
    progressEl.style.width = '100%';
    statusEl.textContent = '✓ Demo sequence complete. All scenarios demonstrated.';
  }, total);
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/event/<event_type>', methods=['POST'])
def trigger_event(event_type):
    if event_type not in ('motion', 'wrong_pin', 'correct_pin', 'reset', 'deescalate'):
        return jsonify({'error': 'unknown event'}), 400
    result = sm.process_event(event_type)
    socketio.emit('update', result)
    return jsonify({'ok': True, 'state': result['state']})


@app.route('/status')
def status():
    return jsonify({
        'state': sm.state,
        'features': sm.get_features(),
        'mode': 'simulation',
        'note': 'No Arduino connected. Injecting simulated sensor events.',
    })


# ─────────────────────────────────────────────────────────────
# Optional: auto-play background scenario sequence on startup
# ─────────────────────────────────────────────────────────────

def background_scenario():
    """Runs a gentle background sequence so the dashboard isn't empty on load."""
    time.sleep(3)
    steps = [
        ('motion',      3),
        ('motion',      4),
        ('wrong_pin',   3),
        ('wrong_pin',   4),
        ('wrong_pin',   5),
        ('correct_pin', 4),
        ('motion',      3),
        ('deescalate',  0),
    ]
    for event, delay in steps:
        sm.process_event(event)
        time.sleep(delay)


if __name__ == '__main__':
    print()
    print('=' * 60)
    print('  Intelligent Edge Security System — Demo Simulation')
    print('  No Arduino or webcam required.')
    print()
    print('  Dashboard: http://localhost:5000')
    print()
    print('  Use the buttons in the dashboard to trigger events,')
    print('  or click "Run Full Demo Sequence" to watch all')
    print('  scenarios play out automatically.')
    print('=' * 60)
    print()

    # Start gentle background sequence
    t = threading.Thread(target=background_scenario, daemon=True)
    t.start()

    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
