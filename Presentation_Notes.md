# SCS 6106 – EIS | Presentation Notes
## Assignments 5 & 6 — Intelligent Security System with Edge ML

---

## 🧭 HOW TO PRESENT (flow to follow)

1. Start with the big picture (what problem you are solving)
2. Walk through Assignment 5 (KNN)
3. Walk through Assignment 6 (Edge ML)
4. Show the physical demo
5. Answer questions using the Q&A section below

---

## 🔴 THE PROBLEM (open with this)

> "Traditional security systems are reactive — they only alert you
> after something has already happened. Our system is intelligent.
> It learns from past events, classifies threats in real time, and
> takes action automatically — even without internet or cloud."

Real-life use cases:
- Home or office intruder detection
- Campus access control
- Server room security
- Unattended warehouse monitoring

---

## 📌 SYSTEM OVERVIEW

The system has three layers working together:

| Layer | Component | Role |
|-------|-----------|------|
| **Sensing** | PIR sensor, Keypad | Collect real-world data |
| **Edge Device** | Arduino | Run ML inference on-chip |
| **Intelligence** | KNN + MLP models | Classify threat level |
| **Evidence** | Laptop camera | Capture intruder snapshots |

---

## 📘 ASSIGNMENT 5 — KNN Threat Classifier (x/10)

### What it is
K-Nearest Neighbours is a simple but powerful ML algorithm.
When a new security event occurs, it asks:
> "Which past events look most like this one? And what were those labelled as?"

### Features used (what the Arduino measures)
| Feature | Meaning |
|---------|---------|
| motion_count | How many times PIR fired in the last 60 seconds |
| failed_pins | Wrong PIN attempts in the last 60 seconds |
| hour_of_day | Time of day (0–23) — late night is riskier |
| reset_pressed | Was the manual reset used? |
| consecutive_motion | PIR triggers one after another (persistent intruder) |

### Static data
- 30 manually labelled historical events used for training
- Three classes: NORMAL 🟢, SUSPICIOUS 🟡, HIGH RISK 🔴

### Real-time data
- Arduino sends sensor readings as JSON over USB Serial every time motion is detected or a wrong PIN is entered
- Python reads these in real time and classifies instantly

### The algorithm (built from scratch — no shortcuts)
1. Measure the distance from the new event to all 30 training samples
2. Pick the 3 closest ones (K=3)
3. Take a majority vote of their labels
4. That vote is the prediction

### What happens with the result
- NORMAL → camera stays idle
- SUSPICIOUS → camera activates, live feed shown
- HIGH RISK → camera captures and saves a timestamped photo (evidence)

### Model accuracy
- Evaluated using Leave-One-Out Cross-Validation
- **93.3% accuracy** on the training dataset

### Where is the data saved?
> "We save snapshots of detected intruders in a local folder called
> `snapshots/` on the PC running the Python script. Each file is
> timestamped and labelled with the threat level."

---

## 📗 ASSIGNMENT 6 — Edge ML Deployment (x/20)

### What "edge" means
> "Instead of sending data to the cloud for a server to analyse,
> the intelligence runs directly on the Arduino chip itself.
> This means it works with no internet, no server, no latency."

### The model: MLP (Multi-Layer Perceptron)
A neural network — more powerful than KNN.

```
INPUT (5 features)
     ↓
HIDDEN LAYER 1  → 8 neurons  → ReLU activation
     ↓
HIDDEN LAYER 2  → 6 neurons  → ReLU activation
     ↓
OUTPUT LAYER    → 3 neurons  → Softmax (probabilities)
     ↓
PREDICTION: NORMAL / SUSPICIOUS / HIGH RISK
```

Total learnable parameters: **123 weights and biases**

### Three stages

**Stage 1 — Train on PC**
- Use scikit-learn's MLPClassifier
- 36 labelled training samples
- Achieved 100% training and test accuracy

**Stage 2 — Export weights**
- Every learned weight and bias is written into `model_weights.h`
- This is a plain C header file — just numbers in arrays
- This file is compiled directly into the Arduino's flash memory

**Stage 3 — Run on Arduino (edge)**
- The Arduino sketch includes `model_weights.h`
- It performs the forward pass itself using basic C math:
  multiply → add bias → ReLU → repeat → softmax
- No internet, no Python, no sklearn on the chip

### What "deploying to the edge" means (say this clearly)
> "We trained the model on a powerful laptop. But once training is
> done, the laptop is no longer needed. The model lives inside the
> Arduino's flash memory and makes decisions in milliseconds,
> completely standalone. That is edge deployment."

### Where are the weights saved?
> "The trained model weights are saved in a file called
> `model_weights.h` — a C header file that is compiled directly
> into the Arduino's program memory (flash storage). The Arduino
> reads these numbers from its own chip, not from any external source."

### Edge verification
- We proved the Arduino's math produces identical predictions to sklearn
- All 10 test cases matched 100% ✅

---

## 🔊 BUZZER BEHAVIOUR (explain this during physical demo)

| Event | Sound | Why |
|-------|-------|-----|
| Motion detected | One short beep (100ms) | Quick alert — someone is there |
| Wrong PIN entered | Long continuous buzz (1 second) | Alarm — unauthorised attempt |
| Correct PIN | Two short happy beeps | Confirmation — system disarmed |
| HIGH RISK (ML) | Three rapid urgent beeps | Critical — escalated threat |

---

## 📷 CAMERA INTEGRATION

- Uses the laptop's built-in webcam (OpenCV)
- Runs in a background thread — never slows down the serial reading
- Overlays the threat level and timestamp directly on the video feed
- Saves a `.jpg` snapshot automatically when threat = HIGH RISK
- Snapshot filename format: `ALERT_HIGH RISK_20250413_142310.jpg`

This is what makes the system practical for real-world deployment —
it creates timestamped photographic evidence automatically.

---

## 🗂️ FILES IN THIS PROJECT

| File | Purpose |
|------|---------|
| `SecuritySystem_Final.ino` | Arduino sketch — runs on the chip |
| `model_weights.h` | Exported MLP weights — included by the .ino |
| `Assignment5_KNN_Camera.py` | PC script — KNN + camera (Assignment 5) |
| `Assignment6_EdgeML.py` | PC script — trains MLP, exports weights (Assignment 6) |

---

## ▶️ HOW TO RUN (for demo day)

### Step 1 — Upload Arduino sketch
1. Put `SecuritySystem_Final.ino` and `model_weights.h` in the same folder
2. Open `SecuritySystem_Final.ino` in Arduino IDE
3. Tools → Board → your Arduino model
4. Tools → Port → COM4
5. Click Upload ✅

### Step 2 — Generate fresh model weights (Assignment 6)
```
python Assignment6_EdgeML.py
```
This trains the MLP, prints accuracy, verifies edge inference.

### Step 3 — Start the live classifier (Assignment 5)
```
python Assignment5_KNN_Camera.py
```
This opens the camera and starts reading from the Arduino.

### Step 4 — Trigger events on the Arduino
- Walk past the PIR sensor → beep + KNN classifies
- Enter wrong PIN (e.g. 9999 #) → buzz + threat level updates
- Enter correct PIN (1234 #) → double beep + system resets
- Press reset button → system returns to ready state

---

## ❓ LIKELY EXAM / PRESENTATION QUESTIONS

**Q: Why KNN and not something simpler like a threshold?**
> "A threshold only checks one feature at a time. KNN considers all five
> features together, so it understands context. Motion at 2am with
> three failed PINs is very different from motion at 10am with no
> failed PINs — KNN captures that difference."

**Q: Why retrain with MLP for Assignment 6 instead of just using the KNN?**
> "KNN cannot be deployed to the edge easily — at inference time it
> needs to store and search all training data, which is too much memory
> for an Arduino. An MLP, once trained, compresses all knowledge into
> just 123 numbers. That fits in Arduino flash memory easily."

**Q: How is this different from a normal alarm system?**
> "A normal alarm just checks if motion equals yes or no. Our system
> looks at patterns — how often motion occurs, what time it is, how
> many failed PINs have been entered — and makes an intelligent judgment.
> It can distinguish between a family member coming home late versus
> a potential intruder trying multiple PINs."

**Q: What is the accuracy of your model?**
> "KNN achieved 93.3% accuracy on leave-one-out cross-validation.
> The MLP achieved 100% on our dataset. In a production system we
> would collect more real-world data to make those numbers even more
> meaningful."

**Q: Where is the ML model stored?**
> "The KNN model's training data lives in the Python script on the PC.
> For the MLP (Assignment 6), the trained weights are stored in
> `model_weights.h` which is compiled into the Arduino's flash memory —
> that is what makes it a true edge deployment."

**Q: What is ReLU / Softmax?**
> "ReLU is an activation function — it introduces non-linearity by
> simply passing positive values through and blocking negatives.
> Softmax converts the final layer's raw scores into probabilities
> that sum to 1, so we can say 'there is a 96% chance this is HIGH RISK'."

**Q: Could this work without a laptop?**
> "Assignment 6 is already designed for that. The MLP runs entirely
> on the Arduino. For the camera, the next step would be adding an
> ESP32-CAM module, which has a built-in camera and Wi-Fi and can
> run similar inference on-chip."

---

## 💡 MINI PROJECT ANGLE (end your presentation with this)

> "This system demonstrates the complete pipeline of an intelligent
> IoT security solution — from raw sensor data, through ML classification,
> to physical actuation and photographic evidence — all running at the
> edge without any cloud dependency. With an ESP32-CAM replacing the
> laptop camera, and a GSM module for SMS alerts, this becomes a
> fully deployable, standalone smart security system for homes,
> offices, or any unmanned facility."

---

*Good luck with your presentation! You've got this. 🚀*
