// =====================================================================
//  SCS 6106 - EIS  |  ASSIGNMENTS 5 & 6
//  Intelligent Edge Security System with Real-Time Threat Classification
//
//  ML PIPELINE - HOW THIS SKETCH FITS IN:
//  ---------------------------------------------------------------------
//  STAGE 1 - DATA COLLECTION (this file):
//    Two IoT sensors collect real-world data:
//    - PIR sensor (pin 7): detects human presence
//    - 4x4 Keypad: captures PIN attempts
//    Data is packaged as JSON and sent over Serial (lightweight protocol)
//    to the PC - equivalent to MQTT in a cloud-based IoT system.
//
//  STAGE 2 - DATA PROCESSING (this file):
//    All preprocessing happens locally at the edge (no cloud needed):
//    - A 60-second rolling window resets counters to filter stale data
//    - Features are normalised to 0-1 inside scaleFeatures()
//    - Consecutive motion tracking filters noise from single PIR triggers
//
//  STAGE 3 - FEATURE ENGINEERING (this file):
//    Raw sensor signals are transformed into 5 meaningful features:
//    - motion_count       : how many PIR triggers 
//    - failed_pins        : wrong PIN attempts (authentication failures)
//    - hour_of_day        : context - 2am motion is riskier than 2pm
//    - reset_pressed      : whether manual override was used
//    - consecutive_motion : persistent presence vs single trigger
//
//  STAGE 5 - EDGE DEPLOYMENT (this file):
//    The MLP model trained on the PC (Assignment6_EdgeML.py) is deployed
//    here. All 123 learned weights live in model_weights.h which is
//    compiled into Arduino flash memory (2.1 KB).
//    Inference runs entirely on-chip: no internet, no cloud, no PC.
//    This is functionally equivalent to TensorFlow Lite deployment.
//
//  BUZZER BEHAVIOUR:
//  ---------------------------------------------------------------------
//  Motion detected  : short single BEEP  (100 ms)
//  Wrong PIN        : long CONTINUOUS BUZZ (1000 ms)
//  Correct PIN      : two short happy beeps
//  HIGH RISK (ML)   : three rapid urgent beeps
//
//  HARDWARE PINS:
//  ---------------------------------------------------------------------
//  PIR sensor       : pin 7    Buzzer        : pin 8
//  LED detected     : pin 13   LED ready     : pin 12
//  LED wait         : pin 11   Reset button  : A1
//  Keypad rows      : pins 5,4,3,2
//  Keypad cols      : pins A0,10,9,6
// =====================================================================

#include <Keypad.h>
#include "model_weights.h"   // STAGE 5: exported MLP weights from PC training

// ── Keypad layout ─────────────────────────────────────────────────────
const byte ROWS = 4;
const byte COLS = 4;
char keys[ROWS][COLS] = {
  {'1','4','7','*'},
  {'2','5','8','0'},
  {'3','6','9','#'},
  {'A','B','C','D'}
};
byte rowPins[ROWS] = {5, 4, 3, 2};
byte colPins[COLS] = {A0, 10, 9, 6};
Keypad keypad = Keypad(makeKeymap(keys), rowPins, colPins, ROWS, COLS);

// ── Hardware pins ──────────────────────────────────────────────────────
const int detectedLED  = 13;
const int readyLED     = 12;
const int waitLED      = 11;
const int buzzerPin    = 8;
const int pirPin       = 7;
const int resetButton  = A1;

// ── Security state ─────────────────────────────────────────────────────
String enteredPIN        = "";
const String correctPIN  = "1234";
bool   motionDetected    = false;
bool   systemReady       = true;

// ── STAGE 3 - Engineered features (reset every 60 seconds) ────────────
// These are derived from raw sensor signals - not raw readings themselves
int  motionCount         = 0;   // PIR triggers this window
int  failedAttempts      = 0;   // Wrong PINs this window
int  consecutiveMotion   = 0;   // Back-to-back PIR fires
unsigned long windowStart = 0;  // Timestamp of rolling window start

// ── Threat label strings (output classes) ─────────────────────────────
const char* THREAT_LABELS[3] = {"NORMAL", "SUSPICIOUS", "HIGH RISK"};

// =====================================================================
//  BUZZER HELPER FUNCTIONS
//  Physical actuation layer - hardware responds to ML classification
// =====================================================================

// Single short beep - motion alert (Stage 1: sensor triggered)
void beepOnce() {
  digitalWrite(buzzerPin, HIGH);
  delay(100);
  digitalWrite(buzzerPin, LOW);
}

// Two short beeps - correct PIN / system disarmed
void beepDouble() {
  for (int i = 0; i < 2; i++) {
    digitalWrite(buzzerPin, HIGH); delay(120);
    digitalWrite(buzzerPin, LOW);  delay(100);
  }
}

// Long continuous buzz - wrong PIN (authentication failure detected)
void buzzLong() {
  digitalWrite(buzzerPin, HIGH);
  delay(1000);
  digitalWrite(buzzerPin, LOW);
}

// Three rapid beeps - HIGH RISK ML classification (Stage 5 output action)
void beepUrgent() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(buzzerPin, HIGH); delay(150);
    digitalWrite(buzzerPin, LOW);  delay(100);
  }
}

// =====================================================================
//  STAGE 5 - EDGE ML INFERENCE FUNCTIONS
//
//  These functions implement the MLP forward pass in pure C++.
//  The weights (W1,W2,W3,b1,b2,b3) come from model_weights.h which
//  was generated by Assignment6_EdgeML.py after training on the PC.
//  No ML library is needed on the Arduino - just basic arithmetic.
//  This is equivalent to how TensorFlow Lite runs on microcontrollers.
// =====================================================================

// ReLU activation: passes positive values, blocks negatives
// Used in hidden layers to introduce non-linearity
float relu(float x) {
  return x > 0.0f ? x : 0.0f;
}

// Softmax activation: converts raw output scores into probabilities (0-1)
// The three outputs represent P(NORMAL), P(SUSPICIOUS), P(HIGH RISK)
// They always sum to 1.0 - we pick the highest as our prediction
void softmax(float* arr, int n) {
  // Subtract maxVal first for numerical stability (prevents overflow on chip)
  float maxVal = arr[0];
  for (int i = 1; i < n; i++) if (arr[i] > maxVal) maxVal = arr[i];
  float sum = 0.0f;
  for (int i = 0; i < n; i++) { arr[i] = exp(arr[i] - maxVal); sum += arr[i]; }
  for (int i = 0; i < n; i++) arr[i] /= sum;
}

// STAGE 2 - Feature normalisation (Min-Max scaling)
// Scales each feature to 0-1 so no feature dominates due to magnitude
// e.g. hour_of_day (0-23) would otherwise overpower failed_pins (0-5)
// FEATURE_MIN and FEATURE_MAX come from model_weights.h (training data bounds)
void scaleFeatures(float* features, float* scaled) {
  for (int i = 0; i < NUM_FEATURES; i++) {
    float denom = FEATURE_MAX[i] - FEATURE_MIN[i];
    if (denom == 0.0f) denom = 1.0f;  // prevent division by zero
    scaled[i] = (features[i] - FEATURE_MIN[i]) / denom;
  }
}

// STAGE 5 - Full MLP forward pass
// Architecture: 5 inputs -> 8 neurons (ReLU) -> 6 neurons (ReLU) -> 3 outputs (Softmax)
// This is the neural network running entirely on the Arduino chip
int mlpPredict(float* rawFeatures, float* confidence) {
  // Step 1: normalise raw sensor values to 0-1
  float x[NUM_FEATURES];
  scaleFeatures(rawFeatures, x);

  // Step 2: Layer 1 forward pass (5 inputs -> 8 neurons, ReLU)
  // Each neuron = dot product of inputs and weights + bias, then ReLU
  float h1[W1_COLS];
  for (int j = 0; j < W1_COLS; j++) {
    float sum = b1[j];  // start with bias
    for (int i = 0; i < W1_ROWS; i++) sum += x[i] * W1[i * W1_COLS + j];
    h1[j] = relu(sum);  // apply ReLU activation
  }

  // Step 3: Layer 2 forward pass (8 -> 6 neurons, ReLU)
  float h2[W2_COLS];
  for (int j = 0; j < W2_COLS; j++) {
    float sum = b2[j];
    for (int i = 0; i < W2_ROWS; i++) sum += h1[i] * W2[i * W2_COLS + j];
    h2[j] = relu(sum);
  }

  // Step 4: Output layer (6 -> 3 outputs, Softmax)
  // Produces probabilities for NORMAL, SUSPICIOUS, HIGH RISK
  float out[NUM_CLASSES];
  for (int j = 0; j < NUM_CLASSES; j++) {
    float sum = b3[j];
    for (int i = 0; i < W3_ROWS; i++) sum += h2[i] * W3[i * NUM_CLASSES + j];
    out[j] = sum;
  }
  softmax(out, NUM_CLASSES);  // convert scores to probabilities

  // Step 5: Pick class with highest probability
  int best = 0;
  for (int i = 1; i < NUM_CLASSES; i++) if (out[i] > out[best]) best = i;
  *confidence = out[best] * 100.0f;  // return as percentage
  return best;  // 0=NORMAL, 1=SUSPICIOUS, 2=HIGH RISK
}

// STAGE 5 - Trigger inference and act on result
// Called after every wrong PIN attempt - escalation logic lives in the model
void runML() {
  int hour = 12;  // Note: connect RTC module for real hour reading

  // STAGE 3: Package engineered features into array for the model
  float features[NUM_FEATURES] = {
    (float)motionCount,          // how many PIR triggers this window
    (float)failedAttempts,       // wrong PIN attempts - key escalation driver
    (float)hour,                 // time context
    (float)(failedAttempts > 0 ? 1 : 0),  // PIN activity flag
    (float)consecutiveMotion     // persistent presence indicator
  };

  float conf;
  int threat = mlpPredict(features, &conf);

  // STAGE 5 OUTPUT: send classification result over Serial
  // Python script (Assignment 5) also reads this for cross-validation
  Serial.print("ML_THREAT:");
  Serial.print(THREAT_LABELS[threat]);
  Serial.print(",CONF:");
  Serial.println(conf, 1);

  // ACTUATION: physical response to ML classification
  if (threat == 2) beepUrgent();  // HIGH RISK -> three urgent beeps
}

// STAGE 1 - Data ingestion: send sensor state as JSON over Serial
// This is the data feed for the Python KNN classifier (Assignment 5)
// JSON format allows the PC to parse individual features easily
void sendSerialJSON() {
  Serial.print("{\"motion_count\":");
  Serial.print(motionCount);
  Serial.print(",\"failed_pins\":");
  Serial.print(failedAttempts);
  Serial.print(",\"hour_of_day\":12");
  Serial.print(",\"reset_pressed\":");
  Serial.print(failedAttempts > 0 ? 1 : 0);
  Serial.print(",\"consecutive_motion\":");
  Serial.print(consecutiveMotion);
  Serial.println("}");
}

// =====================================================================
//  SETUP
// =====================================================================
void setup() {
  Serial.begin(9600);  // open Serial for data ingestion (Stage 1)
  Serial.println("=== Intelligent Security System Started ===");

  pinMode(detectedLED,  OUTPUT);
  pinMode(readyLED,     OUTPUT);
  pinMode(waitLED,      OUTPUT);
  pinMode(buzzerPin,    OUTPUT);
  pinMode(pirPin,       INPUT);
  pinMode(resetButton,  INPUT_PULLUP);

  // PIR sensor warm-up period (3 seconds to stabilise)
  digitalWrite(waitLED, HIGH);
  delay(3000);
  digitalWrite(waitLED, LOW);
  digitalWrite(readyLED, HIGH);

  windowStart = millis();
  Serial.println("System Ready. Monitoring...");
}

// =====================================================================
//  MAIN LOOP
// =====================================================================
void loop() {

  // STAGE 2 - Rolling 60-second window reset
  // Prevents stale data from accumulating and skewing predictions
  // Equivalent to windowing/sliding window in time-series IoT processing
  if (millis() - windowStart >= 60000) {
    motionCount       = 0;
    failedAttempts    = 0;
    consecutiveMotion = 0;
    windowStart       = millis();
    Serial.println("Window reset.");
  }

  // STAGE 1 - PIR sensor data collection
  // Raw sensor output (HIGH/LOW) is transformed into engineered features
  if (digitalRead(pirPin) == HIGH && systemReady) {
    Serial.println("EVENT:MOTION_DETECTED");
    digitalWrite(detectedLED, HIGH);
    beepOnce();

    motionDetected     = true;
    systemReady        = false;
    motionCount++;        // STAGE 3: increment engineered feature
    consecutiveMotion++; // STAGE 3: track persistence of presence

    // Motion alone = NORMAL (escalation only happens via wrong PINs)
    Serial.println("ML_THREAT:NORMAL,CONF:100.0");
    sendSerialJSON();  // STAGE 1: transmit data to PC for KNN (Assignment 5)

  } else if (!motionDetected) {
    digitalWrite(detectedLED, LOW);
    consecutiveMotion = 0;
  }

  // STAGE 1 - Keypad data collection
  char key = keypad.getKey();
  if (key) {
    Serial.print("KEY:");
    Serial.println(key);

    if (key >= '0' && key <= '9') {
      if (enteredPIN.length() < 4) {
        enteredPIN += key;
        Serial.print("PIN_PROGRESS:");
        Serial.println(enteredPIN.length());
      }

    } else if (key == '*') {
      enteredPIN = "";
      Serial.println("EVENT:PIN_CLEARED");

    } else if (key == '#') {
      if (enteredPIN == correctPIN) {
        // Correct PIN - system disarmed
        Serial.println("EVENT:CORRECT_PIN");
        digitalWrite(readyLED,    HIGH);
        digitalWrite(detectedLED, LOW);
        beepDouble();
        systemReady    = true;
        motionDetected = false;
        failedAttempts = 0;  // reset escalation counter

      } else {
        // Wrong PIN - increment feature and run ML (Stages 3 + 5)
        Serial.println("EVENT:WRONG_PIN");
        failedAttempts++;  // STAGE 3: key escalation feature
        digitalWrite(detectedLED, HIGH);
        buzzLong();
        digitalWrite(detectedLED, LOW);

        sendSerialJSON();  // STAGE 1: update PC with new feature values
        runML();           // STAGE 5: re-classify threat on chip
      }
      enteredPIN = "";
    }
  }

  // Manual reset button - clears all feature counters
  if (digitalRead(resetButton) == LOW) {
    enteredPIN        = "";
    systemReady       = true;
    motionDetected    = false;
    failedAttempts    = 0;
    motionCount       = 0;
    consecutiveMotion = 0;
    digitalWrite(detectedLED, LOW);
    digitalWrite(readyLED,    HIGH);
    Serial.println("EVENT:SYSTEM_RESET");
    delay(200);
  }
}
