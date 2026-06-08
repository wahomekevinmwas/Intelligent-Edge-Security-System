# Hardware Wiring Guide

## Component Connection Reference

### Arduino Uno Pin Map

```
                         ┌──────────────────┐
                         │   ARDUINO UNO    │
                         │                  │
           Keypad Row 1 ─┤ D2           A0 ├─ Keypad Col C4
           Keypad Row 2 ─┤ D3           A1 ├─ Reset Button → GND
           Keypad Row 3 ─┤ D4           A2 ├─ (free)
           Keypad Row 4 ─┤ D5           A3 ├─ (free)
        Keypad Col C1  ─┤ D6            5V ├─ PIR VCC
            PIR Signal ─┤ D7           GND ├─ Common Ground
                Buzzer ─┤ D8           GND ├─ Common Ground
         Keypad Col C2 ─┤ D9           GND ├─ Common Ground
         Keypad Col C3 ─┤ D10           3V ├─ (free)
            Amber LED  ─┤ D11          SCL ├─ (free)
            Green LED  ─┤ D12          SDA ├─ (free)
              Red LED  ─┤ D13          Vin ├─ (free)
                         │                  │
                         │    [RESET]  [USB]│──── PC (9600 baud)
                         └──────────────────┘
```

---

## LED Wiring (all three identical)

```
Arduino Pin ──────── 220Ω resistor ──────── LED Anode (+)
                                            LED Cathode (−) ──── GND

Pin 11: Amber LED
Pin 12: Green LED  
Pin 13: Red LED
```

> **Why 220Ω?** At 5V, a typical LED drops ~2V. (5V - 2V) / 220Ω ≈ 13.6mA — safely within the Arduino's 40mA per-pin limit and the LED's forward current rating.

---

## PIR Sensor (HC-SR501)

```
HC-SR501 rear view (dome facing away):

┌─────────────────────┐
│  [POT1]    [POT2]   │  ← Sensitivity (left) and Delay (right) potentiometers
│                     │
│  OUT   GND   VCC    │  ← Three pins at bottom
└─────────────────────┘
    │     │     │
    │     │     └──── Arduino 5V
    │     └────────── Arduino GND
    └──────────────── Arduino D7
```

**Potentiometer settings for demo/testing:**
- POT1 (Sensitivity): Turn fully clockwise for maximum range (~7m)
- POT2 (Delay): Turn fully counter-clockwise for minimum retrigger delay (~3 seconds)

**Trigger mode jumper:** Set to `H` (re-trigger mode) — PIR keeps output HIGH while motion is detected.

---

## 4×4 Matrix Keypad

```
Keypad pinout (left to right, facing the keys):

Pin: 1   2   3   4   5   6   7   8
     R1  R2  R3  R4  C1  C2  C3  C4
     │   │   │   │   │   │   │   │
     D2  D3  D4  D5  D6  D9  D10 A0
```

> The Keypad library handles row scanning automatically. Do not rearrange the pin order without updating `KEYPAD_ROWS`, `KEYPAD_COLS`, and `rowPins[]` / `colPins[]` in the `.ino` file.

---

## Reset Button

```
Arduino A1 ───────── [Button] ───────── GND
```

- `INPUT_PULLUP` is set in firmware — no external resistor needed
- Pin reads HIGH normally, LOW when button is pressed
- Only way to exit LOCKOUT state
- Also available for landlord "acknowledge" actions in SUSPICIOUS state

---

## Buzzer (Active)

```
Arduino D8 ──────── Buzzer (+) positive leg
GND        ──────── Buzzer (−) negative leg
```

> Use an **active** buzzer (has internal oscillator — makes sound on DC voltage).  
> A **passive** buzzer requires a PWM frequency signal — it will not work with `tone()` in the same way.

---

## Power and Ground Rail

All components share the Arduino's GND. If using a breadboard, establish a common ground rail and connect it to any Arduino GND pin. The Arduino is powered via USB from the PC (which also provides the serial connection).

**Do not power external components from the Arduino's 5V rail beyond:**
- 1× HC-SR501 (~65mA peak)
- 3× LEDs (~14mA each at 220Ω)
- 1× Buzzer (~30mA)

Total draw ≈ 150mA — within the Arduino's USB-powered limit of ~400mA.

---

## Full Assembly Checklist

- [ ] Arduino Uno connected to PC via USB-B cable
- [ ] All three LED anodes through 220Ω resistors to D11, D12, D13
- [ ] All three LED cathodes to GND rail
- [ ] PIR VCC → 5V, GND → GND, OUT → D7
- [ ] Keypad rows → D2, D3, D4, D5
- [ ] Keypad columns → D6, D9, D10, A0
- [ ] Buzzer (+) → D8, (-) → GND
- [ ] Reset button → A1 and GND (no resistor needed)
- [ ] `SecuritySystem_Final.ino` uploaded (check port in Arduino IDE)
- [ ] Serial Monitor at 9600 baud shows JSON output
- [ ] Python `config.py` serial port matches Arduino COM/tty port

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| No JSON on Serial Monitor | Wrong COM port selected | Check Device Manager (Windows) or `ls /dev/tty*` (Linux/Mac) |
| PIR triggers constantly | Sensitivity too high, or fluorescent light interference | Reduce POT1; shield PIR from direct light |
| PIR never triggers | Delay pot too high, or wrong pin | Reduce POT2; verify OUT → D7 |
| Keypad not responding | Column/row swap | Re-check pin order against keymap in `.ino` |
| Dashboard shows no data | Wrong serial port in `config.py` | Set `SERIAL_PORT` to match Arduino port |
| Camera not opening | Wrong camera index | Try `cv2.VideoCapture(0)`, `(1)`, `(2)` in sequence |
| MLP always returns 0 | PROGMEM read error | Verify `#include <avr/pgmspace.h>` and `pgm_read_float()` calls |
