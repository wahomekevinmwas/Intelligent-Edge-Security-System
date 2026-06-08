# Screenshots Strategy

## What to Include in the Repo

Your MSc report already contains runtime screenshots. Here's how to organise them for maximum GitHub impact.

---

## Required Screenshots (from your existing report images)

| Filename | Source in Report | What It Shows |
|---|---|---|
| `serial_monitor.png` | Figure 6 | Arduino Serial Monitor — JSON output stream |
| `dashboard_high_risk.png` | Figure 7 | Flask dashboard — HIGH RISK with KNN/MLP comparison |
| `camera_suspicious.png` | Figure 8 | Camera snapshot — SUSPICIOUS overlay |
| `camera_normal.png` | Figure 9 | Camera snapshot — NORMAL activity |

**How to extract from the report PDF:**  
Open the report PDF → zoom to 200% on each figure → screenshot the figure only (crop out the caption) → save as PNG at the filename above.

---

## Recommended Additional Screenshots

These improve the GitHub repo significantly and take 5 minutes to capture with the simulation running:

| Filename | How to Capture |
|---|---|
| `dashboard_ready.png` | Launch simulation → screenshot before any events |
| `dashboard_lockout.png` | Click "Wrong PIN" 5 times → screenshot LOCKOUT state |
| `dashboard_drift_warning.png` | In simulation, trigger drift manually (run full demo) |
| `knn_mlp_comparison.png` | Close crop of just the classifier comparison panel |

---

## Demo Video (Strongly Recommended)

A 2–3 minute screen recording showing the full demo sequence is more persuasive than 10 screenshots for a hardware project.

**Suggested recording flow:**
1. Start: Show physical Arduino hardware (5 seconds)
2. Open Serial Monitor → walk past PIR → show JSON appearing (20 seconds)
3. Switch to browser dashboard → wrong PIN sequence → LOCKOUT (40 seconds)
4. Reset button → return to READY (10 seconds)
5. Run simulation script — show it works without hardware (30 seconds)

**Free tools:** OBS Studio (Windows/Linux/Mac), QuickTime (Mac), Xbox Game Bar (Win+G on Windows)

**Upload to:** YouTube (unlisted) → paste link in README demo video section

---

## README Image Embedding

Once screenshots are in `docs/screenshots/`, the README table renders them automatically:

```markdown
| Arduino Serial Monitor | Flask Dashboard — HIGH RISK |
|---|---|
| ![Serial](docs/screenshots/serial_monitor.png) | ![Dashboard](docs/screenshots/dashboard_high_risk.png) |
```

GitHub renders images up to 1MB inline. Keep screenshots under 500KB — use PNG for UI/dashboard, JPEG for camera snapshots.
