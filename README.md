# PünktlichPills — Electronic Pill Dispenser Simulation

A Python-based simulation of a non-invasive electronic pill dispenser designed to reduce missed or accidental medication doses for elderly and chronically ill users. The project combines **event simulation** with **sensor signal processing** to model both the device's behavior and the reliability of its load-cell-based dispensing detection.

## Overview

The E-Pill Dispenser automates medication delivery on a fixed daily schedule and confirms each dose using a load-cell sensor in the pill collection compartment. This repo contains two simulations:

1. **Device Behavior Simulation** (`sim3-fin.py`) — an animated timeline of dispensing events, missed doses, alert activations, and battery decay.
2. **Signal Processing Simulation** — a study of how noise corrupts the load-cell sensor signal and how filtering techniques recover it.

## Target Use Case

Designed for elderly individuals with memory loss, caretakers, and patients managing chronic conditions such as hypertension or diabetes. The device is intended for residential or assisted-living environments (ECHO or Elder Care Homes) and requires no app connectivity or technical expertise to operate.

## Part 1: Device Behavior Simulation

`sim3-fin.py` models a full day of dispenser activity using an animated Matplotlib visualization with three synchronized panels.

**Simulation parameters:**
- 400-second simulated runtime, 4 scheduled doses per day
- 10-second dose window per event, 5-second resume-alert delay
- Configurable missed-dose indices to test alert behavior (default: doses #2 and #4 missed)
- Exponential battery decay model with low (20%) and critical (10%) thresholds

**Panels:**
| Panel | Shows |
|---|---|
| Medication Events Timeline | Dispensed, taken, and missed dose events over time |
| Multi-Modal Alert Activity | Buzzer, LED, and vibration intensity triggered on missed doses |
| Battery Level Monitoring | Exponential decay curve with threshold warnings |

**Tech stack:** `numpy`, `matplotlib` (with `matplotlib.animation`), `scipy.signal`

**Run it:**
```bash
python sim3-fin.py
```

## Part 2: Load-Cell Signal Processing

This simulation addresses a different problem: how to reliably detect a pill-drop event (a small ~5g weight change) from a noisy load-cell signal corrupted by Gaussian noise and 60Hz powerline interference.

**Filter pipeline evaluated:**
1. **Moving Average** — 5-point smoothing to suppress rapid noise
2. **60Hz Notch Filter** — removes powerline interference (IIR notch, Q=30)
3. **Bandpass Filter** — isolates the sensor's frequency band of interest
4. **Kalman Filter** — recursive Bayesian estimator adaptively tracking the true signal

**Results — Signal-to-Noise Ratio (SNR) improvement:**

| Filtering Method | SNR Before (dB) | SNR After (dB) |
|---|---|---|
| No Filtering (Raw Signal) | 1.2 | 1.2 |
| Notch Filter | 1.2 | 4.5 |
| Bandpass Filter | 1.2 | 7.8 |
| Kalman Filter | 1.2 | 11.3 |

The Kalman filter provided the largest SNR gain, confirming that adaptive estimation best isolates the pill-drop event from sensor and environmental noise. Combining smoothing, frequency-specific rejection, and adaptive filtering produces a reliable signal suitable for triggering dispensing confirmation and missed-dose logging.

## Future Work

- Port the filtering pipeline to real-time microcontroller firmware
- Implement adaptive filter tuning based on ambient environmental variability
- Integrate device behavior simulation with live sensor data streams
- Add unit tests for dose-scheduling and alert-triggering logic

## Author

Ahmed Mudassir Ashraf — Study Programme "Artificial Intelligence for Industrial Applications," OTH Amberg-Weiden
