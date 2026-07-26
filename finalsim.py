import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, lfilter # I changed filtfilt to lfilter for live processing
from matplotlib.animation import FuncAnimation
from collections import deque # to store a sliding window of data
import random # For random pill events

# --- Simulation Parameters ---
# These will now control the live simulation speed and duration
SIMULATION_DURATION_SECONDS = 60 # Total time the simulation will run
SAMPLING_RATE = 200 # Hz (original code effectively uses 100 Hz due to 1000 points in 10s)
TIME_STEP = 1.0 / SAMPLING_RATE # Time increment per frame
SIM_INTERVAL_MS = TIME_STEP * 1000 # Milliseconds per frame for FuncAnimation
MAX_DISPLAY_POINTS = SAMPLING_RATE * 10 # Display last 10 seconds of data

# --- Initial Load Cell Configuration ---
BASE_WEIGHT = 100.0  # Constant 50g weight for an empty compartment
PILL_WEIGHT = 5.0   # Weight change per pill 

# --- Compartment Status Thresholds (Adjust these based on expected filtered signal) ---
# These thresholds apply to the *filtered* signal (preferably Kalman output).
PILL_ADDED_THRESHOLD_DELTA = PILL_WEIGHT * 0.5 # A significant increase to detect a pill being added
PILL_REMOVED_THRESHOLD_DELTA = -PILL_WEIGHT * 0.5 # A significant decrease to detect a pill being removed

# Absolute load thresholds for overall compartment status
COMPARTMENT_EMPTY_THRESHOLD = BASE_WEIGHT + (PILL_WEIGHT * 0.1) # Just above empty weight
COMPARTMENT_LOW_THRESHOLD = BASE_WEIGHT + (PILL_WEIGHT * 5)     # E.g., less than 5 pills
COMPARTMENT_FULL_THRESHOLD = BASE_WEIGHT + (PILL_WEIGHT * 20)    # E.g., more than 20 pills

# --- Global Variables for Live Simulation State ---
current_true_weight = BASE_WEIGHT # Starts empty
global_time_s = 0.0

# Initial event tracking for the first simulated dispense
initial_dispense_triggered = False

# For random events
next_random_event_time = 0.0 # Will be set after initial dispense

# Use deques for efficient appending and popping of old data
time_data = deque(maxlen=MAX_DISPLAY_POINTS)
raw_signal_data = deque(maxlen=MAX_DISPLAY_POINTS)
smoothed_signal_data = deque(maxlen=MAX_DISPLAY_POINTS)
notched_signal_data = deque(maxlen=MAX_DISPLAY_POINTS)
kalman_filtered_data = deque(maxlen=MAX_DISPLAY_POINTS)

current_compartment_status = "UNKNOWN"

# --- Helper function to get initial conditions for lfilter ---
# scipy.signal.lfilter_zi is not directly exposed but used internally by filtfilt sometimes.
# For a standard IIR filter, we need to manually manage the state.
# A simpler way to get zi is often to filter a few zero samples.
def lfilter_zi(b, a):
    # This approximates the initial conditions for lfilter.
    # For a stable IIR filter, filtering zeros for a while gives a good starting zi.
    # Or, one can analytically calculate it. For simplicity, filtering some zeros.
    return np.zeros(max(len(a), len(b)) - 1)

# --- Filter Coefficients (Calculated once) ---
# 60Hz Notch Filter - second filter (calculated once, outside the loop)
fs = SAMPLING_RATE # Sampling frequency
f0 = 60.0  # Frequency to remove
Q = 30.0   # Quality factor
b_notch, a_notch = iirnotch(f0, Q, fs)

# --- Filter States for Live Processing ---
# For IIR filters (like Notch), `lfilter` with `zi` maintains state
# `filtfilt` is for offline processing, `lfilter` is better for real-time
zi_notch = lfilter_zi(b_notch, a_notch) * 0.0 # Initialize state for notch filter

# Kalman Filter state variables
# These will be updated in each step of the live simulation
kalman_x_hat = BASE_WEIGHT # Initial estimate of the state (current load)
kalman_P = 1.0 # Initial estimate error covariance
kalman_process_variance = 1e-5 # From your original code
kalman_measurement_variance = 1.0 # From your original code

# --- Your original filter functions, adapted for single sample processing ---

# 1. Moving Average Filter - adapted for live (stateful)
# We'll maintain a window of recent raw samples and apply MA to it.
ma_window = deque(maxlen=5) # Window size from original code

def moving_average_live(sample):
    ma_window.append(sample)
    # Ensure window is full before returning a meaningful average
    if len(ma_window) < ma_window.maxlen:
        # If window not full, just return the last sample or average of available
        return np.mean(ma_window) if ma_window else sample
    return np.mean(ma_window)

# 2. 60Hz Notch Filter - adapted for live using lfilter
def notch_filter_live(sample):
    global zi_notch
    # lfilter returns (output_array, final_state)
    filtered_sample, zi_notch_new = lfilter(b_notch, a_notch, [sample], zi=zi_notch)
    zi_notch = zi_notch_new # Update the state for the next call
    return filtered_sample[0]

# 3. Kalman Filter - adapted for live (stateful)
def kalman_filter_live(measurement):
    global kalman_x_hat, kalman_P

    # Prediction
    x_hat_minus = kalman_x_hat
    P_minus = kalman_P + kalman_process_variance

    # Update
    K = P_minus / (P_minus + kalman_measurement_variance)
    kalman_x_hat = x_hat_minus + K * (measurement - x_hat_minus)
    kalman_P = (1 - K) * P_minus

    return kalman_x_hat

# --- Plotting Setup ---
fig, ax1 = plt.subplots(figsize=(14, 7))
ax2 = ax1.twinx() # For the status text

# Lines for the plots
line_raw, = ax1.plot([], [], 'gray', alpha=0.4, label='Raw Signal')
line_smoothed, = ax1.plot([], [], 'blue', linestyle='--', label='Moving Average')
line_notched, = ax1.plot([], [], 'orange', linestyle='--', label='Notch Filtered')
line_kalman, = ax1.plot([], [], 'green', linewidth=2, label='Kalman Filter Output')

# Threshold lines (use base weight + an offset for thresholds relative to pills)
line_comp_empty_threshold = ax1.axhline(y=COMPARTMENT_EMPTY_THRESHOLD, color='red', linestyle=':', linewidth=1, label='Empty Threshold')
line_comp_low_threshold = ax1.axhline(y=COMPARTMENT_LOW_THRESHOLD, color='orange', linestyle=':', linewidth=1, label='Low Quantity Threshold')
line_comp_full_threshold = ax1.axhline(y=COMPARTMENT_FULL_THRESHOLD, color='purple', linestyle=':', linewidth=1, label='Full Threshold')


# Text annotation for status
status_text = ax2.text(0.98, 0.95, '', transform=ax2.transAxes,
                       fontsize=14, color='black', ha='right', va='top',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.8))

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Load Cell Output (g)")
ax1.set_title("Pill Dispenser Load Monitoring with Filtered Signals")
ax1.grid(True)
ax1.legend(loc='upper left')
ax2.set_ylabel("Status")
ax2.set_yticks([]) # Hide y-axis for the status

# --- Update Function for FuncAnimation ---
def update(frame):
    global global_time_s, current_true_weight, initial_dispense_triggered
    global current_compartment_status, next_random_event_time

    global_time_s = frame * TIME_STEP

    # --- Simulate pill dispensing/adding events dynamically ---
    # Initial fill of the dispenser (e.g., 10 pills added at start)
    if global_time_s == 0.0:
        current_true_weight += (10 * PILL_WEIGHT)
        print(f"[{global_time_s:.2f}s] SIMULATION: Dispenser filled with 10 pills. True Weight: {current_true_weight:.2f}g")
        next_random_event_time = global_time_s + random.uniform(5, 10) # Schedule first random event

    # Simulate a pill being dispensed at 5 seconds, lasting for 1 second
    if global_time_s >= 5.0 and global_time_s < 6.0 and not initial_dispense_triggered:
        if current_true_weight >= (BASE_WEIGHT + PILL_WEIGHT): # Ensure there's a pill to dispense
            current_true_weight -= PILL_WEIGHT # Weight drops by one pill
            print(f"[{global_time_s:.2f}s] SIMULATION: Pill Dispensing Event. True Weight: {current_true_weight:.2f}g")
        else:
            print(f"[{global_time_s:.2f}s] SIMULATION: Attempted dispense, but compartment is empty!")
        initial_dispense_triggered = True # Mark as triggered to avoid repeated changes

    # Random pill removal/addition after initial dispense
    if global_time_s >= next_random_event_time:
        event_type = random.choice(['add', 'remove'])
        if event_type == 'remove':
            if current_true_weight > BASE_WEIGHT: # Ensure there's something to remove
                current_true_weight -= PILL_WEIGHT
                print(f"[{global_time_s:.2f}s] SIMULATION: Random Pill Removed. True Weight: {current_true_weight:.2f}g")
        else: # 'add'
            # Prevent endless filling, e.g., max 25 pills
            if current_true_weight < (BASE_WEIGHT + 25 * PILL_WEIGHT):
                current_true_weight += PILL_WEIGHT
                print(f"[{global_time_s:.2f}s] SIMULATION: Random Pill Added. True Weight: {current_true_weight:.2f}g")
        next_random_event_time = global_time_s + random.uniform(5, 15) # Schedule next random event

    # Simulate noisy load cell data (from your original code)
    noise = np.random.normal(0, 1) # Single sample of white noise
    powerline_noise = 2 * np.sin(2 * np.pi * 60 * global_time_s) # Single sample of 60Hz interference
    raw_sample = current_true_weight + noise + powerline_noise

    # --- Append new data to deques ---
    time_data.append(global_time_s)
    raw_signal_data.append(raw_sample)

    # --- Apply Filters Live ---
    smoothed_sample = moving_average_live(raw_sample)
    smoothed_signal_data.append(smoothed_sample)

    notched_sample = notch_filter_live(smoothed_sample)
    notched_signal_data.append(notched_sample)

    kalman_filtered_sample = kalman_filter_live(notched_sample)
    kalman_filtered_data.append(kalman_filtered_sample)

    # --- Compartment Status Detection (using Kalman output for robustness) ---
    # This logic detects changes and reports overall status
    if len(kalman_filtered_data) > 1:
        last_load = kalman_filtered_data[-2]
        current_load = kalman_filtered_data[-1]
        load_change = current_load - last_load

        # Detect discrete events (pill added/removed) based on change
        if load_change > PILL_ADDED_THRESHOLD_DELTA:
            current_compartment_status = "PILL ADDED"
            status_color = 'green'
        elif load_change < PILL_REMOVED_THRESHOLD_DELTA:
            current_compartment_status = "PILL REMOVED"
            status_color = 'orange'
        else:
            # If no discrete event, check overall quantity thresholds
            if current_load < COMPARTMENT_EMPTY_THRESHOLD:
                current_compartment_status = "COMPARTMENT EMPTY"
                status_color = 'red'
            elif current_load < COMPARTMENT_LOW_THRESHOLD:
                current_compartment_status = "LOW QUANTITY"
                status_color = 'darkorange'
            elif current_load >= COMPARTMENT_FULL_THRESHOLD:
                current_compartment_status = "COMPARTMENT FULL"
                status_color = 'darkgreen'
            else:
                current_compartment_status = "STABLE / SUFFICIENT"
                status_color = 'blue'

        # Update text annotation
        # Estimate number of pills for display
        estimated_pills = max(0, round((current_load - BASE_WEIGHT) / PILL_WEIGHT))
        status_text.set_text(f"Status: {current_compartment_status}\nEstimated Pills: {estimated_pills}\nTime: {global_time_s:.2f}s")
        status_text.set_color(status_color)


    # --- Update Plot Data ---
    line_raw.set_data(time_data, raw_signal_data)
    line_smoothed.set_data(time_data, smoothed_signal_data)
    line_notched.set_data(time_data, notched_signal_data)
    line_kalman.set_data(time_data, kalman_filtered_data)

    # Adjust x-axis limits dynamically for sliding window
    ax1.set_xlim(time_data[0], time_data[0] + MAX_DISPLAY_POINTS * TIME_STEP)
    # Adjust y-axis limits dynamically based on current max/min in the window,
    # or keep fixed for better comparison of events
    # For now, keeping a wider fixed range to see events
    ax1.set_ylim(BASE_WEIGHT - (PILL_WEIGHT * 2), BASE_WEIGHT + (PILL_WEIGHT * 26))


    # Return all artists that were modified
    return line_raw, line_smoothed, line_notched, line_kalman, status_text, \
           line_comp_empty_threshold, line_comp_low_threshold, line_comp_full_threshold # Include lines to be redrawn


# --- Create and Run the Animation ---
# `frames` specifies the number of frames (iterations).
# `interval` is the delay between frames in milliseconds.
num_frames = int(SIMULATION_DURATION_SECONDS * SAMPLING_RATE)
ani = FuncAnimation(fig, update, frames=num_frames, interval=SIM_INTERVAL_MS, blit=False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title/label overlap
plt.show()