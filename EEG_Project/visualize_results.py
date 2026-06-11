import mne
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import pyriemann

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
model_filename = 'silent_command_model_sub1_riemann.pkl'
clf = joblib.load(model_filename)
print(f"✅ Model loaded: {model_filename}")

filename = "A01T.gdf"
raw = mne.io.read_raw_gdf(filename, preload=True, verbose=False)

# Keep only EEG channels
raw.pick(raw.ch_names[:22])

# ⚠️ Create a copy for visualization BEFORE we apply the prediction filter
# We need the raw full-spectrum data to see Delta (0.5-4Hz) and Theta (4-8Hz)
raw_for_viz = raw.copy()

# ==========================================
# 2. PREDICTION PIPELINE (Standard)
# ==========================================
print("📉 Filtering for Prediction (7-35 Hz)...")
raw.filter(7., 35., fir_design='firwin', skip_by_annotation='edge', verbose=False)

events, event_id = mne.events_from_annotations(raw)
target_event_ids = {'769': event_id['769'], '770': event_id['770']}

epochs = mne.Epochs(raw, events, event_id=target_event_ids, tmin=0.5, tmax=2.5, baseline=None, preload=True, verbose=False)
data = epochs.get_data()

# Scale data (Volts -> Microvolts) for Riemannian validity
data = data * 1e6 

# Predict
y_pred = clf.predict(data)

# Get True Labels (0 = Left, 1 = Right)
true_labels = epochs.events[:, -1]
left_id = event_id['769']
y_true = np.array([0 if pid == left_id else 1 for pid in true_labels])

# ==========================================
# 3. METRICS (F1 & Confusion Matrix)
# ==========================================
cm = confusion_matrix(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = np.mean(y_true == y_pred)

print("\n" + "="*40)
print(f"🏆 CLASSIFICATION REPORT")
print("="*40)
print(f"Accuracy: {accuracy:.2%}")
print(f"F1 Score: {f1:.4f}")
print("-" * 40)

# ==========================================
# 4. VISUALIZATION DASHBOARD
# ==========================================
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 3) # 2 Rows, 3 Columns

# --- PLOT A: CONFUSION MATRIX ---
ax_cm = fig.add_subplot(gs[0, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Left', 'Right'], yticklabels=['Left', 'Right'], ax=ax_cm, annot_kws={"size": 16})
ax_cm.set_xlabel('Predicted Command', fontsize=12)
ax_cm.set_ylabel('True Command', fontsize=12)
ax_cm.set_title(f'Confusion Matrix\nF1 Score: {f1:.3f}', fontsize=14, fontweight='bold')

# --- PLOT B: EEG FREQUENCY BANDS (Visualization) ---
# We take ONE trial (e.g., the first "Right" command) and decompose it
trial_idx = np.where(y_true == 1)[0][0] # Get index of first Right turn
# Get data for channel C3 (Motor Cortex - Left Hemisphere)
c3_idx = raw_for_viz.ch_names.index('EEG-C3') 
# Extract 2 seconds of raw data from that trial
start_sample = epochs.events[trial_idx, 0]
end_sample = start_sample + int(2.0 * raw_for_viz.info['sfreq'])
segment = raw_for_viz.get_data(start=start_sample, stop=end_sample)[c3_idx] * 1e6 # Convert to uV
time_axis = np.linspace(0, 2, len(segment))

# Define Bands
bands = {
    'Delta (0.5-4 Hz)': (0.5, 4),
    'Theta (4-8 Hz)': (4, 8),
    'Alpha (8-13 Hz)': (8, 13),
    'Beta (13-30 Hz)': (13, 30),
    'Gamma (30-50 Hz)': (30, 50)
}

# Plot raw signal
ax_raw = fig.add_subplot(gs[0, 1:])
ax_raw.plot(time_axis, segment, color='black', alpha=0.7)
ax_raw.set_title(f'Raw EEG Signal (Channel C3 - Trial {trial_idx})', fontweight='bold')
ax_raw.set_ylabel('Amplitude (uV)')

# Plot filtered bands
positions = [gs[1, 0], gs[1, 1], gs[1, 2]]
colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']

# We will plot the bands on the remaining grid spots. 
# Since we have 5 bands and 3 spots, we'll combine them into a multi-line plot or use subplots.
# Let's use subplots for clarity. We create a nested grid for the bottom row.

ax_bands = fig.add_subplot(gs[1, :])
ax_bands.axis('off') # Hide main axis, use it as container
inner_grid = ax_bands.inset_axes([0, 0, 1, 1])
inner_grid.axis('off')

# Manually create axes for bands
axes_bands = []
for i in range(5):
    # Create 5 small axes across the bottom
    ax = fig.add_subplot(2, 5, i + 6) # Row 2, columns 1-5
    axes_bands.append(ax)

for i, (band_name, (low, high)) in enumerate(bands.items()):
    # Apply bandpass filter to the segment
    # Note: Using a simple filter on short data (2s) requires care, we use simple FFT filter here for viz
    freqs = np.fft.rfftfreq(len(segment), 1/raw_for_viz.info['sfreq'])
    fft_vals = np.fft.rfft(segment)
    
    # Zero out frequencies outside the band
    mask = (freqs >= low) & (freqs <= high)
    fft_filtered = fft_vals * mask
    filtered_signal = np.fft.irfft(fft_filtered)
    
    axes_bands[i].plot(time_axis, filtered_signal, color=colors[i])
    axes_bands[i].set_title(band_name, fontsize=10)
    axes_bands[i].set_ylim(-15, 15) # Fixed scale for comparison
    if i == 0:
        axes_bands[i].set_ylabel('uV')
    axes_bands[i].tick_params(labelbottom=False)

plt.tight_layout()
plt.show()