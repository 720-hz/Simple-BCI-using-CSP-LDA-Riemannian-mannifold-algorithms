import mne
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. LOAD MODEL
# ==========================================
model_filename = 'silent_command_model_sub1.pkl'
clf = joblib.load(model_filename)
print(f"✅ Model loaded: {model_filename}")

# ==========================================
# 2. LOAD RAW DATA
# ==========================================
filename = "A01T.gdf"
# Preload is required for filtering
raw = mne.io.read_raw_gdf(filename, preload=True)

# ==========================================
# 3. CHANNEL SELECTION
# ==========================================
print(f"⚠️ Original channel count: {len(raw.ch_names)}")
print("✂️  Keeping only the first 22 EEG channels...")
raw.pick(raw.ch_names[:22])

# ==========================================
# 4. PREPROCESSING
# ==========================================
# A. Filter 8-30Hz (Mu/Beta Band)
print("📉 Applying Bandpass Filter (8-30 Hz)...")
raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge')

# B. Epoching
events, event_id = mne.events_from_annotations(raw)
target_event_ids = {'769': event_id['769'], '770': event_id['770']}

# We need a longer baseline for visualization purposes
epochs = mne.Epochs(
    raw, 
    events, 
    event_id=target_event_ids, 
    tmin=0.5, 
    tmax=3.5, 
    baseline=None, 
    preload=True
)

data = epochs.get_data() 

# ==========================================
# 5. UNIT SCALING
# ==========================================
print(f"📏 Raw Data Mean (Volts): {np.mean(np.abs(data)):.2e}")
data = data * 1e6  # Convert to Microvolts
print(f"📏 Scaled Data Mean (uV):  {np.mean(np.abs(data)):.2f}")

# ==========================================
# 6. PREDICTION
# ==========================================
print(f"🔮 Predicting on {len(data)} trials...")
predictions = clf.predict(data)

# Calculate Accuracy
true_labels = epochs.events[:, -1]
left_id = event_id['769']
# Convert event IDs to 0 and 1
y_true = np.array([0 if pid == left_id else 1 for pid in true_labels])

correct = np.sum(predictions == y_true)
accuracy = correct / len(predictions)

print("\n" + "="*40)
print(f"🏆 ADJUSTED ACCURACY: {accuracy:.2%}")
print("="*40)

# ==========================================
# 8. VISUALIZATION DASHBOARD (NEW ADDITION)
# ==========================================
print("📊 Generating Graphs...")
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2)

# --- GRAPH 1: CONFUSION MATRIX ---
# This proves your 86% accuracy visually
ax1 = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_true, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Left Intention', 'Right Intention'],
            yticklabels=['Left Intention', 'Right Intention'], ax=ax1, annot_kws={"size": 14})
ax1.set_title(f'Confusion Matrix\nAccuracy: {accuracy:.1%}', fontsize=14, fontweight='bold')
ax1.set_ylabel('True Intention')
ax1.set_xlabel('Predicted Intention')

# --- GRAPH 2: AVERAGE BRAINWAVES (EVOKED) ---
# This compares the average "Left" signal vs "Right" signal
# We take channel C3 (Left Hemisphere) or C4 (Right Hemisphere)
ax2 = fig.add_subplot(gs[0, 1])
# Average all trials for Left (769) and Right (770)
evoked_left = epochs['769'].average()
evoked_right = epochs['770'].average()
# Pick a specific motor cortex channel to show (e.g., C3 or Cz)
# Note: We must check if channel names match. Usually 'EEG-C3' or similar.
channel_to_plot = raw.ch_names[7] # Index 7 is usually C3 in this dataset
ax2.plot(evoked_left.times, evoked_left.data[7] * 1e6, label='Left Intention (Avg)', color='#e74c3c')
ax2.plot(evoked_right.times, evoked_right.data[7] * 1e6, label='Right Intention (Avg)', color='#3498db')
ax2.set_title(f'Average Brain Signal Comparison (Channel: {channel_to_plot})', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Amplitude (uV)')
ax2.legend()

# --- GRAPH 3: POWER SPECTRAL DENSITY (PSD) ---
# This proves the "Alpha Rhythm" (8-12Hz) exists in your data
ax3 = fig.add_subplot(gs[1, :])
# Compute PSD for the first few trials
psd_left = epochs['769'].compute_psd(fmin=8, fmax=30).get_data(return_freqs=True)
psd_right = epochs['770'].compute_psd(fmin=8, fmax=30).get_data(return_freqs=True)

# Mean across all trials and channels for a smooth line
mean_psd_left = np.mean(psd_left[0], axis=(0, 1))
mean_psd_right = np.mean(psd_right[0], axis=(0, 1))
freqs = psd_left[1]

ax3.plot(freqs, 10 * np.log10(mean_psd_left), label='Left Intention Spectrum', color='#e74c3c', linewidth=2)
ax3.plot(freqs, 10 * np.log10(mean_psd_right), label='Right Intention Spectrum', color='#3498db', linewidth=2, linestyle='--')
ax3.fill_between(freqs, 10 * np.log10(mean_psd_left), 10 * np.log10(mean_psd_right), color='gray', alpha=0.1)

ax3.set_title('Frequency Power Spectrum (8-30 Hz)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Power Spectral Density (dB)')
ax3.legend()

plt.tight_layout()
print("✅ Graphs Generated! Check the popup window.")
plt.show()