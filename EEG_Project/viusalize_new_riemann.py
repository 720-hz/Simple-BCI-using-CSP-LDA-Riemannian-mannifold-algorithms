import mne
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import pyriemann  # Required to load the Riemannian model

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
model_filename = 'silent_command_model_sub1_riemann.pkl'
try:
    clf = joblib.load(model_filename)
    print(f"✅ Model loaded: {model_filename}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Ensure 'pyriemann' is installed and the .pkl file is in the folder.")
    exit()

filename = "A01T.gdf"
# Verbose=False to keep terminal clean
raw = mne.io.read_raw_gdf(filename, preload=True, verbose=False)

# Keep only EEG channels (First 22)
raw.pick(raw.ch_names[:22])

# ==========================================
# 2. PREDICTION PIPELINE (Riemannian)
# ==========================================
# Riemannian usually works best with covariance matrices.
# We filter 7-35 Hz as per your original script.
print("📉 Filtering for Prediction (7-35 Hz)...")
raw.filter(7., 35., fir_design='firwin', skip_by_annotation='edge', verbose=False)

events, event_id = mne.events_from_annotations(raw)
target_event_ids = {'769': event_id['769'], '770': event_id['770']}

# Epoching (0.5 to 2.5s)
epochs = mne.Epochs(
    raw, 
    events, 
    event_id=target_event_ids, 
    tmin=0.5, 
    tmax=2.5, 
    baseline=None, 
    preload=True, 
    verbose=False
)

data = epochs.get_data()

# Scale data (Volts -> Microvolts)
# Riemannian geometry handles scaling well, but consistency is key.
data = data * 1e6 

# Predict
print(f"🔮 Predicting on {len(data)} trials using Riemannian Geometry...")
y_pred = clf.predict(data)

# Get True Labels (0 = Left, 1 = Right)
true_labels = epochs.events[:, -1]
left_id = event_id['769']
y_true = np.array([0 if pid == left_id else 1 for pid in true_labels])

# ==========================================
# 3. METRICS
# ==========================================
cm = confusion_matrix(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = np.mean(y_true == y_pred)

print("\n" + "="*40)
print(f"🏆 RIEMANNIAN CLASSIFICATION REPORT")
print("="*40)
print(f"Accuracy: {accuracy:.2%}")
print(f"F1 Score: {f1:.4f}")
print("-" * 40)

# ==========================================
# 4. VISUALIZATION DASHBOARD (THESIS STYLE)
# ==========================================
print("📊 Generating Comparative Graphs...")
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2)

# --- PLOT A: CONFUSION MATRIX ---
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, 
            xticklabels=['Left', 'Right'], yticklabels=['Left', 'Right'], 
            ax=ax1, annot_kws={"size": 16})
ax1.set_xlabel('Predicted Intention', fontsize=12)
ax1.set_ylabel('True Intention', fontsize=12)
ax1.set_title(f'Riemannian Confusion Matrix\nAccuracy: {accuracy:.1%}', fontsize=14, fontweight='bold')

# --- PLOT B: AVERAGE BRAINWAVES (EVOKED POTENTIAL) ---
# Compares the average signal of "Thinking Left" vs "Thinking Right"
ax2 = fig.add_subplot(gs[0, 1])

# Calculate averages
evoked_left = epochs['769'].average()
evoked_right = epochs['770'].average()

# Select Channel C3 (Index 7 usually) - Motor Cortex Left Hemisphere
# This channel usually reacts strongly to Right Hand movement
ch_index = 7 
ch_name = raw.ch_names[ch_index]

# Plot
ax2.plot(evoked_left.times, evoked_left.data[ch_index] * 1e6, label='Left Intention (Avg)', color='#e74c3c', linewidth=2)
ax2.plot(evoked_right.times, evoked_right.data[ch_index] * 1e6, label='Right Intention (Avg)', color='#2ecc71', linewidth=2, linestyle='--')

ax2.set_title(f'Neural Response Comparison (Channel: {ch_name})', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Amplitude (uV)')
ax2.legend(loc='upper right')

# --- PLOT C: POWER SPECTRAL DENSITY (PSD) ---
# This proves the "Mu Rhythm" (8-13 Hz) and "Beta" (13-30 Hz) are driving the model
ax3 = fig.add_subplot(gs[1, :])

# Compute PSD
# Note: Fmax is 35 because your Riemannian filter goes up to 35
psd_left = epochs['769'].compute_psd(fmin=7, fmax=35).get_data(return_freqs=True)
psd_right = epochs['770'].compute_psd(fmin=7, fmax=35).get_data(return_freqs=True)

# Average across all trials and all channels to get the "Global Brain State"
mean_psd_left = np.mean(psd_left[0], axis=(0, 1))
mean_psd_right = np.mean(psd_right[0], axis=(0, 1))
freqs = psd_left[1]

# Plot
ax3.plot(freqs, 10 * np.log10(mean_psd_left), label='Left Intention Spectrum', color='#e74c3c', linewidth=2)
ax3.plot(freqs, 10 * np.log10(mean_psd_right), label='Right Intention Spectrum', color='#2ecc71', linewidth=2, linestyle='--')
ax3.fill_between(freqs, 10 * np.log10(mean_psd_left), 10 * np.log10(mean_psd_right), color='gray', alpha=0.1)

ax3.set_title('Frequency Power Spectrum (7-35 Hz)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Power Density (dB)')
ax3.legend()

plt.tight_layout()
print("✅ Dashboard Generated!")
plt.show()