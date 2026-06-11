import mne
import numpy as np
import joblib
import pyriemann

# ==========================================
# 1. LOAD RIEMANNIAN MODEL
# ==========================================
model_filename = 'silent_command_model_sub1_riemann.pkl'
clf = joblib.load(model_filename)
print(f"✅ Model loaded: {model_filename}")

# ==========================================
# 2. LOAD RAW DATA
# ==========================================
filename = "A01T.gdf"
# Preload is required for filtering
raw = mne.io.read_raw_gdf(filename, preload=True)

# ==========================================
# 3. PREPROCESSING
# ==========================================
# A. Channel Selection
# We must remove the EOG channels (last 3) to match the 22-channel model
print("✂️  Keeping only the first 22 EEG channels...")
raw.pick(raw.ch_names[:22])

# B. Filtering (7-35 Hz)
# Matches the "Ultra-Pro" training config
print("📉 Applying Bandpass Filter (7-35 Hz)...")
raw.filter(7., 35., fir_design='firwin', skip_by_annotation='edge')

# C. Epoching (0.5 - 2.5s)
# Riemannian is sensitive to timing; this matches training exactly.
events, event_id = mne.events_from_annotations(raw)
target_event_ids = {'769': event_id['769'], '770': event_id['770']}

epochs = mne.Epochs(
    raw,
    events,
    event_id=target_event_ids,
    tmin=0.5,
    tmax=2.5,
    baseline=None,
    preload=True
)

data = epochs.get_data()

# ==========================================
# 4. SCALING (THE FIX)
# ==========================================
# The Tangent Space projection relies on a Reference Matrix from training.
# If training was uV and prediction is V, the projection fails.
# We convert Volts -> Microvolts to match the Training Energy.
print(f"📏 Raw Data Mean: {np.mean(np.abs(data)):.2e}")
data = data * 1e6
print(f"📏 Scaled Mean:  {np.mean(np.abs(data)):.2f} (Target: ~5-50)")

# ==========================================
# 5. PREDICTION
# ==========================================
print(f"🔮 Predicting on {len(data)} trials...")
predictions = clf.predict(data)

# Calculate Accuracy
true_labels = epochs.events[:, -1]
left_id = event_id['769']
y_true = np.array([0 if pid == left_id else 1 for pid in true_labels])

correct = np.sum(predictions == y_true)
accuracy = correct / len(predictions)

# ==========================================
# 6. RESULTS
# ==========================================
print("\n" + "="*40)
print(f"🏆 RIEMANNIAN ACCURACY: {accuracy:.2%}")
print("="*40)
print(f"Predicted Left (0):  {np.sum(predictions == 0)}")
print(f"Predicted Right (1): {np.sum(predictions == 1)}")

print("\nSample Comparisons:")
print("Trial | True   | Pred   | Result")
print("-" * 35)
for i in range(min(15, len(predictions))):
    res = "✅" if predictions[i] == y_true[i] else "❌"
    t_lbl = "Left" if y_true[i] == 0 else "Right"
    p_lbl = "Left" if predictions[i] == 0 else "Right"
    print(f"{i+1:5} | {t_lbl:6} | {p_lbl:6} | {res}")