import joblib
import numpy as np

# 1. Load the model brain
try:
    model = joblib.load('silent_command_model_sub1.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading file: {e}")

# 2. Create sample data
# Your model expects 8 channels of EEG data.
# Shape = (1 trial, 8 channels, 100 time points)
test_data = np.random.rand(1, 22, 100)

# 3. Predict the command
prediction = model.predict(test_data)
print(f"🤖 The model thinks this command is: {prediction}")
print(f"The classes this model can recognize are: {model.classes_}")