import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os

print("Starting the final model training process...")

# --- Create a 'models' directory to store our brains ---
if not os.path.exists('models'):
    os.makedirs('models')
    print("Created 'models' directory.")

# --- Load and Prepare Data ---
df = pd.read_csv('Processed.csv')
print("Dataset loaded successfully.")

# --- Preprocessing: We need to convert text columns to numbers ---
encoders = {}
categorical_cols = ['Gender', 'University', 'Department', 'Academic Year', 'Current CGPA', 'waiver_or_scholarship']

print("Encoding all categorical features...")
for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].astype(str))
    encoders[col] = encoder

# Save all encoders in one file. This is very important!
joblib.dump(encoders, 'models/encoders.joblib')
print("All encoders are saved in 'models/encoders.joblib'")
print("-" * 30)


# --- Train and Save All Three Models ---

# 1. ANXIETY MODEL
print("Training ANXIETY model...")
anxiety_features = [
    'Age', 'Gender', 'University', 'Department', 'Academic Year', 'Current CGPA', 'waiver_or_scholarship',
    'GAD1', 'GAD2', 'GAD3', 'GAD4', 'GAD5', 'GAD6', 'GAD7'
]
anxiety_model = LogisticRegression(max_iter=1000)
anxiety_model.fit(df[anxiety_features], df['Anxiety Label'])
joblib.dump(anxiety_model, 'models/anxiety_model.joblib')
print("Anxiety model saved!")

# 2. DEPRESSION MODEL
print("Training DEPRESSION model...")
depression_features = [
    'Age', 'Gender', 'University', 'Department', 'Academic Year', 'Current CGPA', 'waiver_or_scholarship',
    'PHQ1', 'PHQ2', 'PHQ3', 'PHQ4', 'PHQ5', 'PHQ6', 'PHQ7', 'PHQ8', 'PHQ9'
]
depression_model = LogisticRegression(max_iter=1000)
depression_model.fit(df[depression_features], df['Depression Label'])
joblib.dump(depression_model, 'models/depression_model.joblib')
print("Depression model saved!")

# 3. STRESS MODEL
print("Training STRESS model...")
stress_features = [
    'Age', 'Gender', 'University', 'Department', 'Academic Year', 'Current CGPA', 'waiver_or_scholarship',
    'PSS1', 'PSS2', 'PSS3', 'PSS4', 'PSS5', 'PSS6', 'PSS7', 'PSS8', 'PSS9', 'PSS10'
]
stress_model = LogisticRegression(max_iter=2000)
stress_model.fit(df[stress_features], df['Stress Label'])
joblib.dump(stress_model, 'models/stress_model.joblib')
print("Stress model saved!")

print("\n✅ SUCCESS! All models and encoders are ready in the 'models' folder.")