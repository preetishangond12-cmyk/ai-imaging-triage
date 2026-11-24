import pandas as pd
import os

csv_path = r"C:\Users\Preeti\OneDrive\Desktop\AI Imaging Triage\dataset\chexpert\train_cheXbert.csv"

output_path = r"C:\Users\Preeti\OneDrive\Desktop\AI Imaging Triage\dataset\chexpert\chexpert_subset.csv"

df = pd.read_csv(csv_path)

# Keep only frontal X-rays
frontal = df[df["Frontal/Lateral"] == "Frontal"].copy()

# Extract patient ID
frontal["patient_id"] = frontal["Path"].str.extract(r"(patient\d+)")

# Select 5,000 images
subset = frontal.sample(n=5000, random_state=42)

# Save subset
subset.to_csv(output_path, index=False)

print("Subset created successfully!")
print("Images:", len(subset))
print("Patients:", subset["patient_id"].nunique())
print("Saved to:")
print(output_path)