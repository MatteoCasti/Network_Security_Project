# Update 29 01 2026
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# =========================================================
# CONFIGURATION PARAMETERS
# =========================================================
# Endpoint for the deployed ML model (Inference Server)
SERVER_URL = "http://127.0.0.1:8080/predict"
# Seed for reproducibility across different demo runs
RANDOM_STATE = 42
# We constrain the demo size to ensure a quick execution while maintaining statistical relevance
# max dataset samples 83934
TOTAL_SAMPLES = 83934

# =========================================================
# DATASET LOADING
# =========================================================
# Load historical client data containing features and ground truth labels
df = pd.read_csv("client_data.csv")

# =========================================================
# STRATIFIED SAMPLING STRATEGY
# =========================================================
# Segregate the dataset based on the 'Label' (1: Malicious, 0: Benign)
df_pos = df[df["Label"] == 1]
df_neg = df[df["Label"] == 0]

# Strategy: Maintain all rare "Benign" instances to test the model's specificity
df_benign_sampled = df_neg

# Calculate the remaining quota to reach TOTAL_SAMPLES using Malicious instances
# This ensures the demo dataset is balanced or representative of the specific test case
needed_malicious = TOTAL_SAMPLES - len(df_benign_sampled)
df_malicious_sampled = df_pos.sample(n=needed_malicious, random_state=RANDOM_STATE)

# Combine, shuffle (fraction=1), and reset index to simulate a real-time stream of requests
df_demo = (
    pd.concat([df_benign_sampled, df_malicious_sampled])
    .sample(frac=1, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

print(f"--- Dataset Info ---")
print(f"Malicious in CSV: {len(df_pos)} | Benign in CSV: {len(df_neg)}")
print(f"Sending {len(df_demo)} requests ({len(df_benign_sampled)} benign + {len(df_malicious_sampled)} malicious)")
print("-" * 30)

# =========================================================
# INFERENCE EXECUTION ENGINE
# =========================================================
# Use requests.Session() to persist the TCP connection (HTTP Keep-Alive)
# This significantly improves performance for bulk requests during a demo
session = requests.Session()
y_true = []   # Ground truth
y_scores = [] # Raw model probabilities
y_pred = []   # Discrete classifications

for i, row in df_demo.iterrows():
    true_label = int(row["Label"])
    # Extract feature vector by dropping the target column
    features = row.drop("Label").tolist()

    try:
        # Send POST request with JSON payload to the REST API
        response = session.post(SERVER_URL, json={"features": features}, timeout=5)
        result = response.json()

        if "error" in result:
            print(f"[{i}] ERROR: {result['error']}")
            continue

        # Extract the probability and apply a standard 0.5 classification threshold
        prob = float(result["probability"])
        pred = 1 if prob >= 0.5 else 0

        y_true.append(true_label)
        y_scores.append(prob)
        y_pred.append(pred)

        # Real-time logging: Crucial for visual feedback during the demo execution
        print(f"[{i}] True={true_label} | Pred={pred} | P(malicious)={prob:.3f}")

    except Exception as e:
        print(f"[{i}] Connection Error: {e}")

session.close()

# =========================================================
# PERFORMANCE METRICS & VISUALIZATION
# =========================================================
# Compute Receiver Operating Characteristic (ROC) and Area Under the Curve (AUC)
# These metrics evaluate the model's discriminative power
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
# Compute Confusion Matrix to visualize True Positives, False Positives, etc.
cm = confusion_matrix(y_true, y_pred)

print(f"\n--- FINAL RESULTS ---")
print(f"AUC: {roc_auc:.4f}")
print("Confusion Matrix:")
print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")

# Setup visualization environment
plt.figure(figsize=(12, 5))

# Plot 1: ROC Curve (Shows the trade-off between Sensitivity and Specificity)
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--') # Baseline (random guess)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)

# Plot 2: Confusion Matrix Heatmap (Visual representation of classification accuracy)
plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Benign (0)', 'Malicious (1)'],
            yticklabels=['Benign (0)', 'Malicious (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

plt.tight_layout()
plt.show()