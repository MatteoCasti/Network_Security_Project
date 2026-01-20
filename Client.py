# =========================================================
#  IMPORTS
# =========================================================
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 20/01/2026

SERVER_URL = "http://127.0.0.1:8080/predict"

# =========================================================
# LOAD DATASET
# =========================================================
df = pd.read_csv("client_data.csv")
print(f" Inviando {len(df)} richieste al server...")

y_true = []
y_scores = []

# =========================================================
# SESSIONE HTTP (RIUSA SOCKET)
# =========================================================
session = requests.Session()

# =========================================================
# ITERAZIONE
# =========================================================
for i, row in df.iterrows():

    y_true.append(int(row["Label"]))
    features = row.drop("Label").tolist()

    response = session.post(
        SERVER_URL,
        json={"features": features},
        timeout=5
    )

    result = response.json()

    if "error" in result:
        print(f"[{i}]  SERVER ERROR: {result['error']}")
        continue

    y_scores.append(float(result["probability"]))

    # Log solo ogni 1000 richieste
    if i % 1000 == 0:
        print(f"[{i}] OK")

session.close()

# =========================================================
# ROC CURVE
# =========================================================
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

print(f"\n AUC = {roc_auc:.4f}")

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – ML Firewall")
plt.legend()
plt.grid(True)
plt.show()
