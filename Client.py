import pandas as pd                  # Import pandas for CSV loading and data manipulation
import requests                      # Import requests to send HTTP requests to the server
import time                          # Import time to simulate network delays

SERVER_URL = "http://127.0.0.1:8080/predict"
# URL of the ML-based firewall server REST endpoint (POST /predict)

# =========================================================
# LOAD CLIENT DATASET
# =========================================================
df = pd.read_csv("client_data.csv")
# Load the client-side dataset containing feature vectors and ground-truth labels

print(f"Inviando {len(df)} richieste al server...")
# Print how many requests will be sent to the server

# =========================================================
# ITERATE OVER CLIENT FLOWS
# =========================================================
for i, row in df.iterrows():
    # Iterate row-by-row over the client dataset
    # Each row represents a single network flow (already feature-extracted)

    true_label = row["Label"]
    # Extract the original ground-truth label from the dataset

    # =====================================================
    # MAP NUMERICAL LABEL TO HUMAN-READABLE STRING
    # =====================================================
    if true_label == 0 or true_label == "0":
        true_label_str = "BENIGN"
        # Label 0 corresponds to benign traffic
    elif true_label == 1 or true_label == "1":
        true_label_str = "MALICIOUS"
        # Label 1 corresponds to malicious traffic
    else:
        true_label_str = str(true_label)
        # Fallback in case the label is already a string or has a different format

    # =====================================================
    # FEATURE VECTOR EXTRACTION
    # =====================================================
    features = row.drop("Label").tolist()
    # Remove the Label column and convert remaining feature values into a Python list
    # This list represents the numerical feature vector sent to the firewall

    # =====================================================
    # SEND REQUEST TO ML FIREWALL SERVER
    # =====================================================
    response = requests.post(
        SERVER_URL,
        json={"features": features}
    )
    # Send an HTTP POST request containing the feature vector as JSON payload

    result = response.json()
    # Parse the JSON response returned by the server

    prediction = result["prediction"]
    # Extract the predicted class ("BENIGN" or "MALICIOUS") from the response

    prob = result["probability"]
    # Extract the predicted probability of malicious traffic

    # =====================================================
    # OUTPUT RESULT (GROUND TRUTH vs PREDICTION)
    # =====================================================
    print(
        f"[{i}] True={true_label_str} | Pred={prediction} "
        f"(p_malicious={prob:.3f})"
    )
    # Print:
    # - index of the request
    # - original dataset label (ground truth)
    # - server prediction
    # - probability of malicious class

    # =====================================================
    # SIMULATE REAL NETWORK TRAFFIC TIMING
    # =====================================================
    time.sleep(0.3)
    # Introduce a delay between requests to simulate non-burst traffic
