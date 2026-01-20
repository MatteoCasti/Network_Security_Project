from flask import Flask, request, jsonify          # Import Flask web framework, HTTP request handler, and JSON response utility
import numpy as np                                 # Import NumPy for numerical array handling
import joblib                                     # Import joblib to load the trained ML model and scaler
import datetime                                   # Import datetime to timestamp firewall events
import json
import pandas as pd
# Import json to serialize events into log format
import os                                         # Import os for filesystem checks and operations
# 19/01/2026
# ================================================
# FIREWALL CONFIGURATION PARAMETERS
# ================================================
THRESHOLD = 0.50                                  # Probability threshold above which traffic is classified as malicious
BLOCK_IPS_FILE = "blocked_ips.txt"                # File used to persistently store blocked IP addresses
LOG_FILE = "firewall_logs.jsonl"                  # Log file storing one JSON event per line


# ================================================
# RUNTIME FIREWALL STATISTICS
# ================================================
stats = {
    "total_requests": 0,                          # Total number of requests processed by the firewall
    "benign": 0,                                  # Counter for benign traffic
    "malicious": 0,                               # Counter for malicious traffic
    "blocked_ips": set(),                         # Set of currently blocked IP addresses
    "recent_events": []                           # Circular buffer of the most recent firewall events
}

# ================================================
# LOAD TRAINED MODEL AND FEATURE SCALER
# ================================================
print("Loading model & scaler…")                  # Log message indicating startup phase
model = joblib.load("random_forest_model.pkl")    # Load the trained Random Forest classifier from disk
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
# Load the StandardScaler used during training
print("Model loaded.")                            # Confirm successful loading

app = Flask(__name__)                             # Instantiate the Flask web application

# ================================================
# UTILITY FUNCTION: LOG FIREWALL EVENTS
# ================================================
def log_event(event):                             # Define a function to log firewall decisions
    event_json = json.dumps(event)                # Serialize the event dictionary to JSON format
    with open(LOG_FILE, "a") as f:                 # Open the log file in append mode
        f.write(event_json + "\n")                 # Write the event as a single JSON line

    stats["recent_events"].append(event)           # Store the event in memory for dashboard usage
    if len(stats["recent_events"]) > 10:           # Check if more than 10 events are stored
        stats["recent_events"].pop(0)              # Remove the oldest event to keep buffer size constant

# ================================================
# LOAD PERSISTENT BLOCKED IP LIST (STATEFUL BEHAVIOR)
# ================================================
if os.path.exists(BLOCK_IPS_FILE):                 # Check if a previous blocked IP list exists
    with open(BLOCK_IPS_FILE, "r") as f:            # Open the file in read mode
        for line in f:                              # Iterate over each line in the file
            ip = line.strip()                       # Remove newline and whitespace characters
            if ip:                                  # Ensure the line is not empty
                stats["blocked_ips"].add(ip)       # Add the IP to the in-memory blocked set

# ================================================
# API ENDPOINT: /predict
# ================================================
@app.route("/predict", methods=["POST"])           # Define REST endpoint for ML-based traffic classification
def predict():                                     # Function handling prediction requests
    try:
        data = request.get_json()                  # Parse JSON payload from incoming HTTP request


        features_df = pd.DataFrame(
            [data["features"]],
            columns=feature_names
        )

        features_scaled = scaler.transform(features_df)

        source_ip = data.get("ip", "unknown")
        # Extract source IP if provided, otherwise default to "unknown"

        # ========================================
        # STATEFUL FIREWALL CHECK: BLOCKED IP
        # ========================================
        if source_ip in stats["blocked_ips"]:       # Check if source IP is already blocked
            return jsonify({
                "prediction": "MALICIOUS",         # Force malicious classification
                "probability": 1.0,                 # Assign maximum malicious probability
                "action": "BLOCK",                  # Firewall action: block traffic
                "reason": "IP already blocked"      # Explain reason for blocking
            })

        # ========================================
        # FEATURE NORMALIZATION
        # ========================================
        features_scaled = scaler.transform(features_df)
        # Apply the same scaling used during model training

        # ========================================
        # MACHINE LEARNING INFERENCE
        # ========================================
        prob = model.predict_proba(features_scaled)[0, 1]
        # Extract probability of the malicious class

        is_malicious = prob >= THRESHOLD
        # Compare probability against decision threshold

        # ========================================
        # FIREWALL POLICY DECISION
        # ========================================
        action = "BLOCK" if is_malicious else "ALLOW"
        # Decide whether to block or allow traffic

        # ========================================
        # UPDATE FIREWALL STATISTICS
        # ========================================
        stats["total_requests"] += 1                # Increment total processed requests
        if is_malicious:
            stats["malicious"] += 1                 # Increment malicious counter
        else:
            stats["benign"] += 1                    # Increment benign counter

        # ========================================
        # DYNAMIC BLACKLIST UPDATE
        # ========================================
        if is_malicious and source_ip != "unknown":
            stats["blocked_ips"].add(source_ip)     # Add IP to blocked list
            with open(BLOCK_IPS_FILE, "a") as f:    # Persist blocked IP to disk
                f.write(source_ip + "\n")

        # ========================================
        # EVENT CONSTRUCTION AND LOGGING
        # ========================================
        event = {
            "timestamp": datetime.datetime.now().isoformat(),  # Event timestamp
            "ip": source_ip,                                   # Source IP address
            "probability": float(prob),                        # Malicious probability
            "action": action,                                  # Firewall decision
            "prediction": "MALICIOUS" if is_malicious else "BENIGN",
            # Final classification label
            "features": data["features"]                       # Feature vector used
        }

        log_event(event)                      # Log the firewall decision

        return jsonify(event)                 # Return decision to the client

    except Exception as e:
        return jsonify({
            "prediction": "ERROR",
            "probability": 0.0,
            "action": "ERROR",
            "error": str(e)
        }), 400

        # Return error message if request processing fails

# ================================================
# API ENDPOINT: /stats
# ================================================
@app.route("/stats", methods=["GET"])          # Define endpoint to retrieve firewall statistics
def get_stats():                               # Function handling statistics requests
    return jsonify({
        "total_requests": stats["total_requests"],      # Total processed requests
        "benign": stats["benign"],                       # Number of benign flows
        "malicious": stats["malicious"],                 # Number of malicious flows
        "blocked_ip_count": len(stats["blocked_ips"]),  # Count of blocked IP addresses
        "blocked_ips": list(stats["blocked_ips"]),       # List of blocked IPs
        "recent_events": stats["recent_events"]          # Recent firewall decisions
    })

# ================================================
# SERVER STARTUP
# ================================================
if __name__ == "__main__":                      # Check if script is executed directly
    print("ML Firewall running at http://127.0.0.1:8080")
    # Inform the user of the server address
    app.run(host="127.0.0.1", port=8080)        # Start Flask server on localhost