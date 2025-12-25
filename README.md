# Network_Security_Project

# ML-Based Firewall for Network Intrusion Detection

## Overview

This project implements a **machine learning–based firewall** for network intrusion detection, developed as part of a university **Network Security** course.

The system is designed to:

* Train an ML model on real network traffic data (TII-SSRC-23 dataset)
* Deploy the trained model inside a Python server acting as a **software firewall**
* Receive traffic representations from a client via HTTP/REST
* Decide in real time whether to **ALLOW** or **BLOCK** traffic based on learned network patterns

The focus of the project is **network-level feature analysis and firewall behavior**, not the sophistication of the ML algorithm itself.

---

## Architecture

The system follows a **Client–Server architecture**:

```
+------------------+        HTTP / REST        +----------------------+
|  Client (CSV)    |  --------------------->  |  ML Firewall Server  |
|                  |   feature vectors (POST) |  (Flask + ML Model)  |
+------------------+                           +----------------------+
```

### Client

* Reads rows from a CSV dataset (`client_data.csv`)
* Each row represents a **network flow** already processed into numerical features
* Sends feature vectors to the server via HTTP POST requests
* Used only to **simulate traffic** (not a real packet sniffer)

### Server (ML Firewall)

* Loads a trained ML model and scaler
* Receives feature vectors from the client
* Computes the probability that traffic is malicious
* Applies a **firewall policy** based on a probability threshold
* Maintains a **dynamic blacklist** of blocked IPs
* Logs all decisions for inspection

---

## Dataset

### TII-SSRC-23

The project uses the **TII-SSRC-23 dataset**, which contains:

* Realistic network traffic
* Benign and malicious flows
* Multiple attack categories (DoS, brute force, botnet, reconnaissance, etc.)

Each flow is represented by **86 features**, including:

* Packet statistics
* Timing and inter-arrival metrics
* TCP flag counters
* Flow activity and idle behavior

Only **10% of the dataset** is used during training to reduce memory and computation requirements.

---

## Feature Engineering

The project works entirely on **flow-level features**, not raw packets.

Examples of feature groups:

* **Volume-based**: total packets, total bytes, packet sizes
* **Time-based**: inter-arrival times, flow duration, activity/idle times
* **Protocol behavior**: TCP flag counts (SYN, ACK, RST, etc.)
* **Directional metrics**: forward vs backward traffic asymmetry

Some features (e.g. IP addresses, ports, timestamps) are intentionally **removed** to:

* Avoid information leakage
* Ensure generalization
* Keep the model independent from specific hosts

---

## Machine Learning Model

* Algorithm: **Random Forest Classifier**
* Reason:

  * Robust to noisy features
  * Interpretable
  * Widely used in intrusion detection literature

### Training pipeline

1. Feature scaling (StandardScaler)
2. Class balancing (SMOTE)
3. Train/Test split with stratification
4. Model evaluation using:

   * Precision / Recall / F1-score
   * ROC curve and AUC

Cross-validation (Stratified K-Fold) is used to assess model stability.

---

## Firewall Logic

The server behaves like a **stateful intrusion prevention system (IPS)**:

1. Receive a feature vector
2. Normalize features using the training scaler
3. Compute:

   ```
   P(malicious | features)
   ```
4. Apply decision rule:

   * If `P >= threshold` → BLOCK
   * Else → ALLOW
5. If blocked:

   * Add source IP to dynamic blacklist
   * Persist blacklist to disk

This mimics the behavior of a real firewall enforcing security policies.

---

## REST API

### POST /predict

Receives a feature vector and returns a firewall decision.

**Request (JSON):**

```json
{
  "features": [f1, f2, ..., fn],
  "ip": "192.168.1.10"
}
```

**Response (JSON):**

```json
{
  "prediction": "MALICIOUS",
  "probability": 0.87,
  "action": "BLOCK"
}
```

### GET /stats

Returns runtime firewall statistics:

* Total requests
* Benign vs malicious counts
* Blocked IPs
* Recent decisions

---

## PCAP Test Set

A small **PCAP test set** is generated automatically:

* Randomly selected from the original dataset
* Includes both benign and malicious traffic
* Used only for demonstration and analysis

The PCAPs are **not parsed live** during inference, but help demonstrate:

* Attack categories
* Relation between raw traffic and extracted features

---

## Limitations

* No live packet capture
* Feature extraction is offline (CSV-based)
* Client simulates traffic, not real network packets

These choices are intentional to keep the focus on **network feature analysis and firewall behavior**.

---

## How to Run

### Training (Colab)

1. Open the training notebook in Google Colab
2. Upload `kaggle.json`
3. Run all cells
4. Download:

   * `random_forest_model.pkl`
   * `scaler.pkl`
   * `client_data.csv`

### Server

```bash
python server.py
```

### Client

```bash
python client.py
```

---

## Educational Goals

This project demonstrates:

* How network traffic can be modeled as numerical features
* How ML decisions translate into firewall actions
* The difference between **classification** and **policy enforcement**
* The role of ML inside modern software-based security systems

---

## Author

Student project for Network Security course.
