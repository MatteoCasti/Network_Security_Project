# Network_Security_Project  
## Machine Learning–Based Firewall for Network Intrusion Detection

---

## 1. Introduction

This project presents the design and implementation of a **machine learning–based firewall** for **network intrusion detection**, developed within the context of a university **Network Security** course.

The system demonstrates how **flow-level network features**, extracted offline from traffic traces, can be used by a machine learning model to support **security decisions** typically performed by firewalls and intrusion prevention systems.

Rather than focusing on advanced model complexity, the project emphasizes:

- Network traffic representation
- Feature consistency between training and deployment
- Decision-making logic at the firewall level
- Integration of machine learning into a security-oriented system

---

## 2. System Architecture

The system is structured according to a **Client–Server architecture**, where machine learning inference is embedded within a software firewall.

```
+------------------+        HTTP / REST        +----------------------+
|  Client (CSV)    |  --------------------->  |  ML Firewall Server  |
|                  |   feature vectors (POST) |  (Flask + ML Model)  |
+------------------+                           +----------------------+
```

### 2.1 Client Component

The client component:

- Reads traffic data from a CSV file (`client_data.csv`)
- Treats each row as a **single network flow**
- Extracts the numerical feature vector
- Sends the features to the server via HTTP POST requests

The client **does not capture live traffic** and is used exclusively to simulate network activity during testing.

---

### 2.2 Firewall Server

The server acts as a **machine learning–enhanced firewall** and performs the following tasks:

- Loads a trained classification model and a feature scaler
- Receives feature vectors from the client
- Estimates the probability that a flow is malicious
- Applies a security policy (allow or block)
- Maintains a dynamic blacklist of blocked IP addresses
- Logs all security decisions

---

## 3. Dataset Description

### 3.1 TII-SSRC-23 Dataset

The project uses the **TII-SSRC-23 dataset**, which contains labeled network flows representing both benign and malicious traffic.

The dataset includes multiple attack categories, such as:

- Denial of Service (DoS)
- Brute force attacks
- Botnet activity
- Reconnaissance

Each network flow is originally described by **86 statistical features**, capturing:

- Packet-level statistics
- Inter-arrival time distributions
- TCP flag behavior
- Flow activity and idle periods

To reduce computational overhead, only **10% of the dataset** is randomly sampled for training.

---

## 4. Feature Engineering and Preprocessing

The system operates exclusively on **flow-level features**, without direct packet inspection.

### 4.1 Feature Removal

Several attributes are deliberately removed prior to training:

- Flow identifiers and IP addresses
- Source and destination ports
- Protocol identifiers
- Timestamps
- Traffic type and traffic subtype labels

This design choice:

- Prevents information leakage
- Improves generalization
- Avoids learning host-specific or dataset-specific artifacts

---

### 4.2 Data Cleaning

The following preprocessing steps are applied:

- Removal of duplicate flows
- Replacement of infinite values
- Elimination of missing values
- Encoding of remaining categorical attributes
- Conversion of labels to binary format (0 = benign, 1 = malicious)

---

### 4.3 Feature Consistency

To ensure correct deployment:

- The ordered list of feature names used during training is stored
- The server enforces the same feature order at inference time

This prevents feature misalignment between training and prediction.

---

## 5. Machine Learning Model

### 5.1 Model Choice

The classification task is performed using a **Random Forest Classifier**.

The algorithm is chosen because it:

- Handles high-dimensional feature spaces effectively
- Is robust to noise and correlated features
- Is widely adopted in network intrusion detection research

---

### 5.2 Training Pipeline

The training process follows these steps:

1. Stratified train–test split
2. Feature normalization using `StandardScaler`
3. Class imbalance mitigation using **SMOTE**
4. Model training on the balanced dataset
5. Evaluation on an unseen test set

---

### 5.3 Evaluation

Model performance is assessed using:

- Precision, recall, and F1-score
- Receiver Operating Characteristic (ROC) curve
- Area Under the Curve (AUC)

These metrics provide insight into both detection capability and false-positive behavior.

---

## 6. Firewall Decision Logic

The server implements **stateful firewall behavior**.

For each incoming request:

1. The feature vector is normalized
2. The model computes the probability of malicious traffic:
P(malicious | flow features)

3. A threshold-based decision rule is applied:
- Probability ≥ threshold → **BLOCK**
- Probability < threshold → **ALLOW**
4. Blocked source IP addresses are added to a persistent blacklist

This mechanism emulates how intrusion prevention systems enforce security policies based on risk estimation.

---

## 7. REST API

### 7.1 POST `/predict`

Accepts a feature vector and returns a firewall decision.

**Request format:**
```json
{
"features": [f1, f2, ..., fn],
"ip": "192.168.1.10"
}
```
The feature vector must follow the exact order used during training.

Response format:
```json
{
  "prediction": "MALICIOUS",
  "probability": 0.91,
  "action": "BLOCK"
}
```
### 7.2 GET /stats
- Returns runtime information about the firewall:

- Total processed flows

- Number of benign and malicious flows

- List of blocked IP addresses

- Recent firewall events

## 8. PCAP Demonstration Set
- A small PCAP dataset is generated for demonstration purposes:

- Randomly selected from the original dataset

- Contains both benign and malicious traffic

- Used only for visualization and analysis

- Live packet inspection is not performed during inference.

## 9. Limitations
- No real-time packet capture

- Feature extraction is offline

- The client simulates traffic flows

- These limitations are intentional and allow the project to focus on security logic and ML integration rather than packet processing.

## 10. Execution Instructions
### 10.1 Model Training (Google Colab)
- Open the training notebook in Google Colab

- Upload kaggle.json

- Run all cells

- Download the generated files:

- random_forest_model.pkl

- scaler.pkl

- feature_names.pkl

- client_data.csv

### 10.2 Firewall Server

```bash
    python server.py
```

### 10.3 Client
```bash
    python client.py
```

## 11. Educational Objectives
This project illustrates:

- How network traffic can be abstracted into numerical features

 - How machine learning supports intrusion detection

- How ML predictions translate into firewall actions

- The distinction between detection and enforcement