# -*- coding: utf-8 -*-                          # File encoding declaration
"""ML Firewall Training Notebook"""               # Notebook description

# =========================================================
# INSTALL DEPENDENCIES (COLAB-SPECIFIC)
# =========================================================
!pip install pandas scikit-learn joblib xgboost imbalanced-learn kaggle
# Installs all required Python packages in the Colab environment

# =========================================================
# CORE PYTHON & ML LIBRARIES
# =========================================================
import pandas as pd                               # DataFrame manipulation
import numpy as np                                # Numerical operations
import matplotlib.pyplot as plt                   # Plotting utilities
import joblib                                    # Model serialization
import random                                    # Random sampling
import os                                        # File-system operations
import shutil                                    # File copy/delete utilities
import zipfile                                   # ZIP archive handling

# =========================================================
# SCIKIT-LEARN COMPONENTS
# =========================================================
from sklearn.model_selection import train_test_split, StratifiedKFold
# Dataset splitting and cross-validation utilities

from sklearn.preprocessing import StandardScaler, LabelEncoder
# Feature normalization and categorical encoding

from sklearn.ensemble import RandomForestClassifier
# Random Forest classifier implementation

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
# Model evaluation metrics

from imblearn.over_sampling import SMOTE
# Synthetic Minority Over-sampling Technique

from google.colab import files
# Colab file upload/download utilities

# =========================================================
# KAGGLE AUTHENTICATION
# =========================================================
files.upload()                                   # Upload kaggle.json credentials

!mkdir -p ~/.kaggle                              # Create Kaggle config directory
!mv kaggle.json ~/.kaggle/                       # Move API token
!chmod 600 ~/.kaggle/kaggle.json                 # Secure file permissions

# =========================================================
# DATASET DOWNLOAD
# =========================================================
!kaggle datasets download -d daniaherzalla/tii-ssrc-23
# Download the TII-SSRC-23 dataset from Kaggle

!unzip tii-ssrc-23.zip -d tii_ssrc23
# Extract dataset archive

# =========================================================
# GLOBAL CONFIGURATION
# =========================================================
csv_path = "tii_ssrc23/csv/data.csv"              # Path to feature CSV
RANDOM_STATE = 42                                 # Reproducibility seed
N_SPLITS = 5                                      # Number of CV folds
CLIENT_SIZE = 0.5                                 # Fraction of test data for client

# =========================================================
# DATA LOADING (10% SUBSAMPLING)
# =========================================================
print("📥 Loading dataset...")                    # User feedback

df_full = pd.read_csv(csv_path)                   # Load full CSV into memory
df = df_full.sample(frac=0.10, random_state=RANDOM_STATE)
# Randomly sample 10% to reduce memory usage

df = df.reset_index(drop=True)                    # Reset DataFrame indices
del df_full                                       # Free RAM

# =========================================================
# DATA CLEANING
# =========================================================
cols_to_drop = [                                  # Non-ML or identifier fields
    'Flow ID', 'Src IP', 'Src Port',
    'Dst IP', 'Dst Port', 'Protocol',
    'Timestamp', 'Traffic Subtype'
]

df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
# Remove irrelevant or leaking columns

df = df.replace([np.inf, -np.inf], np.nan)        # Replace infinite values
df = df.dropna()                                  # Remove rows with missing values
df = df.drop_duplicates()                         # Remove duplicate flows

# =========================================================
# LABEL ENCODING
# =========================================================
if df['Label'].dtype == 'object':                 # Check if label is textual
    df['Label'] = df['Label'].map({
        'Benign': 0,
        'Malicious': 1
    })                                            # Binary label encoding

# =========================================================
# ENCODE REMAINING CATEGORICAL FEATURES
# =========================================================
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])
    # Convert categorical fields to numerical form

print(f"Dataset ready: {df.shape}")               # Print dataset dimensions

# =========================================================
# TRAIN / TEST / CLIENT SPLIT
# =========================================================
X = df.drop(columns=['Label'])                    # Feature matrix
y = df['Label']                                  # Target vector

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.2,                               # 80% train, 20% temp
    stratify=y,                                  # Preserve class ratio
    random_state=RANDOM_STATE
)

X_test, X_client, y_test, y_client = train_test_split(
    X_temp, y_temp,
    test_size=CLIENT_SIZE,                       # Half test, half client
    stratify=y_temp,
    random_state=RANDOM_STATE
)

# =========================================================
# FEATURE SCALING AND BALANCING
# =========================================================
scaler = StandardScaler()                         # Z-score normalization
smote = SMOTE(random_state=RANDOM_STATE)          # Balance malicious class

X_train_scaled = scaler.fit_transform(X_train)    # Fit scaler on training data
X_test_scaled = scaler.transform(X_test)          # Apply same scaler to test

X_train_res, y_train_res = smote.fit_resample(
    X_train_scaled, y_train
)                                                 # Balance training set

# =========================================================
# MODEL TRAINING
# =========================================================
rf_model = RandomForestClassifier(
    n_estimators=200,                             # Number of trees
    random_state=RANDOM_STATE,
    n_jobs=-1                                     # Parallel processing
)

rf_model.fit(X_train_res, y_train_res)            # Train classifier

# =========================================================
# MODEL EVALUATION
# =========================================================
y_pred = rf_model.predict(X_test_scaled)          # Hard predictions
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
# Malicious probability

print(classification_report(y_test, y_pred))      # Precision/Recall/F1
print("AUC:", roc_auc_score(y_test, y_prob))      # ROC-AUC metric

# =========================================================
# SAVE MODEL & SCALER
# =========================================================
joblib.dump(rf_model, "random_forest_model.pkl")  # Persist trained model
joblib.dump(scaler, "scaler.pkl")                 # Persist scaler

# =========================================================
# CREATE CLIENT DATASET
# =========================================================
client_data = X_client.copy()                     # Copy client features
client_data['Label'] = y_client                  # Attach ground truth
client_data.to_csv("client_data.csv", index=False)
# Save CSV for client simulation

files.download("client_data.csv")                 # Download to local machine

# =========================================================
# EXPORT FULL FEATURE LIST (86 FEATURES)
# =========================================================
df_cols = pd.read_csv(csv_path, nrows=1).columns  # Read column headers only

with open("master_feature_list_86.txt", "w") as f:
    for c in df_cols:
        f.write(c + "\n")                         # Write each feature name

# =========================================================
# RANDOM PCAP SELECTION FOR SERVER TESTING
# =========================================================
BASE = "tii_ssrc23/pcap"                          # PCAP root directory
OUTPUT = "pcap_test_set"                          # Output folder

if os.path.exists(OUTPUT):
    shutil.rmtree(OUTPUT)                         # Remove previous folder

os.makedirs(OUTPUT, exist_ok=True)                # Create fresh directory

def pick_random_pcap(source_dir, output_dir, label):
    files = [f for f in os.listdir(source_dir) if f.endswith(".pcap")]
    if not files:
        return
    chosen = random.choice(files)                 # Random PCAP selection
    shutil.copy(
        os.path.join(source_dir, chosen),
        os.path.join(output_dir, f"{label}_{chosen}")
    )

# Benign traffic
pick_random_pcap(f"{BASE}/benign/audio", OUTPUT, "benign_audio")
pick_random_pcap(f"{BASE}/benign/background", OUTPUT, "benign_background")
pick_random_pcap(f"{BASE}/benign/text", OUTPUT, "benign_text")
pick_random_pcap(f"{BASE}/benign/video", OUTPUT, "benign_video")

# Malicious traffic
pick_random_pcap(f"{BASE}/malicious/bruteforce", OUTPUT, "mal_bruteforce")
pick_random_pcap(f"{BASE}/malicious/dos", OUTPUT, "mal_dos")
pick_random_pcap(f"{BASE}/malicious/information-gathering", OUTPUT, "mal_infogathering")
pick_random_pcap(f"{BASE}/malicious/mirai-botnet", OUTPUT, "mal_mirai")

# =========================================================
# ZIP PCAP TEST SET
# =========================================================
with zipfile.ZipFile("pcap_test_set.zip", "w") as z:
    for f in os.listdir(OUTPUT):
        z.write(os.path.join(OUTPUT, f), f)        # Add PCAPs to ZIP

print("PCAP test set ready.")
