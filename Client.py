import pandas as pd
import requests
import time

SERVER_URL = "http://127.0.0.1:8080/predict"

# carica dataset client
df = pd.read_csv("client_data.csv")

print(f"📤 Inviando {len(df)} richieste al server...")

for i, row in df.iterrows():
    features = row.drop("Label").tolist()

    response = requests.post(SERVER_URL, json={"features": features})
    result = response.json()

    prediction = result["prediction"]
    prob = result["probability"]

    print(f"[{i}] Server → {prediction} (p={prob:.3f})")

    # simula traffico
    time.sleep(0.3)
