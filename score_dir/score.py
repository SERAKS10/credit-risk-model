import json
import joblib
import numpy as np
import pandas as pd
import os

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    model = joblib.load(model_path)
    print("Model loaded successfully!")

def run(raw_data):
    data = json.loads(raw_data)
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:,1]
    return {
        "predictions": predictions.tolist(),
        "default_probability": probabilities.tolist()
    }
