import joblib
import pandas as pd

model = joblib.load("cloud_model.pkl")

def predict_optimization(data):

    df = pd.DataFrame([{
        "cpu_usage": data.get("cpu_usage",0),
        "memory_usage": data.get("memory_usage",0),
        "disk_io": data.get("disk_io",0),
        "latency_ms": data.get("latency_ms",0),
        "throughput": data.get("throughput",0),
        "cost": data.get("cost",0)
    }])

    prediction = model.predict(df)[0]

    return prediction