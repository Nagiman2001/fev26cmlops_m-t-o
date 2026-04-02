# Importation des modules
from pathlib import Path
import pandas as pd
import mlflow.sklearn

# Définition des variables
MODEL_NAME = "WeatherRandomForest"

# mêmes features que training
FEATURES = [
    "Humidity3pm",
    "Humidity9am",
    "Rainfall",
    "WindGustSpeed",
    "Pressure3pm",
    "MaxTemp",
    "Temp3pm",
    "Year",
    "Month"
]


def load_model():    
    """ Charge le modèle depuis MLflow Registry en stage Production """
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")


def predict(data: dict):
    """ Retourne prediction et probabilité """
    model = load_model()

    df = pd.DataFrame([data])

    # Sécurité : ordre des colonnes
    df = df[FEATURES]

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(proba)
    }


# Test 
if __name__ == "__main__":
    sample_data = {
        "Humidity3pm": 70,
        "Humidity9am": 80,
        "Rainfall": 0.0,
        "WindGustSpeed": 35,
        "Pressure3pm": 1012,
        "MaxTemp": 25,
        "Temp3pm": 23,
        "Year": 2026,
        "Month": 4
    }

    result = predict(sample_data)
    print(result)
    