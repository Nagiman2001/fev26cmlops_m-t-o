# Importation des modules
from pathlib import Path
import sqlite3
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# NEW : Importation pour l'utilisation de MFLow
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


# --- DEFINITION DE VARIABLES ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DB_PATH = DATA_DIR / "weather.db"
TABLE_NAME = "weather_data"
MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"

EXPERIMENT_NAME = "weather_prediction"
MODEL_NAME = "WeatherRandomForest"

#  VARIABLES UTILISÉES POUR L'ENTRAÎNEMENT DU MODÈLE
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
# --------------------------------------------------------------------------------------


# --- LOADING DATA ---------------------------------------------------------------------
def load_data():
    """ Chargement de la Base de données weather.db"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df
# --------------------------------------------------------------------------------------


# --- MAIN -----------------------------------------------------------------------------
def main():
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Création d'un espace de travail pour MFLOW, pour stocker les runs
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Chargement des données
    df = load_data()

    # Séparation des variables explicatives et target
    X = df[FEATURES]
    y = df["RainTomorrow"]

    # Train Test SPLIT avec une taille de test : 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Démarrage d'un run MLflow = un entrainement (fit) avec MFLow
    with mlflow.start_run():

        # Instanciation du modèle
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

        # FIT !!!
        model.fit(X_train, y_train)

        # Prédiction + Métrique F1-Score
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        # Printing des résultats
        print("F1-score :", round(f1 , 4))
        print(classification_report(y_test, y_pred))

        # CODE DE LOGGING MLFLOW
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("features", FEATURES)
        mlflow.log_metric("f1_score", f1)                                                       # Métrique F1
        mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")     # Artefacts : rapport de classification 
        mlflow.sklearn.log_model(model, "model", registered_model_name=MODEL_NAME)              # Versionning

        # Sauvegarde locale
        joblib.dump(model, MODEL_PATH)
        

    # COMPARAISON des PERFORMANCES ENTRE MODELES
    # Instanciation pour intérargir avec le Model Registry
    client = MlflowClient()

    # Récupération des versions de models
    models_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])

    # si le modèle existe
    if models_versions:
        
        # Chargement du modèle en production
        model_version = models_versions[0]
        model_prod = mlflow.sklearn.load_model(model_version.source)
        
        y_pred_old = model_prod.predict(X_test)
        old_f1 = f1_score(y_test, y_pred_old)

        print(f"Ancien modèle F1 : {round(old_f1, 4)}")
        print(f"Nouveau modèle F1 : {round(f1, 4)}")

        if f1 > old_f1:

            print("Le nouveau modèle est meilleur que l'ancien")

            # Archiver ancien modèle
            client.transition_model_version_stage(name=MODEL_NAME, version=model_version.version, stage="Archived")

            # Récupérer dernière version
            latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[-1]

            # Permuter avec le nouveau modèle
            client.transition_model_version_stage(name=MODEL_NAME, version=latest_version.version, stage="Production")

    else:
        # Si aucun modèle en production
        latest_version = client.get_latest_versions(MODEL_NAME)[-1]
        client.transition_model_version_stage(name=MODEL_NAME, version=latest_version.version, stage="Production")


# Lancement du script
if __name__ == "__main__":
    main()