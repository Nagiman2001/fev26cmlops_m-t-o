# Importation des modules
import numpy as np
import joblib
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# NEW : Importation pour l'utilisation de MFLOW
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Connexion PostgreSQL via SQLAlchemy
from sqlalchemy import create_engine


# --- DEFINITION DE VARIABLES ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

TABLE_NAME = "weather_data"
MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"

EXPERIMENT_NAME = "weather_prediction"
MODEL_NAME = "WeatherRandomForest"

#  Features sélectionnées POUR L'ENTRAÎNEMENT DU MODÈLE
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


# Configuration de connexion à POSTGRESQL et MLFLOW --------------------------------------
DB_USER = "postgres"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "weather_db"

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

MLFLOW_TRACKING_URI = "http://localhost:5000"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# --- Bruit Gaussien sur les features pour vérier les métriques ------------------------
def add_noise(df, features, noise_level=0.05):
    """Ajoute un bruit gaussien aux colonnes numériques pour simuler la variabilité des données."""
    df_noisy = df.copy()
    for col in features:
        df_noisy[col] += np.random.normal(0, noise_level * df[col].std(), size=len(df))
    return df_noisy


# --- LOADING DATA depuis POSTGRESQL --------------------------------------------------
def load_data():
    query = f"SELECT * FROM {TABLE_NAME}"
    df = pd.read_sql(query, engine)
    return df


# --- MAIN -----------------------------------------------------------------------------
def main():

    # Chargement des données PostGreSQL, sécurité Dropna et ajout de bruit
    df = load_data()
    df = add_noise(df, FEATURES, noise_level=0.05)

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

        # LOGGING MLFLOW
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
    models_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    # si le modèle existe
    if models_versions:

        # Chargement du modèle en production
        model_prod = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")

        y_pred_old = model_prod.predict(X_test)
        old_f1 = f1_score(y_test, y_pred_old)

        print(f"Ancien modèle F1 : {round(old_f1, 4)}")
        print(f"Nouveau modèle F1 : {round(f1, 4)}")

        if f1 > old_f1:

            print("Le nouveau modèle est meilleur que l'ancien")

            # dernière version enregistrée
            latest_version = max(models_versions, key=lambda v: int(v.version))

            # Archiver ancien modèle
            client.transition_model_version_stage(name=MODEL_NAME, version=latest_version.version, stage="Archived")

            # récupérer version suivante
            new_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            latest_version = max(new_versions, key=lambda v: int(v.version))

            # Permuter avec le nouveau modèle
            client.transition_model_version_stage(name=MODEL_NAME, version=latest_version.version, stage="Production")

    else:
        # Si aucun modèle en production
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest_version = max(versions, key=lambda v: int(v.version))
        client.transition_model_version_stage(name=MODEL_NAME, version=latest_version.version, stage="Production")

# Lancement du script
if __name__ == "__main__":
    main()
