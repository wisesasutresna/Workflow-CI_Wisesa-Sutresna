import os
import argparse
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-interaktif
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def load_data():
    print("Membaca dataset preprocessed...")
    try:
        return pd.read_csv("diabetes_preprocessing.csv")
    except FileNotFoundError:
        return pd.read_csv("diabetes_preprocessing")

def train_and_eval(df, random_state, learning_rate):
    # Setup experiment jika dijalankan di luar mlflow run CLI
    is_mlflow_run = "MLFLOW_RUN_ID" in os.environ
    if not is_mlflow_run:
        mlflow.set_experiment("Diabetes Prediction - Wisesa")
        mlflow.start_run(run_name="GB_CI_Retrain")
        
    mlflow.sklearn.autolog()
    
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    print(f"Melatih Gradient Boosting dengan random_state={random_state}, learning_rate={learning_rate}...")
    try:
        model = GradientBoostingClassifier(
            random_state=random_state, 
            learning_rate=learning_rate
        )
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Akurasi Model Retrained: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds))
    finally:
        if not is_mlflow_run:
            mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state", type=int, default=7)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()
    
    dataset = load_data()
    train_and_eval(dataset, args.random_state, args.learning_rate)
