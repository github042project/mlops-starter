# src/train.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.preprocess import make_data, preprocess
from src.utils import save_model
import os

def train_and_log(n_estimators=50, random_state=42):
    # 1. make & preprocess
    df = make_data(n_samples=1000, random_state=random_state)
    df = preprocess(df)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # 2. create model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # 3. Start MLflow run
    mlflow.set_experiment("mlops_starter_experiment")
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        mlflow.log_metric("val_accuracy", acc)

        # 4. log model to MLflow and local store
        mlflow.sklearn.log_model(model, "model")
        # also save locally so FastAPI can load without MLflow model server
        save_model(model, path=os.path.join("app", "model_store", "model.joblib"))

    print(f"Training complete. Validation accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    train_and_log()
