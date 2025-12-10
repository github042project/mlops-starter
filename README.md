MLOps Starter Project

A complete end-to-end Machine Learning workflow with tracking, deployment, orchestration, and reproducibility

<p align="center"> <img src="https://img.shields.io/badge/Framework-MLOps-red?style=for-the-badge" /> <img src="https://img.shields.io/badge/Tracking-MLflow-blue?style=for-the-badge" /> <img src="https://img.shields.io/badge/API-FastAPI-green?style=for-the-badge" /> <img src="https://img.shields.io/badge/Container-Docker-blue?style=for-the-badge" /> <img src="https://img.shields.io/badge/Orchestration-Airflow-purple?style=for-the-badge" /> </p>
🚀 Project Overview

This project is a beginner-friendly yet industry-relevant MLOps Starter Template designed to help you learn:

✔️ ML workflow structure

✔️ MLflow experiment tracking

✔️ Model training + evaluation

✔️ Production-ready FastAPI endpoint

✔️ Packaging model inside Docker

✔️ Local MLOps pipeline simulation

This acts as a foundation for real-world MLOps Engineer / ML Engineer roles.

## 📁 Project Architecture

```bash
mlops-starter/
│
├── app/                     # FastAPI application for inference
│   ├── main.py              # API endpoints
│   └── model_store/         # Saved MLflow model (after training)
│
├── src/                     # Core ML pipeline
│   ├── preprocess.py        # Feature engineering
│   ├── train.py             # Model training + MLflow logging
│   └── utils.py             # Helper functions
│
├── data/                    # Raw and processed datasets
│
├── Dockerfile               # Docker container definition
├── requirements.txt         # Python package dependencies
├── mlflow.db                # Local MLflow SQLite database
├── README.md                # Project documentation
└── .gitignore               # Ignore unnecessary files
```



✨ Features
✔️ Full ML Lifecycle

Preprocessing

Training

Metrics logging

Model versioning

Saving production model

✔️ MLflow Tracking

Parameters

Metrics

Artifacts

Auto-save models

Run comparison UI

✔️ FastAPI Deployment

Simple /predict endpoint

JSON input → Model inference

✔️ Dockerized Deployment

Build image

Run container

Production-ready
| Component        | Technology                   |
| ---------------- | ---------------------------- |
| ML Framework     | Scikit-Learn                 |
| API Layer        | FastAPI                      |
| Tracking         | MLflow                       |
| Orchestration    | Prefect / Airflow (optional) |
| Containerization | Docker                       |
| Language         | Python                       |

🛠️ Setup Instructions

1️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run MLflow UI
python -m mlflow ui


Open in browser →
👉 http://127.0.0.1:5000

4️⃣ Train the model
python src/train.py


This will:

✔️ Train a model
✔️ Save metrics
✔️ Save trained model in app/model_store/

5️⃣ Start FastAPI Server
uvicorn app.main:app --reload --port 8000


API → http://127.0.0.1:8000/docs

🔮 Predict Using API

Make prediction using:

curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"features":[0.1, -0.2, 0.3, 0.5]}'

🐳 Docker Deployment
Build image
docker build -t mlops-starter:latest .

Run container
docker run -p 8000:8000 --name mlops-starter mlops-starter:latest


API Docs inside Docker container →
👉 http://127.0.0.1:8000/docs

📂 MLflow Tracking

MLflow stores:

Experiments

Models

Metrics

Artifacts

Stored in:

mlruns/
mlflow.db

📈 Example MLflow Dashboard
Runs
├── metrics: accuracy, loss
├── parameters: model_type, hyperparameters
├── artifacts: saved model


Compare runs visually in your browser.

🤝 Contributing

Pull requests and improvements are welcome!
This project is designed to grow into a full MLOps pipeline using:

Airflow

Prefect

Kubernetes

Model Registry

CI/CD

🧑‍💻 Author

Garvita Varshney
💼 Data Science & MLOps Enthusiast
🌐 github.com/github042project
📧 garvitavarshney042@gmail.com


