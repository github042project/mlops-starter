# flows/pipeline.py
from prefect import flow, task
from src.preprocess import make_data, preprocess
from src.train import train_and_log
import pandas as pd

@task
def make_and_save(n_samples=1000):
    df = make_data(n_samples)
    df.to_csv("data/raw.csv", index=False)
    return "data/raw.csv"

@task
def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = preprocess(df)
    df.to_csv("data/processed.csv", index=False)
    return "data/processed.csv"

@task
def train_task():
    acc = train_and_log()
    return acc

@flow
def ml_pipeline():
    raw = make_and_save()
    proc = load_and_preprocess(raw)
    acc = train_task()
    return acc

if __name__ == "__main__":
    ml_pipeline()
