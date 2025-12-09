# src/preprocess.py
import pandas as pd
from sklearn.datasets import make_classification

def make_data(n_samples=1000, random_state=42):
    # make a tiny classification dataset
    X, y = make_classification(n_samples=n_samples, n_features=8, n_informative=5,
                               n_redundant=0, random_state=random_state)
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    return df

def preprocess(df):
    # very small preprocessing: ensure no NaNs and scale features simply
    df = df.copy()
    df.fillna(0, inplace=True)
    # add a simple normalized version as example
    for c in df.columns:
        if c != "target":
            df[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-9)
    return df
