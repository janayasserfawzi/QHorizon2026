"""
data_prep.py  --  Person 1 (data + fair comparison lead)

Shared data module for the Q//Horizon project. EVERY model (quantum or classical)
must get its data from load_split() so the comparison is fair.

Dataset: Wireless Indoor Localization (UCI id 422)
  Bhatt, R. (2017). https://doi.org/10.24432/C51880
  2000 samples, 7 WiFi signal-strength features, 4 rooms.

Usage:
    from data_prep import load_split
    X_train, X_test, y_train, y_test = load_split(seed=0, n_train=200)

    # binary version (rooms 1 and 2 only), 5 qubits
    load_split(seed=0, n_train=100, classes=[1, 2], n_features=5)

Run this file directly to download, validate and sanity-check the data:
    python data_prep.py
"""
import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_CSV = os.path.join(DATA_DIR, "wifi_localization.csv")
ZIP_URL = "https://archive.ics.uci.edu/static/public/422/wireless+indoor+localization.zip"
FEATURE_NAMES = [f"wifi_{i}" for i in range(1, 8)]


def load_raw() -> pd.DataFrame:
    """Return the full dataset as a DataFrame: wifi_1..wifi_7 + 'room' (1..4).
    Tries (1) local cache, (2) ucimlrepo, (3) direct UCI zip download."""
    if os.path.exists(CACHE_CSV):
        return pd.read_csv(CACHE_CSV)

    os.makedirs(DATA_DIR, exist_ok=True)
    df = None

    try:  # option 1: ucimlrepo
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=422)
        X, y = ds.data.features.copy(), ds.data.targets.copy()
        X.columns = FEATURE_NAMES
        X["room"] = y.iloc[:, 0].values
        df = X
    except Exception as e:  # network, package missing, etc.
        print(f"[data_prep] ucimlrepo failed ({e}); trying direct download...")

    if df is None:  # option 2: raw zip from UCI
        raw = urllib.request.urlopen(ZIP_URL, timeout=30).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = [n for n in z.namelist() if n.endswith(".txt")][0]
            df = pd.read_csv(z.open(name), sep=r"\s+", header=None)
        df.columns = FEATURE_NAMES + ["room"]

    df.to_csv(CACHE_CSV, index=False)
    return df


def validate(df: pd.DataFrame) -> None:
    """Fail loudly if the data is not what we expect."""
    assert df.shape == (2000, 8), f"unexpected shape {df.shape}"
    assert not df.isna().any().any(), "missing values found"
    assert set(df["room"].unique()) == {1, 2, 3, 4}, "unexpected room labels"


def load_split(seed: int = 0, n_train: int = 200, n_test: int = 200,
               classes=None, n_features: int = 7):
    """Stratified train/test split + scaling, identical for every model.

    seed        : random seed (we use 0..4 everywhere)
    n_train     : training samples (stratified subsample)
    n_test      : test samples (stratified, same size for every model)
    classes     : e.g. [1, 2] for a binary task; None = all 4 rooms
    n_features  : keep the k most informative features (ANOVA F-score,
                  fit on TRAIN only) = number of qubits for angle encoding

    Returns X_train, X_test, y_train, y_test (labels remapped to 0..k-1).
    Features are scaled to [0, pi] using TRAIN statistics only (no leakage).
    """
    df = load_raw()
    if classes is not None:
        df = df[df["room"].isin(classes)]
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["room"].to_numpy()

    labels = np.sort(np.unique(y))
    y = np.searchsorted(labels, y)  # remap to 0..k-1

    # 1) fixed-size stratified test set, 2) stratified training subsample
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=n_test, stratify=y, random_state=seed)
    if n_train < len(y_pool):
        X_train, _, y_train, _ = train_test_split(
            X_pool, y_pool, train_size=n_train, stratify=y_pool, random_state=seed)
    else:
        X_train, y_train = X_pool, y_pool

    # feature selection (train only)
    if n_features < X_train.shape[1]:
        sel = SelectKBest(f_classif, k=n_features).fit(X_train, y_train)
        X_train, X_test = sel.transform(X_train), sel.transform(X_test)

    # scale to [0, pi] using train statistics; clip test to the same range
    scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = np.clip(scaler.transform(X_test), 0, np.pi)
    return X_train, X_test, y_train, y_test


SPLITS_DIR = os.path.join(DATA_DIR, "splits")


def split_path(task: str, seed: int, n_train: int) -> str:
    return os.path.join(SPLITS_DIR, f"{task}_seed{seed}_n{n_train}.npz")


def load_saved_split(seed: int, n_train: int, task: str = "4class"):
    """OFFICIAL interface for teammates: read a frozen split from data/splits/.
    Same return as load_split(): X_train, X_test, y_train, y_test.
    Created by `python make_splits.py` and committed to the repo."""
    path = split_path(task, seed, n_train)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Pull data/splits/ from the repo, or run "
            f"`python make_splits.py` (check that task/seed/n_train are in the grid).")
    d = np.load(path)
    return d["X_train"], d["X_test"], d["y_train"], d["y_test"]


if __name__ == "__main__":
    df = load_raw()
    validate(df)
    print("Shape:", df.shape)
    print("\nSamples per room:\n", df["room"].value_counts().sort_index())
    print("\nFeature summary:\n", df[FEATURE_NAMES].describe().round(1))

    # sanity check: a classical SVM should score very high on this data
    from sklearn.svm import SVC
    accs = []
    for seed in range(5):
        Xtr, Xte, ytr, yte = load_split(seed=seed, n_train=200)
        accs.append(SVC(kernel="rbf").fit(Xtr, ytr).score(Xte, yte))
    print(f"\nSanity check, RBF-SVM, 200 train, 5 seeds: "
          f"{np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    Xtr, Xte, ytr, yte = load_split(seed=0, n_train=100, classes=[1, 2], n_features=5)
    print("Binary / 5-feature shapes:", Xtr.shape, Xte.shape,
          "| range:", round(Xtr.min(), 2), "to", round(Xtr.max(), 2))
