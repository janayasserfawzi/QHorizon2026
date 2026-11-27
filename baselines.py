import os, csv, time
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from data_prep import SPLITS_DIR, load_saved_split

OUT = "results/person1_classical.csv"
COLUMNS = ["model", "encoding", "n_qubits", "depth", "n_train", "seed",
           "accuracy", "f1", "train_time", "task", "entanglement"]


class TunedModel:
    """fit/predict like the quantum models. Hyperparameters chosen by CV on TRAIN only."""
    def __init__(self, estimator, grid, seed):
        cv = StratifiedKFold(3, shuffle=True, random_state=seed)
        self.search = GridSearchCV(estimator, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    def fit(self, X, y):
        self.search.fit(X, y); return self
    def predict(self, X):
        return self.search.predict(X)


def make_models(seed):
    return {
        "svm_rbf": TunedModel(SVC(), {"C": [0.1, 1, 10, 100],
                              "gamma": [0.1, 0.5, 1, 2, "scale"]}, seed),
        "logreg": TunedModel(LogisticRegression(max_iter=5000),
                             {"C": [0.01, 0.1, 1, 10, 100]}, seed),
        "mlp": TunedModel(MLPClassifier(max_iter=2000, random_state=seed),
                          {"hidden_layer_sizes": [(8,), (16,), (8, 8)],
                           "alpha": [1e-4, 1e-2]}, seed),
    }


def evaluate(model, X_tr, X_te, y_tr, y_te):
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0
    pred = model.predict(X_te)
    return (accuracy_score(y_te, pred),
            f1_score(y_te, pred, average="macro"),
            train_time)


if __name__ == "__main__":
    manifest = pd.read_csv(os.path.join(SPLITS_DIR, "manifest.csv"))
    order = {"4class": 0, "4class_f4": 1, "binary_1v2": 2, "4class_f5": 3, "4class_f6": 4}
    manifest["o"] = manifest["task"].map(order).fillna(9)
    manifest = manifest.sort_values(["o", "n_train", "seed"])

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", newline="") as f:        # "w": fresh file every run, no duplicates
        w = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        w.writeheader()
        for _, r in manifest.iterrows():
            task, seed, n = r.task, int(r.seed), int(r.n_train)
            X_tr, X_te, y_tr, y_te = load_saved_split(seed, n, task)
            for name, m in make_models(seed).items():
                acc, f1, t = evaluate(m, X_tr, X_te, y_tr, y_te)
                w.writerow({"model": name, "encoding": "none", "n_qubits": 0,
                            "depth": 0, "n_train": n, "seed": seed,
                            "accuracy": acc, "f1": f1, "train_time": t, "task": task})
                f.flush()
                print(task, name, n, seed, round(acc, 3), round(f1, 3))