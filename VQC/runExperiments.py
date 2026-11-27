"""Person 3 experiments. Appends to results/person3_vqc.csv (resumable).
  python VQC/runExperiments.py --stage debug   # ~1 min sanity check, binary task
  python VQC/runExperiments.py --stage depth   # depth 1-4, Adam, 4class
  python VQC/runExperiments.py --stage optim   # Adam vs SGD vs COBYLA at depth 2
  python VQC/runExperiments.py --stage deep    # depth 5-6, Adam
  python VQC/runExperiments.py --stage all
"""
import argparse, os, sys
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import load_saved_split
from vqc import VQC

OUT = os.path.join(REPO_ROOT, "results", "person3_vqc.csv")
SEEDS, NS = [0, 1, 2, 3, 4], [25, 50, 100, 200]
KEY = ["task", "optimizer", "depth", "n_train", "seed"]
N_QUBITS = {"4class": 7, "4class_f6": 6, "4class_f5": 5, "4class_f4": 4, "binary_1v2": 7}
N_CLASSES = {"binary_1v2": 2}


def plan(stage):
    p = []
    if stage == "debug":
        return [dict(task="binary_1v2", optimizer="adam", depth=d, n_train=n, seed=0)
                for d in (1, 2) for n in (25, 50)]
    if stage in ("depth", "all"):
        p += [dict(task="4class", optimizer="adam", depth=d, n_train=n, seed=s)
              for d in (1, 2, 3, 4) for n in NS for s in SEEDS]
    if stage in ("optim", "all"):
        p += [dict(task="4class", optimizer=o, depth=2, n_train=n, seed=s)
              for o in ("sgd", "cobyla") for n in NS for s in SEEDS]
    if stage in ("deep", "all"):
        p += [dict(task="4class", optimizer="adam", depth=d, n_train=n, seed=s)
              for d in (5, 6) for n in NS for s in SEEDS]
    return p


ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="debug", choices=["debug", "depth", "optim", "deep", "all"])
a = ap.parse_args()

done = set()
if os.path.exists(OUT):
    d = pd.read_csv(OUT)
    done = set(map(tuple, d[KEY].values.tolist()))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
for e in plan(a.stage):
    if tuple(e[k] for k in KEY) in done:
        continue
    Xtr, Xte, ytr, yte = load_saved_split(e["seed"], e["n_train"], e["task"])
    nq, nc = N_QUBITS[e["task"]], N_CLASSES.get(e["task"], 4)
    m = VQC(nq, e["depth"], e["optimizer"], seed=e["seed"], n_classes=nc).fit(Xtr, ytr)
    pred = m.predict(Xte)
    row = dict(model=f"VQC-{e['optimizer']}", encoding="angle", n_qubits=nq,
               depth=e["depth"], n_train=e["n_train"], seed=e["seed"],
               accuracy=accuracy_score(yte, pred),
               f1=f1_score(yte, pred, average="macro"), train_time=m.train_time,
               task=e["task"], optimizer=e["optimizer"])
    pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    print(row["model"], e["task"], e["depth"], e["n_train"], e["seed"],
          round(row["accuracy"], 3), f"{row['train_time']:.1f}s", flush=True)
