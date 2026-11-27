"""
check_C_tuning.py -- Person 4
The quantum-kernel SVM uses a fixed C=1 while the classical models are tuned by CV.
Does tuning C (3-fold CV on the TRAINING set only, C in 0.1/1/10/100) change the kernel results?
Runs the 4-qubit task, n_train 25 and 100, seeds 0-4 -> results/c_tuning_p4.csv (about 1 min).
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from data_prep import load_saved_split
from Kernel.kernel import QuantumKernelSVM

rows = []
for enc, ent in (("angle", "none"), ("zz", "linear")):
    for n in (25, 100):
        for seed in range(5):
            Xtr, Xte, ytr, yte = load_saved_split(seed, n, "4class_f4")
            m = QuantumKernelSVM(enc, 1, ent).fit(Xtr, ytr)
            Ktr = m.last_K_train_
            Kte = np.abs(m._encode_states(Xte) @ m.train_states_.conj().T) ** 2
            fixed = accuracy_score(yte, SVC(kernel="precomputed", C=1).fit(Ktr, ytr).predict(Kte))
            gs = GridSearchCV(SVC(kernel="precomputed"), {"C": [0.1, 1, 10, 100]},
                              cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                              scoring="f1_macro").fit(Ktr, ytr)
            rows.append(dict(encoding=enc, n_train=n, seed=seed, acc_C1=fixed,
                             acc_tuned=accuracy_score(yte, gs.predict(Kte)), chosen_C=gs.best_params_["C"]))
d = pd.DataFrame(rows)
d.to_csv(os.path.join(ROOT, "results", "c_tuning_p4.csv"), index=False)
s = d.groupby(["encoding", "n_train"])[["acc_C1", "acc_tuned"]].mean()
s["tuned_minus_C1"] = s.acc_tuned - s.acc_C1
print(s.round(3).to_string())
