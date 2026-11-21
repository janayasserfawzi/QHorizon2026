"""
make_splits.py  --  Person 1

Creates the frozen train/test splits for every (task, seed, n_train) and saves
them to data/splits/. Commit that folder to the repo: everybody then loads the
SAME files with data_prep.load_saved_split(seed, n_train, task).

    python make_splits.py            # generate + sanity-check all splits
    python make_splits.py --verify   # check this machine reproduces the saved files
"""
import os
import sys

import numpy as np
import pandas as pd

from data_prep import SPLITS_DIR, load_raw, load_split, split_path, validate

# ---- The experiment grid: agree on it with P2 and P3, then do not change it ----
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN = [25, 50, 100, 200]
N_TEST = 200  # same size for every model, seed and training size
TASKS = {
    "4class":        dict(classes=None,   n_features=7),  # main task, 7 qubits
    "4class_f6":     dict(classes=None,   n_features=6),  # fewer qubits (Gold)
    "4class_f5":     dict(classes=None,   n_features=5),
    "4class_f4":     dict(classes=None,   n_features=4),
    "binary_1v2":    dict(classes=[1, 2], n_features=7),  # easy debug task (rooms 1 vs 2)
}
# -------------------------------------------------------------------------------


def generate():
    validate(load_raw())
    os.makedirs(SPLITS_DIR, exist_ok=True)
    rows = []
    for task, kw in TASKS.items():
        for seed in SEEDS:
            y_test_ref = None
            for n in N_TRAIN:
                Xtr, Xte, ytr, yte = load_split(seed=seed, n_train=n, n_test=N_TEST, **kw)

                # ---- sanity checks ----
                assert not (np.isnan(Xtr).any() or np.isnan(Xte).any()), "NaN found"
                assert Xtr.min() >= 0 and Xtr.max() <= np.pi + 1e-9, "train not in [0, pi]"
                assert Xte.min() >= 0 and Xte.max() <= np.pi + 1e-9, "test not in [0, pi]"
                assert len(ytr) == n and len(yte) == N_TEST, "wrong sizes"
                n_classes = len(np.unique(ytr))
                assert n_classes == len(np.unique(yte)), "class missing in train or test"
                # same test samples for every n_train within a seed
                if y_test_ref is None:
                    y_test_ref = yte
                assert np.array_equal(y_test_ref, yte), "test set changed with n_train"

                np.savez_compressed(split_path(task, seed, n),
                                    X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte)
                rows.append(dict(task=task, seed=seed, n_train=n, n_test=N_TEST,
                                 n_features=Xtr.shape[1], n_classes=n_classes,
                                 min_per_class_train=int(np.bincount(ytr).min())))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(SPLITS_DIR, "manifest.csv"), index=False)
    print(f"Saved {len(rows)} splits to {SPLITS_DIR}")
    print(f"Smallest class in any training set: {manifest.min_per_class_train.min()} samples")
    print(manifest.groupby("task").size().to_string())


def verify():
    """Re-create every split on THIS machine and compare with the saved files.
    A mismatch usually means a different scikit-learn / numpy version."""
    manifest = pd.read_csv(os.path.join(SPLITS_DIR, "manifest.csv"))
    bad = 0
    for _, r in manifest.iterrows():
        kw = TASKS[r.task]
        new = load_split(seed=int(r.seed), n_train=int(r.n_train), n_test=int(r.n_test), **kw)
        old = np.load(split_path(r.task, int(r.seed), int(r.n_train)))
        same = (np.allclose(new[0], old["X_train"]) and np.allclose(new[1], old["X_test"])
                and np.array_equal(new[2], old["y_train"]) and np.array_equal(new[3], old["y_test"]))
        if not same:
            bad += 1
            print("MISMATCH:", r.task, "seed", r.seed, "n", r.n_train)
    if bad:
        print(f"\n{bad}/{len(manifest)} splits differ. Use the committed files via "
              f"load_saved_split() and align library versions (requirements.txt).")
        sys.exit(1)
    print(f"All {len(manifest)} splits reproduce exactly on this machine.")


if __name__ == "__main__":
    verify() if "--verify" in sys.argv else generate()
