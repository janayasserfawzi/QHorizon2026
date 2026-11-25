
import os
import sys
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_prep import load_saved_split
from Kernel.kernel import QuantumKernelSVM

RESULTS_CSV = os.path.join(REPO_ROOT, "results", "person2_kernel.csv")
DIAG_DIR = os.path.join(REPO_ROOT, "results", "kernel_diagnostics")


def export_gram_matrices():
    """Save Gram matrices for seed=0, n_train=100 across qubit counts & encodings."""
    os.makedirs(DIAG_DIR, exist_ok=True)
    for task in ["4class_f4", "4class_f6", "4class"]:
        X_train, _, y_train, _ = load_saved_split(seed=0, n_train=100, task=task)
        # Sort samples by room label so block-diagonal structure is visible in heatmaps
        order = np.argsort(y_train)
        X_sorted = X_train[order]
        y_sorted = y_train[order]

        for enc, depth in [("angle", 1), ("zz", 1), ("zz", 2)]:
            model = QuantumKernelSVM(encoding=enc, depth=depth, entanglement="linear")
            model.fit(X_sorted, y_sorted)
            fname = f"gram_{task}_{enc}_d{depth}.npy"
            np.save(os.path.join(DIAG_DIR, fname), model.last_K_train_)
            stats = model.kernel_concentration_stats()
            print(f"Saved {fname} | off-diag mean={stats['k_offdiag_mean']:.4f}, std={stats['k_offdiag_std']:.4f}")


def export_summary_table():
    """Aggregate mean +/- std across seeds for slides & report tables."""
    if not os.path.exists(RESULTS_CSV):
        print(f"No results found at {RESULTS_CSV} yet.")
        return
    df = pd.read_csv(RESULTS_CSV)
    summary = (
        df.groupby(["task", "encoding", "n_qubits", "depth", "n_train"], as_index=False)
        .agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            time_mean=("train_time", "mean"),
            offdiag_std_mean=("k_offdiag_std", "mean"),
            n_seeds=("seed", "count"),
        )
    )
    out_path = os.path.join(DIAG_DIR, "person2_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved aggregated summary to {out_path}:\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    export_gram_matrices()
    export_summary_table()