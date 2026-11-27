
import argparse
import os
import sys
import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_prep import load_saved_split
from Kernel.kernel import QuantumKernelSVM

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "person2_kernel.csv")

ALL_SEEDS = [0, 1, 2, 3, 4]
ALL_N_TRAIN = [25, 50, 100, 200]


def build_experiment_plan(stage: str):
    """Returns a list of experiment dicts (task, encoding, depth, entanglement, n_train, seed)."""
    plan = []

    if stage == "debug":
       
        for enc in ["angle", "zz"]:
            for n in [25, 50, 100]:
                plan.append(
                    dict(
                        task="binary_1v2",
                        encoding=enc,
                        depth=1,
                        entanglement="linear",
                        n_train=n,
                        seed=0,
                    )
                )
        return plan

    if stage in ("binary", "all"):
        # Complete binary_1v2 grid across all seeds & training sizes for fair comparison
        for seed in ALL_SEEDS:
            for n in ALL_N_TRAIN:
                plan.append(
                    dict(
                        task="binary_1v2",
                        encoding="angle",
                        depth=1,
                        entanglement="none",
                        n_train=n,
                        seed=seed,
                    )
                )
                plan.append(
                    dict(
                        task="binary_1v2",
                        encoding="zz",
                        depth=1,
                        entanglement="linear",
                        n_train=n,
                        seed=seed,
                    )
                )

    if stage in ("main", "all"):
        # Main 7-qubit task ("4class"): compare Angle vs ZZ across all seeds & training sizes
        for seed in ALL_SEEDS:
            for n in ALL_N_TRAIN:
                plan.append(
                    dict(
                        task="4class",
                        encoding="angle",
                        depth=1,
                        entanglement="none",
                        n_train=n,
                        seed=seed,
                    )
                )
                plan.append(
                    dict(
                        task="4class",
                        encoding="zz",
                        depth=1,
                        entanglement="linear",
                        n_train=n,
                        seed=seed,
                    )
                )
                plan.append(
                    dict(
                        task="4class",
                        encoding="zz",
                        depth=2,
                        entanglement="linear",
                        n_train=n,
                        seed=seed,
                    )
                )

    if stage in ("qubits", "all"):
        # Varying number of qubits (4, 5, 6 qubits) across encodings, seeds & training sizes
        for task in ["4class_f4", "4class_f5", "4class_f6"]:
            for seed in ALL_SEEDS:
                for n in ALL_N_TRAIN:
                    plan.append(
                        dict(
                            task=task,
                            encoding="angle",
                            depth=1,
                            entanglement="none",
                            n_train=n,
                            seed=seed,
                        )
                    )
                    plan.append(
                        dict(
                            task=task,
                            encoding="zz",
                            depth=1,
                            entanglement="linear",
                            n_train=n,
                            seed=seed,
                        )
                    )

    return plan


def load_completed_keys(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    return set(
        zip(
            df["task"],
            df["encoding"],
            df["depth"].astype(int),
            df["entanglement"],
            df["n_train"].astype(int),
            df["seed"].astype(int),
        )
    )


def run_single(cfg: dict) -> dict:
    X_train, X_test, y_train, y_test = load_saved_split(
        seed=cfg["seed"],
        n_train=cfg["n_train"],
        task=cfg["task"],
    )
    n_qubits = int(X_train.shape[1])

    model = QuantumKernelSVM(
        encoding=cfg["encoding"],
        depth=cfg["depth"],
        entanglement=cfg["entanglement"] if cfg["encoding"] == "zz" else "linear",
        C=1.0,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro"))
    conc = model.kernel_concentration_stats()

    return {
        "model": f"QKernel-{cfg['encoding']}",
        "encoding": cfg["encoding"],
        "n_qubits": n_qubits,
        "depth": cfg["depth"],
        "n_train": cfg["n_train"],
        "seed": cfg["seed"],
        "accuracy": acc,
        "f1": f1,
        "train_time": train_time,
        "task": cfg["task"],
        "entanglement": cfg["entanglement"],
        "k_offdiag_mean": conc["k_offdiag_mean"],
        "k_offdiag_std": conc["k_offdiag_std"],
    }


def main():
    parser = argparse.ArgumentParser(description="Person 2: Quantum Kernel SVM Experiments")
    parser.add_argument(
        "--stage",
        choices=["debug", "binary", "main", "qubits", "all"],
        default="debug",
        help="Which experiment sweep to run (default: debug)",
    )
    parser.add_argument("--out", default=OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plan = build_experiment_plan(args.stage)
    completed = load_completed_keys(args.out)

    pending = [
        c
        for c in plan
        if (
            c["task"],
            c["encoding"],
            c["depth"],
            c["entanglement"],
            c["n_train"],
            c["seed"],
        )
        not in completed
    ]

    print(
        f"[Stage: {args.stage}] Total planned: {len(plan)} | "
        f"Already done: {len(plan) - len(pending)} | Remaining: {len(pending)}"
    )

    for idx, cfg in enumerate(pending, start=1):
        row = run_single(cfg)
        pd.DataFrame([row]).to_csv(
            args.out,
            mode="a",
            header=not os.path.exists(args.out),
            index=False,
        )
        print(
            f"[{idx}/{len(pending)}] task={cfg['task']} enc={cfg['encoding']} "
            f"q={row['n_qubits']} d={cfg['depth']} n={cfg['n_train']} seed={cfg['seed']} "
            f"-> acc={row['accuracy']:.3f}, f1={row['f1']:.3f}, time={row['train_time']:.2f}s"
        )


if __name__ == "__main__":
    main()