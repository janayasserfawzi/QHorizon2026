"""
noise_experiments.py -- Person 4

Question: how much does hardware-like noise (plus finite shots) hurt the quantum kernel?

Uses Person 2's QuantumKernelSVM(fidelity=...) hook. The kernel entry K(x, x') is estimated
by running U(x')^dagger U(x) on a noisy simulator and measuring P(all zeros).
Results are APPENDED to results/noise_p4.csv with the shared contract columns plus
extras (task, noise_p2, shots), so a re-run skips finished rows.

    python Person4/noise_experiments.py --stage debug   # ~1 min, checks the pipeline
    python Person4/noise_experiments.py --stage main    # the real sweep
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from data_prep import load_saved_split
from Kernel.kernel import QuantumKernelSVM

OUT_CSV = os.path.join(REPO_ROOT, "results", "noise_p4.csv")
BASIS = ["rz", "sx", "x", "cx"]


def make_noise_model(p2: float) -> NoiseModel:
    """Depolarizing noise: 2-qubit gate error p2, 1-qubit gate error p2/10, readout error p2."""
    nm = NoiseModel(basis_gates=BASIS)
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["rz", "sx", "x"])
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
        nm.add_all_qubit_readout_error(ReadoutError([[1 - p2, p2], [p2, 1 - p2]]))
    return nm


def make_fidelity(p2: float, shots: int, seed: int) -> ComputeUncompute:
    backend = AerSimulator(noise_model=make_noise_model(p2), basis_gates=BASIS, seed_simulator=seed)
    sampler = SamplerV2(default_shots=shots, seed=seed,
                        options=dict(backend_options=dict(noise_model=make_noise_model(p2),
                                                          basis_gates=BASIS)))
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    return ComputeUncompute(sampler=sampler, pass_manager=pm)


def run_one(cfg: dict) -> dict:
    Xtr, Xte, ytr, yte = load_saved_split(seed=cfg["seed"], n_train=cfg["n_train"], task=cfg["task"])
    model = QuantumKernelSVM(encoding=cfg["encoding"], depth=cfg["depth"],
                             entanglement="linear",
                             fidelity=make_fidelity(cfg["noise_p2"], cfg["shots"], cfg["seed"]))
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    train_time = time.perf_counter() - t0
    pred = model.predict(Xte)
    return {
        "model": f"QKernel-{cfg['encoding']}-noisy",
        "encoding": cfg["encoding"], "n_qubits": int(Xtr.shape[1]), "depth": cfg["depth"],
        "n_train": cfg["n_train"], "seed": cfg["seed"],
        "accuracy": float(accuracy_score(yte, pred)),
        "f1": float(f1_score(yte, pred, average="macro")),
        "train_time": train_time,
        "task": cfg["task"], "noise_p2": cfg["noise_p2"], "shots": cfg["shots"],
    }


def plan(stage: str):
    if stage == "debug":
        return [dict(task="4class_f4", encoding="angle", depth=1, n_train=25, seed=0,
                     noise_p2=p, shots=1024) for p in (0.0, 0.02)]
    # main: 4 qubits, 25 training points, 3 seeds, sweep noise for both encodings.
    # The noisy kernel needs ~N^2/2 circuit pairs, so keep n_train small.
    return [dict(task="4class_f4", encoding=enc, depth=1, n_train=25, seed=s,
                 noise_p2=p, shots=1024)
            for enc in ("angle", "zz") for p in (0.0, 0.005, 0.01, 0.02, 0.05) for s in (0, 1, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["debug", "main"], default="debug")
    ap.add_argument("--out", default=OUT_CSV)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        d = pd.read_csv(a.out)
        done = set(zip(d.task, d.encoding, d.depth, d.n_train, d.seed, d.noise_p2, d.shots))
    todo = [c for c in plan(a.stage)
            if (c["task"], c["encoding"], c["depth"], c["n_train"], c["seed"],
                c["noise_p2"], c["shots"]) not in done]
    print(f"{len(todo)} runs to do")
    for i, c in enumerate(todo, 1):
        r = run_one(c)
        pd.DataFrame([r]).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
        print(f"[{i}/{len(todo)}] {c['encoding']} p2={c['noise_p2']} seed={c['seed']} "
              f"acc={r['accuracy']:.3f} time={r['train_time']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
