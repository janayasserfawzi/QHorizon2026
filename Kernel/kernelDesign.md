# Quantum Kernel Track 

Implementation of a **Quantum Kernel + Classical SVM** classifier for the UCI Wireless Indoor Localization dataset (`4 rooms`, `7 Wi-Fi RSSI features` scaled to $[0, \pi]$).

## Design & Architecture
* **Classifier (`qkernel_model.py`):** Exposes a scikit-learn-compatible `QuantumKernelSVM(encoding, depth, entanglement, C, fidelity)` interface using `SVC(kernel="precomputed")`.
* **Encodings Evaluated:**
  * **Angle Encoding (`z_feature_map`, `depth=1`):** Product-state $R_Z$ rotations with zero entanglement; serves as the quantum baseline.
  * **ZZ Feature Map (`zz_feature_map`, `depth=1, 2`, `entanglement="linear"`):** Adds two-qubit $ZZ$ entangling gates encoding pairwise router interactions $(\pi - x_i)(\pi - x_j)$.
* **Simulation Optimization:** Uses exact batched `Statevector` inner products $|\langle \psi(x_i) | \psi(x_j) \rangle|^2$ in $\mathcal{O}(N)$ state evaluations (mathematically identical to `FidelityQuantumKernel` to machine precision), while falling back to Qiskit's `FidelityQuantumKernel` when a noisy `fidelity` primitive is passed by Person 4.
* **Kernel Concentration Metric:** Logs off-diagonal Gram matrix mean and standard deviation (`k_offdiag_mean`, `k_offdiag_std`) to diagnose exponential kernel concentration as qubit count and circuit depth increase.

## Experiment Grid (`run_experiments.py`)
All runs pull identical frozen splits via `data_prep.load_saved_split(seed, n_train, task)` across `SEEDS = [0..4]` and `N_TRAIN = [25, 50, 100, 200]`:
1. **Debug (`--stage debug`):** Binary sanity check (`rooms 1 vs 2`, 7 qubits).
2. **Main 7-Qubit Benchmark (`--stage main`):** `angle` (`d=1`) vs. `zz` (`d=1, 2`) on `4class`.
3. **Qubit Scaling (`--stage qubits`):** `4, 5, 6, 7` qubits (`4class_f4` through `4class`) using top-$k$ ANOVA F-score features.

## Usage
```bash
# Run full experiment suite (saves incrementally to results/person2_kernel.csv)
python Kernel/run_experiments.py --stage all

# Export Gram matrices (.npy) and 5-seed summary table for report plots
python Kernel/export_diagnostics.py