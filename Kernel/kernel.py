
import numpy as np
from sklearn.svm import SVC
from qiskit.circuit.library import z_feature_map, zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel


class QuantumKernelSVM:
    """
    Contract-compliant Quantum Kernel + Classical SVM classifier.
    Supports:
      - 'angle': Product-state ZFeatureMap (no entanglement)
      - 'zz':    Entangling ZZFeatureMap (captures pairwise Wi-Fi router correlations)
    """

    def __init__(
        self,
        encoding: str = "angle",
        depth: int = 1,
        entanglement: str = "linear",
        C: float = 1.0,
        fidelity=None,
    ):
        self.encoding = encoding.lower()
        self.depth = int(depth)
        self.entanglement = entanglement
        self.C = float(C)
        self.fidelity = fidelity

        self.n_qubits_ = None
        self.feature_map_ = None
        self.qkernel_ = None
        self.svc_ = SVC(kernel="precomputed", C=self.C)
        self.X_train_ = None
        self.last_K_train_ = None

    def _build_kernel(self, n_qubits: int) -> None:
        if self.encoding == "angle":
            # Pure product-state rotation encoding (baseline quantum kernel)
            fmap = z_feature_map(feature_dimension=n_qubits, reps=self.depth)
        elif self.encoding == "zz":
            # Second-order Pauli-Z entangling feature map
            fmap = zz_feature_map(
                feature_dimension=n_qubits,
                reps=self.depth,
                entanglement=self.entanglement,
            )
        else:
            raise ValueError(f"Unsupported encoding '{self.encoding}'. Choose 'angle' or 'zz'.")

        self.n_qubits_ = n_qubits
        self.feature_map_ = fmap
        self.qkernel_ = FidelityQuantumKernel(
            feature_map=fmap,
            fidelity=self.fidelity,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train_ = np.asarray(X_train, dtype=float)
        self._build_kernel(n_qubits=self.X_train_.shape[1])

        # Compute symmetric N_train x N_train Gram matrix
        K_train = self.qkernel_.evaluate(x_vec=self.X_train_)
        self.last_K_train_ = K_train
        self.svc_.fit(K_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.qkernel_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_te = np.asarray(X_test, dtype=float)

        # Compute N_test x N_train kernel matrix
        K_test = self.qkernel_.evaluate(x_vec=X_te, y_vec=self.X_train_)
        return self.svc_.predict(K_test)

    def kernel_concentration_stats(self) -> dict:
        """
        Measures exponential kernel concentration (off-diagonal mean & std).
        When std -> 0, K_train approaches the Identity matrix and generalization drops.
        """
        if self.last_K_train_ is None:
            return {"k_offdiag_mean": np.nan, "k_offdiag_std": np.nan}
        K = self.last_K_train_
        mask = ~np.eye(K.shape[0], dtype=bool)
        off_diag = K[mask]
        return {
            "k_offdiag_mean": float(np.mean(off_diag)),
            "k_offdiag_std": float(np.std(off_diag)),
        }