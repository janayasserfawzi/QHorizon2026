# Kernel/qkernel_model.py
import numpy as np
from sklearn.svm import SVC
from qiskit.circuit.library import z_feature_map, zz_feature_map
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.kernels import FidelityQuantumKernel


class QuantumKernelSVM:
    """
    Fast Statevector-batched Quantum Kernel + Classical SVM classifier.
    Computes exact kernel entries |<psi(x)|psi(x')>|^2 in O(N) statevector
    evaluations instead of O(N^2) circuit simulations when fidelity=None.
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
        self.train_states_ = None
        self.last_K_train_ = None

    def _build_kernel(self, n_qubits: int) -> None:
        if self.encoding == "angle":
            fmap = z_feature_map(feature_dimension=n_qubits, reps=self.depth)
        elif self.encoding == "zz":
            fmap = zz_feature_map(
                feature_dimension=n_qubits,
                reps=self.depth,
                entanglement=self.entanglement,
            )
        else:
            raise ValueError(f"Unsupported encoding '{self.encoding}'. Choose 'angle' or 'zz'.")

        self.n_qubits_ = n_qubits
        self.feature_map_ = fmap

        if self.fidelity is not None:
            # Fallback for Person 4's noisy Sampler fidelity experiments
            self.qkernel_ = FidelityQuantumKernel(
                feature_map=fmap,
                fidelity=self.fidelity,
            )

    def _encode_states(self, X: np.ndarray) -> np.ndarray:
        """Bind each row of X into the feature map and return an (N, 2^n_qubits) matrix."""
        params = list(self.feature_map_.parameters)
        states = np.empty((len(X), 2 ** self.n_qubits_), dtype=np.complex128)
        for i, row in enumerate(X):
            bound = self.feature_map_.assign_parameters(dict(zip(params, row)))
            states[i] = Statevector.from_instruction(bound).data
        return states

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train_ = np.asarray(X_train, dtype=float)
        self._build_kernel(n_qubits=self.X_train_.shape[1])

        if self.fidelity is None:
            # Fast exact path: N statevectors + 1 matrix multiply
            self.train_states_ = self._encode_states(self.X_train_)
            K_train = np.abs(self.train_states_ @ self.train_states_.conj().T) ** 2
        else:
            K_train = self.qkernel_.evaluate(x_vec=self.X_train_)

        self.last_K_train_ = K_train
        self.svc_.fit(K_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.X_train_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_te = np.asarray(X_test, dtype=float)

        if self.fidelity is None:
            test_states = self._encode_states(X_te)
            K_test = np.abs(test_states @ self.train_states_.conj().T) ** 2
        else:
            K_test = self.qkernel_.evaluate(x_vec=X_te, y_vec=self.X_train_)

        return self.svc_.predict(K_test)

    def kernel_concentration_stats(self) -> dict:
        if self.last_K_train_ is None:
            return {"k_offdiag_mean": np.nan, "k_offdiag_std": np.nan}
        K = self.last_K_train_
        mask = ~np.eye(K.shape[0], dtype=bool)
        off_diag = K[mask]
        return {
            "k_offdiag_mean": float(np.mean(off_diag)),
            "k_offdiag_std": float(np.std(off_diag)),
        }