"""Person 3 - Variational quantum classifier (PennyLane).
Shared interface: model.fit(X_train, y_train); model.predict(X_test).
X already scaled to [0, pi] by Person 1; one qubit per feature (angle encoding).
Works for binary and multi-class: readout = computational-basis probabilities
of the first ceil(log2 K) qubits, first K outcomes kept and renormalised.
"""
import math
import time
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize


class VQC:
    def __init__(self, n_qubits=7, depth=2, optimizer="adam", steps=60,
                 lr=0.1, seed=0, n_classes=4):
        self.n_qubits, self.depth = n_qubits, depth
        self.optimizer, self.steps, self.lr = optimizer, steps, lr
        self.seed, self.n_classes = seed, n_classes
        self.n_read = max(1, math.ceil(math.log2(n_classes)))
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.train_time, self.loss_history = 0.0, []
        n_read = self.n_read

        @qml.qnode(self.dev, interface="autograd", diff_method="backprop")
        def circuit(x, w):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")   # encoding
            qml.StronglyEntanglingLayers(w, wires=range(n_qubits))       # trainable
            return qml.probs(wires=range(n_read))                        # readout
        self._circuit = circuit

    def _check(self, X):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_qubits:
            raise ValueError(f"expected {self.n_qubits} features, got {X.shape[1]}")
        return X

    def _class_probs(self, X, w):
        p = self._circuit(X, w)[:, :self.n_classes]
        return p / pnp.sum(p, axis=1, keepdims=True)

    def _loss(self, w, X, y_onehot):
        p = self._class_probs(X, w)
        return -pnp.mean(pnp.sum(y_onehot * pnp.log(p + 1e-9), axis=1))  # cross-entropy

    def fit(self, X_train, y_train):
        X = self._check(X_train)
        y = np.asarray(y_train, dtype=int)
        Y = np.eye(self.n_classes)[y]
        rng = np.random.default_rng(self.seed)
        shape = qml.StronglyEntanglingLayers.shape(self.depth, self.n_qubits)
        w0 = rng.uniform(0, 2 * np.pi, size=shape)
        self.loss_history = []
        t0 = time.perf_counter()
        if self.optimizer in ("adam", "sgd"):
            opt = (qml.AdamOptimizer(self.lr) if self.optimizer == "adam"
                   else qml.GradientDescentOptimizer(self.lr))
            w = pnp.array(w0, requires_grad=True)
            for _ in range(self.steps):
                w, loss = opt.step_and_cost(lambda v: self._loss(v, X, Y), w)
                self.loss_history.append(float(loss))
        elif self.optimizer == "cobyla":
            def f(flat):
                l = float(self._loss(pnp.array(flat.reshape(shape)), X, Y))
                self.loss_history.append(l)
                return l
            res = minimize(f, w0.ravel(), method="COBYLA",
                           options={"maxiter": self.steps * 3})
            w = pnp.array(res.x.reshape(shape))
        else:
            raise ValueError(f"unknown optimizer {self.optimizer}")
        self.weights = w
        self.train_time = time.perf_counter() - t0
        return self

    def predict(self, X_test):
        X = self._check(X_test)
        return np.argmax(np.asarray(self._class_probs(X, self.weights)), axis=1)
