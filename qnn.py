"""
qnn.py — Variational Quantum Classifier: model, loss, and training
===================================================================

The classifier maps a 2D point x -> a real number in [-1, 1] via the
expectation value of a Pauli-Z observable on the first qubit:

        f(x; theta, b) = <0| S(x)^dag W(theta)^dag  Z_0  W(theta) S(x) |0> + b

We train theta (and a classical bias b) to minimize a square-loss between
f(x) and the label y in {-1, +1}. Gradients are obtained by the
*parameter-shift rule*, the quantum analogue of backpropagation:

        d<Z>/d(theta_k) = [ f(theta_k + pi/2) - f(theta_k - pi/2) ] / 2

PennyLane computes these gradients automatically (diff_method), but this
module also exposes a hand-written parameter-shift gradient so the rule can be
verified explicitly (see analysis.py).
"""

import pennylane as qml
from pennylane import numpy as pnp

from ansatz import ANSATZE


def make_model(name, n_qubits=2, n_layers=3, diff_method="backprop"):
    """Return (qnode, init_weights, wires) for a named ansatz.

    diff_method="backprop"        -> fast simulator training
    diff_method="parameter-shift" -> hardware-realistic gradients
    """
    spec = ANSATZE[name]
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(x, weights):
        spec["build"](x, weights, wires)
        return qml.expval(qml.PauliZ(0))

    shape = spec["shape"](n_layers, n_qubits)
    rng = pnp.random.default_rng(seed=1234)
    init_weights = pnp.array(rng.uniform(0, 2 * pnp.pi, size=shape), requires_grad=True)
    return circuit, init_weights, wires


def predict_raw(circuit, weights, bias, X):
    """Model output f(x) in (roughly) [-1, 1] for each row of X."""
    return pnp.stack([circuit(x, weights) for x in X]) + bias


def square_loss(circuit, weights, bias, X, y):
    preds = predict_raw(circuit, weights, bias, X)
    return pnp.mean((preds - y) ** 2)


def accuracy(circuit, weights, bias, X, y):
    preds = pnp.sign(predict_raw(circuit, weights, bias, X))
    return float(pnp.mean(preds == y))


def train(circuit, weights, bias, X_train, y_train, X_test, y_test,
          epochs=40, lr=0.1, batch_size=20, seed=0, verbose=False):
    """Adam training loop. Returns trained params + per-epoch history."""
    opt = qml.AdamOptimizer(stepsize=lr)
    rng = pnp.random.default_rng(seed)
    n = len(X_train)

    history = {"loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        idx = rng.permutation(n)[:batch_size]
        Xb = pnp.array(X_train[idx], requires_grad=False)
        yb = pnp.array(y_train[idx], requires_grad=False)

        # Only weights and bias are trainable; the batch is captured by closure.
        def cost(w, b):
            return square_loss(circuit, w, b, Xb, yb)

        weights, bias = opt.step(cost, weights, bias)

        history["loss"].append(float(square_loss(circuit, weights, bias, X_train, y_train)))
        history["train_acc"].append(accuracy(circuit, weights, bias, X_train, y_train))
        history["test_acc"].append(accuracy(circuit, weights, bias, X_test, y_test))
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d} | loss {history['loss'][-1]:.4f} "
                  f"| train {history['train_acc'][-1]:.3f} "
                  f"| test {history['test_acc'][-1]:.3f}")

    return weights, bias, history
