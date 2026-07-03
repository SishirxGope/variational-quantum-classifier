"""
analysis.py — Parameter-shift verification and barren-plateau study
====================================================================

Two QNN-specific investigations that go beyond plain classification:

1. verify_parameter_shift():
   Confirm the parameter-shift rule reproduces the exact gradient by comparing
   it against (a) PennyLane's autodiff and (b) a central finite difference.

2. barren_plateau_scan():
   Measure the variance of a circuit gradient as the qubit count grows. In
   deep hardware-efficient circuits the gradient variance decays exponentially
   with the number of qubits — the "barren plateau" that makes large QNNs
   untrainable. We reproduce this decay empirically.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ansatz import ANSATZE


# ---------------------------------------------------------------------------
# 1. Parameter-shift rule verification
# ---------------------------------------------------------------------------
def verify_parameter_shift(name="StronglyEntangler", n_qubits=2, n_layers=2):
    """Return dict of gradients from param-shift, autodiff, finite-diff."""
    spec = ANSATZE[name]
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(weights, x):
        spec["build"](x, weights, wires)
        return qml.expval(qml.PauliZ(0))

    shape = spec["shape"](n_layers, n_qubits)
    rng = pnp.random.default_rng(7)
    weights = pnp.array(rng.uniform(0, 2 * np.pi, size=shape), requires_grad=True)
    x = pnp.array([0.8, 1.3], requires_grad=False)

    # (a) PennyLane autodiff gradient (uses the parameter-shift rule internally)
    autodiff = qml.grad(circuit, argnums=0)(weights, x).flatten()

    # (b) Manual parameter-shift: shift each angle by +-pi/2
    flat = weights.flatten()
    manual = np.zeros_like(np.asarray(flat))
    s = np.pi / 2
    for k in range(len(flat)):
        plus = np.array(flat, dtype=float); plus[k] += s
        minus = np.array(flat, dtype=float); minus[k] -= s
        fp = circuit(pnp.array(plus.reshape(shape)), x)
        fm = circuit(pnp.array(minus.reshape(shape)), x)
        manual[k] = (fp - fm) / 2.0

    # (c) Central finite difference (numerical reference)
    eps = 1e-4
    finite = np.zeros_like(np.asarray(flat))
    for k in range(len(flat)):
        plus = np.array(flat, dtype=float); plus[k] += eps
        minus = np.array(flat, dtype=float); minus[k] -= eps
        fp = circuit(pnp.array(plus.reshape(shape)), x)
        fm = circuit(pnp.array(minus.reshape(shape)), x)
        finite[k] = (fp - fm) / (2 * eps)

    return {
        "autodiff": np.asarray(autodiff, dtype=float),
        "param_shift": manual,
        "finite_diff": finite,
    }


# ---------------------------------------------------------------------------
# 2. Barren-plateau gradient-variance scan
# ---------------------------------------------------------------------------
def barren_plateau_scan(qubit_range=(2, 3, 4, 5, 6), n_layers=4, n_samples=60):
    """Variance of d<Z>/d(theta_0) over random parameter inits vs #qubits."""
    variances = []
    for n_qubits in qubit_range:
        wires = list(range(n_qubits))
        dev = qml.device("default.qubit", wires=n_qubits)

        def build(weights):
            # hardware-efficient random circuit (RX-RY-RZ + CZ ring), no data
            for layer in range(n_layers):
                for i, w in enumerate(wires):
                    qml.RX(weights[layer, i, 0], wires=w)
                    qml.RY(weights[layer, i, 1], wires=w)
                    qml.RZ(weights[layer, i, 2], wires=w)
                for i in range(n_qubits):
                    qml.CZ(wires=[wires[i], wires[(i + 1) % n_qubits]])

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(weights):
            build(weights)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        rng = pnp.random.default_rng(2024 + n_qubits)
        grads = []
        for _ in range(n_samples):
            w = pnp.array(rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 3)),
                          requires_grad=True)
            g = qml.grad(circuit)(w)
            grads.append(float(np.asarray(g).flatten()[0]))
        variances.append(np.var(grads))

    return list(qubit_range), variances
