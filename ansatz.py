"""
ansatz.py — Variational circuit architectures for the QNN classifier
=====================================================================

A variational quantum classifier is built from two parts:

  1. A *data-encoding* block  S(x)  that loads a classical vector x into the
     qubits (here: angle embedding — one RY rotation per feature, repeated so
     the data is "re-uploaded" between trainable layers).
  2. A *trainable* block  W(theta)  of parameterized rotations + entangling
     gates whose angles theta are optimized against a classification loss.

We compare three trainable blocks of increasing structure:

  - BasicEntangler    : one RX per qubit + a ring of CNOTs   (1 param / qubit / layer)
  - StronglyEntangler : RZ-RY-RZ per qubit + a ring of CNOTs (3 params / qubit / layer)
  - HardwareEfficient : RX-RY-RZ per qubit + a ring of CZs    (3 params / qubit / layer)

Each ansatz exposes:
  - build(x, weights, wires)   : append encoding + trainable gates to a QNode
  - shape(n_layers, n_qubits)  : the trainable-weight tensor shape
"""

import numpy as np
import pennylane as qml


# ---------------------------------------------------------------------------
# Data encoding (shared by all ansatze) — angle embedding with re-uploading
# ---------------------------------------------------------------------------
def _encode(x, wires):
    """Load a 2D point into the qubits via RY rotations (angle embedding)."""
    for i, w in enumerate(wires):
        qml.RY(x[i % len(x)], wires=w)


# ---------------------------------------------------------------------------
# 1. Basic entangler: RX per qubit + CNOT ring
# ---------------------------------------------------------------------------
def basic_build(x, weights, wires):
    n = len(wires)
    for layer in range(weights.shape[0]):
        _encode(x, wires)                       # re-upload data each layer
        for i, w in enumerate(wires):
            qml.RX(weights[layer, i, 0], wires=w)
        for i in range(n):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


def basic_shape(n_layers, n_qubits):
    return (n_layers, n_qubits, 1)


# ---------------------------------------------------------------------------
# 2. Strongly entangling: RZ-RY-RZ per qubit + CNOT ring
# ---------------------------------------------------------------------------
def strong_build(x, weights, wires):
    n = len(wires)
    for layer in range(weights.shape[0]):
        _encode(x, wires)
        for i, w in enumerate(wires):
            qml.RZ(weights[layer, i, 0], wires=w)
            qml.RY(weights[layer, i, 1], wires=w)
            qml.RZ(weights[layer, i, 2], wires=w)
        for i in range(n):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


def strong_shape(n_layers, n_qubits):
    return (n_layers, n_qubits, 3)


# ---------------------------------------------------------------------------
# 3. Hardware-efficient: RX-RY-RZ per qubit + CZ ring
# ---------------------------------------------------------------------------
def hweff_build(x, weights, wires):
    n = len(wires)
    for layer in range(weights.shape[0]):
        _encode(x, wires)
        for i, w in enumerate(wires):
            qml.RX(weights[layer, i, 0], wires=w)
            qml.RY(weights[layer, i, 1], wires=w)
            qml.RZ(weights[layer, i, 2], wires=w)
        for i in range(n):
            qml.CZ(wires=[wires[i], wires[(i + 1) % n]])


def hweff_shape(n_layers, n_qubits):
    return (n_layers, n_qubits, 3)


# ---------------------------------------------------------------------------
# Registry for easy iteration
# ---------------------------------------------------------------------------
ANSATZE = {
    "BasicEntangler": {
        "build": basic_build, "shape": basic_shape,
        "label": "Basic Entangler (RX + CNOT ring)", "color": "#1f77b4",
    },
    "StronglyEntangler": {
        "build": strong_build, "shape": strong_shape,
        "label": "Strongly Entangler (RZRYRZ + CNOT ring)", "color": "#d62728",
    },
    "HardwareEfficient": {
        "build": hweff_build, "shape": hweff_shape,
        "label": "Hardware-Efficient (RXRYRZ + CZ ring)", "color": "#2ca02c",
    },
}
