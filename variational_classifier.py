"""
variational_classifier.py — main experiment for Project 02
===========================================================

Trains three variational quantum classifiers on a non-linearly separable
dataset, compares them against classical baselines, and produces four figures:

  fig1_circuit_diagram.png     — the variational classifier circuit
  fig2_training_curves.png     — loss & accuracy vs epoch for each ansatz
  fig3_decision_boundaries.png — learned decision regions + classical baseline
  fig4_paramshift_barren.png   — parameter-shift verification + barren plateau

Run:  python variational_classifier.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from ansatz import ANSATZE
from qnn import make_model, train, accuracy, predict_raw
from analysis import verify_parameter_shift, barren_plateau_scan

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
np.random.seed(SEED)

N_QUBITS = 2
N_LAYERS = 3
EPOCHS = 45

# ---------------------------------------------------------------------------
# Data: two moons — a classic non-linearly-separable binary problem.
# Labels are mapped to {-1, +1} to match the Pauli-Z expectation range.
# ---------------------------------------------------------------------------
X, y01 = make_moons(n_samples=200, noise=0.20, random_state=SEED)
X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)
y = 2 * y01 - 1  # {0,1} -> {-1,+1}

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=SEED, stratify=y)
X_tr, X_te = pnp.array(X_tr), pnp.array(X_te)
y_tr, y_te = pnp.array(y_tr), pnp.array(y_te)


def main():
    # -----------------------------------------------------------------------
    # 1. Train each ansatz
    # -----------------------------------------------------------------------
    results = {}
    for name in ANSATZE:
        print(f"\nTraining {name} ...")
        circuit, weights, _ = make_model(name, N_QUBITS, N_LAYERS, diff_method="backprop")
        bias = pnp.array(0.0, requires_grad=True)
        weights, bias, hist = train(circuit, weights, bias, X_tr, y_tr, X_te, y_te,
                                    epochs=EPOCHS, lr=0.1, batch_size=25,
                                    seed=SEED, verbose=True)
        n_params = int(np.prod(ANSATZE[name]["shape"](N_LAYERS, N_QUBITS))) + 1
        results[name] = {
            "circuit": circuit, "weights": weights, "bias": bias,
            "hist": hist, "n_params": n_params,
            "test_acc": accuracy(circuit, weights, bias, X_te, y_te),
            "train_acc": accuracy(circuit, weights, bias, X_tr, y_tr),
        }

    # Classical baselines
    lin = accuracy_score(y_te, SVC(kernel="linear").fit(X_tr, y_tr).predict(X_te))
    rbf = accuracy_score(y_te, SVC(kernel="rbf").fit(X_tr, y_tr).predict(X_te))

    print("\n================ SUMMARY ================")
    for name, r in results.items():
        print(f"{name:20s} params={r['n_params']:3d}  "
              f"train={r['train_acc']:.3f}  test={r['test_acc']:.3f}")
    print(f"{'Classical Linear':20s} test={lin:.3f}")
    print(f"{'Classical RBF':20s} test={rbf:.3f}")

    # -----------------------------------------------------------------------
    # Figure 1: circuit diagram (Strongly Entangler, 1 layer for clarity)
    # -----------------------------------------------------------------------
    circ1, w1, _ = make_model("StronglyEntangler", N_QUBITS, 1)
    fig1, ax1 = qml.draw_mpl(circ1, decimals=2, style="pennylane")(
        pnp.array([0.8, 1.3]), w1)
    fig1.suptitle("Figure 1 — Variational classifier circuit (Strongly Entangler, 1 layer)")
    fig1.savefig(os.path.join(HERE, "fig1_circuit_diagram.png"), dpi=120, bbox_inches="tight")

    # -----------------------------------------------------------------------
    # Figure 2: training curves
    # -----------------------------------------------------------------------
    fig2, (axl, axa) = plt.subplots(1, 2, figsize=(13, 5))
    for name, r in results.items():
        c = ANSATZE[name]["color"]
        axl.plot(r["hist"]["loss"], color=c, label=name)
        axa.plot(r["hist"]["test_acc"], color=c, label=f"{name} (test)")
        axa.plot(r["hist"]["train_acc"], color=c, ls="--", alpha=0.5)
    axl.set_title("Training loss vs epoch"); axl.set_xlabel("epoch"); axl.set_ylabel("square loss")
    axl.legend(); axl.grid(alpha=0.3)
    axa.axhline(lin, color="gray", ls=":", label=f"linear SVM ({lin:.2f})")
    axa.set_title("Accuracy vs epoch (dashed = train)"); axa.set_xlabel("epoch")
    axa.set_ylabel("accuracy"); axa.legend(); axa.grid(alpha=0.3)
    fig2.suptitle("Figure 2 — Training dynamics across ansatze")
    fig2.tight_layout()
    fig2.savefig(os.path.join(HERE, "fig2_training_curves.png"), dpi=120)

    # -----------------------------------------------------------------------
    # Figure 3: decision boundaries
    # -----------------------------------------------------------------------
    GRID = 60
    xx, yy = np.meshgrid(np.linspace(0, np.pi, GRID), np.linspace(0, np.pi, GRID))
    grid = pnp.array(np.c_[xx.ravel(), yy.ravel()])

    fig3, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, name in zip(axes[:3], ANSATZE):
        r = results[name]
        Z = pnp.sign(predict_raw(r["circuit"], r["weights"], r["bias"], grid))
        Z = np.array(Z).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        ax.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="coolwarm", edgecolors="k", s=30)
        ax.set_title(f"{name}\ntest acc {r['test_acc']:.2f}")
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
    # classical linear baseline panel
    lin_clf = SVC(kernel="linear").fit(X_tr, y_tr)
    Zl = lin_clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[3].contourf(xx, yy, Zl, alpha=0.3, cmap="coolwarm")
    axes[3].scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="coolwarm", edgecolors="k", s=30)
    axes[3].set_title(f"Classical Linear SVM\ntest acc {lin:.2f}")
    axes[3].set_xlabel("x1"); axes[3].set_ylabel("x2")
    fig3.suptitle("Figure 3 — Learned decision boundaries")
    fig3.tight_layout()
    fig3.savefig(os.path.join(HERE, "fig3_decision_boundaries.png"), dpi=120)

    # -----------------------------------------------------------------------
    # Figure 4: parameter-shift verification + barren plateau
    # -----------------------------------------------------------------------
    print("\nVerifying parameter-shift rule ...")
    grads = verify_parameter_shift()
    print("Running barren-plateau scan ...")
    qubits, variances = barren_plateau_scan()

    fig4, (axg, axb) = plt.subplots(1, 2, figsize=(13, 5))
    idx = np.arange(len(grads["autodiff"]))
    width = 0.27
    axg.bar(idx - width, grads["param_shift"], width, label="parameter-shift", color="#d62728")
    axg.bar(idx, grads["autodiff"], width, label="autodiff", color="#1f77b4")
    axg.bar(idx + width, grads["finite_diff"], width, label="finite-diff", color="#2ca02c")
    axg.set_title("Parameter-shift rule reproduces the exact gradient")
    axg.set_xlabel("parameter index"); axg.set_ylabel("gradient value")
    axg.legend(); axg.grid(alpha=0.3)

    axb.semilogy(qubits, variances, "o-", color="#9467bd")
    axb.set_title("Barren plateau: gradient variance decays with #qubits")
    axb.set_xlabel("number of qubits"); axb.set_ylabel("Var[$\\partial_\\theta \\langle Z Z\\rangle$] (log)")
    axb.grid(alpha=0.3, which="both")
    fig4.suptitle("Figure 4 — Parameter-shift verification and barren-plateau scaling")
    fig4.tight_layout()
    fig4.savefig(os.path.join(HERE, "fig4_paramshift_barren.png"), dpi=120)

    max_abs_err = float(np.max(np.abs(grads["param_shift"] - grads["autodiff"])))
    print(f"\nMax |param-shift - autodiff| gradient error: {max_abs_err:.2e}")
    print(f"Barren-plateau variances: {[f'{v:.2e}' for v in variances]}")
    print(f"\nSaved 4 figures to {HERE}")

    # Persist a compact results summary for the README table
    with open(os.path.join(HERE, "results_summary.txt"), "w") as f:
        for name, r in results.items():
            f.write(f"{name}\tparams={r['n_params']}\ttrain={r['train_acc']:.3f}\ttest={r['test_acc']:.3f}\n")
        f.write(f"Linear\ttest={lin:.3f}\n")
        f.write(f"RBF\ttest={rbf:.3f}\n")
        f.write(f"param_shift_max_err={max_abs_err:.2e}\n")
        f.write(f"barren_qubits={qubits}\n")
        f.write(f"barren_variances={[float(v) for v in variances]}\n")


if __name__ == "__main__":
    main()
