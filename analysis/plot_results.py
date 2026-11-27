"""
plot_results.py -- Person 4

Reads results/person*.csv (+ results/noise_p4.csv if it exists), writes figures to
report/figures/ and tables to results/summary_headline.csv, summary_paired.csv.

    python analysis/plot_results.py

Nothing here picks a "best" configuration on test accuracy: every VQC depth/optimizer is shown.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)

TASK_QUBITS = {"4class_f4": 4, "4class_f5": 5, "4class_f6": 6, "4class": 7}
CLASSICAL = {"svm_rbf": "SVM (RBF)", "logreg": "LogReg", "mlp": "MLP"}
NS = [25, 50, 100, 200]


def load():
    frames = []
    files = sorted(glob.glob(os.path.join(RES, "person*.csv"))) + glob.glob(os.path.join(RES, "noise_p4.csv"))
    for f in files:
        d = pd.read_csv(f)
        frames.append(d)
        print(f"[ok] {os.path.basename(f)}: {len(d)} rows")
    df = pd.concat(frames, ignore_index=True)
    if "noise_p2" not in df:
        df["noise_p2"] = np.nan
    # the early debug rows labelled angle 'linear' are the same model as angle 'none'
    df.loc[df.model == "QKernel-angle", "entanglement"] = "none"
    df = df.drop_duplicates(["model", "encoding", "depth", "n_train", "seed", "task", "noise_p2"], keep="last")
    df["label"] = [name(r) for r in df.itertuples()]
    return df


def name(r):
    if r.model in CLASSICAL:
        return CLASSICAL[r.model]
    if r.model == "QKernel-angle":
        return "Q-kernel angle"
    if r.model == "QKernel-zz":
        return f"Q-kernel ZZ d{int(r.depth)}"
    if r.model.startswith("VQC-"):
        return f"VQC {r.model[4:]} d{int(r.depth)}"
    return r.model


def style(ax, title, xl, yl):
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(alpha=0.3)


def ms(g, x, y="accuracy"):
    return g.groupby(x)[y].agg(["mean", "std"]).reset_index()


MAIN = ["SVM (RBF)", "LogReg", "MLP", "Q-kernel angle", "Q-kernel ZZ d1", "VQC adam d6"]
STYLE = {"SVM (RBF)": ("tab:gray", "--"), "LogReg": ("tab:brown", "--"), "MLP": ("tab:olive", "--"),
         "Q-kernel angle": ("tab:blue", "-"), "Q-kernel ZZ d1": ("tab:orange", "-"),
         "VQC adam d6": ("tab:green", "-")}


def lines(ax, d, x, y="accuracy", labels=MAIN, logy=False):
    for lab in labels:
        g = d[d.label == lab]
        if g.empty:
            continue
        m = ms(g, x, y)
        c, ls = STYLE[lab]
        if logy:
            ax.plot(m[x], m["mean"], marker="o", ms=4, color=c, ls=ls, label=lab)
        else:
            ax.errorbar(m[x], m["mean"], yerr=m["std"].fillna(0), marker="o", ms=4, capsize=2,
                        color=c, ls=ls, label=lab)


def fig1(df):
    d = df[(df.task == "4class") & df.noise_p2.isna()]
    fig, ax = plt.subplots(figsize=(6, 4))
    lines(ax, d, "n_train")
    ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS)
    style(ax, "4 rooms, 7 features: accuracy vs training size", "training samples", "test accuracy (mean ± std, 5 seeds)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_learning_curve.png"), dpi=200)


def fig2(df, n=100):
    d = df[(df.n_train == n) & df.noise_p2.isna() & df.task.isin(TASK_QUBITS)].copy()
    d["q"] = d.task.map(TASK_QUBITS)
    fig, ax = plt.subplots(figsize=(6, 4))
    lines(ax, d, "q")
    ax.set_xticks([4, 5, 6, 7])
    style(ax, f"Accuracy vs qubits / features kept ({n} training samples)", "qubits (= features)", "test accuracy")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_qubits.png"), dpi=200)


def fig3(df):
    d = df[(df.task == "4class") & df.noise_p2.isna()]
    fig, ax = plt.subplots(figsize=(6, 4))
    lines(ax, d, "n_train", "train_time", logy=True)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS)
    style(ax, "Training time (7 qubits, simulator)", "training samples", "seconds (log)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_runtime.png"), dpi=200)


def fig4(df):
    """Every VQC depth, shown honestly. Reference lines = SVM and angle kernel."""
    d = df[(df.task == "4class") & df.model.str.startswith("VQC") & df.noise_p2.isna()]
    if d.empty:
        return
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, n in zip(axs, (25, 200)):
        a = d[(d.model == "VQC-adam") & (d.n_train == n)]
        m = ms(a, "depth")
        ax.errorbar(m.depth, m["mean"], yerr=m["std"], marker="o", capsize=2, color="tab:green", label="VQC (Adam)")
        for opt, mk in (("cobyla", "s"), ("sgd", "^")):
            b = d[(d.model == f"VQC-{opt}") & (d.n_train == n)]
            if len(b):
                ax.errorbar([2], [b.accuracy.mean()], yerr=[b.accuracy.std()], marker=mk, capsize=2,
                            ls="none", label=f"VQC ({opt.upper()}, depth 2)")
        for lab, c in (("SVM (RBF)", "tab:gray"), ("Q-kernel angle", "tab:blue")):
            v = df[(df.label == lab) & (df.task == "4class") & (df.n_train == n)].accuracy.mean()
            ax.axhline(v, color=c, ls="--", lw=1, label=lab)
        style(ax, f"{n} training samples", "VQC depth (layers)", "test accuracy")
    axs[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_vqc_depth.png"), dpi=200)


def fig5():
    d = os.path.join(RES, "kernel_diagnostics")
    items = [("gram_4class_f4_angle_d1.npy", "angle, 4 qubits"), ("gram_4class_f4_zz_d1.npy", "ZZ d1, 4 qubits"),
             ("gram_4class_angle_d1.npy", "angle, 7 qubits"), ("gram_4class_zz_d1.npy", "ZZ d1, 7 qubits")]
    items = [(f, t) for f, t in items if os.path.exists(os.path.join(d, f))]
    if not items:
        return
    fig, axs = plt.subplots(1, len(items), figsize=(3.2 * len(items), 3.2))
    for ax, (f, t) in zip(np.atleast_1d(axs), items):
        im = ax.imshow(np.load(os.path.join(d, f)), vmin=0, vmax=1, cmap="viridis")
        ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs, fraction=0.02)
    fig.suptitle("Kernel matrices, 100 training points sorted by room", fontsize=10)
    fig.savefig(os.path.join(FIG, "fig5_gram_matrices.png"), dpi=200, bbox_inches="tight")


def fig6(df):
    d = df[df.noise_p2.notna()]
    if d.empty:
        print("[noise] no rows yet, fig6 skipped")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for enc, g in d.groupby("encoding"):
        m = ms(g, "noise_p2")
        ax.errorbar(m.noise_p2, m["mean"], yerr=m["std"].fillna(0), marker="o", ms=4, capsize=2, label=enc)
    style(ax, "Quantum kernel under simulated noise (4 qubits, 25 train, 1024 shots)",
          "two-qubit gate / readout error rate", "test accuracy")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_noise.png"), dpi=200)


def fig7(df):
    d = df[df.model.str.startswith("QKernel") & df.k_offdiag_std.notna() & df.noise_p2.isna()
           & (df.task != "binary_1v2")]
    fig, ax = plt.subplots(figsize=(5, 4))
    for enc, g in d.groupby("encoding"):
        ax.scatter(g.k_offdiag_std, g.accuracy, s=14, alpha=0.6, label=enc)
    style(ax, "Kernel spread vs accuracy (each dot = one run)", "std of off-diagonal kernel values", "test accuracy")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_kernel_concentration.png"), dpi=200)


def tables(df):
    d = df[df.noise_p2.isna()]
    # headline: 4 rooms, 7 features, n=100 and n=200, every model/depth/optimizer
    h = (d[(d.task == "4class") & d.n_train.isin([100, 200])]
         .groupby(["label", "n_train"]).accuracy.agg(["mean", "std", "count"]).reset_index())
    h["acc"] = h["mean"].map("{:.3f}".format) + " ± " + h["std"].fillna(0).map("{:.3f}".format)
    tab = h.pivot(index="label", columns="n_train", values="acc")
    tab.to_csv(os.path.join(RES, "summary_headline.csv"))
    print("\nHEADLINE (4 rooms, 7 features; mean ± std over seeds):\n", tab.to_string())
    print("\nSeeds per row (should all be 5):", sorted(h["count"].unique()))

    # paired differences vs SVM (same seed = same split): honest effect size
    rows = []
    for task in ("4class", "4class_f4"):
        for n in NS:
            base = d[(d.label == "SVM (RBF)") & (d.task == task) & (d.n_train == n)].set_index("seed").accuracy
            for lab in sorted(d[(d.task == task) & (d.n_train == n)].label.unique()):
                if lab == "SVM (RBF)":
                    continue
                g = d[(d.label == lab) & (d.task == task) & (d.n_train == n)].set_index("seed").accuracy
                diff = (g - base).dropna()
                if len(diff) >= 3:
                    rows.append(dict(task=task, n_train=n, model=lab, diff_vs_svm=diff.mean(),
                                     diff_std=diff.std(), seeds_better=int((diff > 0).sum()), seeds=len(diff)))
    p = pd.DataFrame(rows).round(3)
    p.to_csv(os.path.join(RES, "summary_paired.csv"), index=False)
    print("\nPAIRED vs SVM, 4class, n=200:\n",
          p[(p.task == "4class") & (p.n_train == 200)].to_string(index=False))


if __name__ == "__main__":
    df = load()
    fig1(df); fig2(df); fig3(df); fig4(df); fig5(); fig6(df); fig7(df)
    tables(df)
    print("\nFigures ->", FIG)
