import glob
import pandas as pd

KEY = ["model", "encoding", "n_qubits", "depth", "n_train", "seed", "task"]
frames = []
for f in sorted(glob.glob("results/person*.csv")):
    d = pd.read_csv(f)
    d["source"] = f
    frames.append(d)
    print(f"{f}: {len(d)} rows, columns = {list(d.columns)}")
df = pd.concat(frames, ignore_index=True)

problems = 0
def warn(msg):
    global problems
    problems += 1
    print("PROBLEM:", msg)

# 1. required columns present
for c in ["model", "n_train", "seed", "accuracy", "f1", "task"]:
    if c not in df.columns:
        warn(f"column '{c}' missing")

# 2. values valid
for c in ["accuracy", "f1"]:
    if df[c].isna().any():
        warn(f"{c} has NaN in {df[df[c].isna()].source.unique()}")
    if ((df[c] < 0) | (df[c] > 1)).any():
        warn(f"{c} outside [0,1]")

# 3. duplicates (include optional columns if present)
key = KEY + [c for c in ["entanglement", "optimizer", "noise"] if c in df.columns]
dups = df[df.duplicated(key, keep=False)]
if len(dups):
    warn(f"{len(dups)} duplicate rows, e.g.\n{dups.head(6)[key + ['source']]}")

# 4. coverage: every config should have all 5 seeds
cfg = [c for c in key if c != "seed"]
cover = df.groupby(cfg, dropna=False)["seed"].nunique().reset_index(name="n_seeds")
short = cover[cover.n_seeds < 5]
if len(short):
    warn(f"{len(short)} configurations have fewer than 5 seeds:\n{short.to_string()}")

# 5. fairness: every task should have the same n_train values for every model family
for task, g in df.groupby("task"):
    sizes = g.groupby("model")["n_train"].apply(lambda s: sorted(s.unique()))
    if sizes.apply(tuple).nunique() > 1:
        warn(f"task {task}: models use different training sizes:\n{sizes.to_string()}")

print("\n" + ("All checks passed." if problems == 0 else f"{problems} problem(s) found."))

# report table: mean ± std over seeds
cfg_cols = [c for c in ["task", "model", "encoding", "n_qubits", "depth",
                        "entanglement", "optimizer", "n_train"] if c in df.columns]
agg = (df.groupby(cfg_cols, dropna=False)[["accuracy", "f1"]]
         .agg(["mean", "std"]).round(3))
agg.to_csv("results/summary_mean_std.csv")
df.to_csv("results/all_results.csv", index=False)
print(agg.to_string())