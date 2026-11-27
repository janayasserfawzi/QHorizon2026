# Person 3 - VQC design rationale (feeds "Quantum model design")

## Circuit (PennyLane, 7 qubits on the main 4-class task)
1. **Encoding: angle encoding (RY per feature).** Person 1 scales features to [0, pi], one half-turn of RY, so nothing wraps around. One qubit per feature, shallow, same encoding as Person 2's angle kernel, so VQC vs kernel compares the *model*, not the encoding.
2. **Trainable block: StronglyEntanglingLayers.** Per layer: 3 rotations per qubit + a ring of CNOTs. Standard hardware-efficient ansatz; every qubit gets full single-qubit freedom plus entanglement, so no hand-tuned design to defend.
3. **Readout: basis-state probabilities of the first 2 qubits -> 4 classes** (rooms 1-4). One measurement setting gives all 4 class probabilities directly. (Binary task: 1 qubit.)
4. **Loss: cross-entropy** on those probabilities. Natural for multi-class, strong gradients when wrong.
5. **Init: uniform [0, 2pi], seeded.** Same seed -> same start, reproducible.

## Sweeps
- **Depth 1-4** (Adam, all 5 seeds x 4 training sizes). Depth 1 is the minimum; deeper adds expressivity (21 params/layer on 7 qubits) at the cost of runtime and trainability.
- **Optimizer at depth 2:** Adam (adaptive, robust), plain gradient descent (shows optimizer matters), COBYLA (gradient-free, what you'd use on noisy hardware). Same budget: 60 steps, full batch, lr 0.1 (COBYLA 180 evaluations).

## Results (4class, mean accuracy over 5 seeds x 4 train sizes)
| depth | Adam acc | train time |
|---|---|---|
| 1 | 0.38 | 1.1 s |
| 2 | 0.72 | 2.1 s |
| 3 | 0.84 | 3.1 s |
| 4 | 0.88 | 3.9 s |

Optimizers at depth 2: Adam 0.72, COBYLA 0.64, SGD 0.41. Takeaways: depth 1 is too weak (near chance, 25%); accuracy still rising at depth 4 with diminishing returns; Adam clearly beats plain SGD at the same steps; COBYLA is competitive without gradients.

## Fairness
Data only via `load_saved_split(seed, n_train, task)`; seeds 0-4; same test sets as every model. Output: `results/person3_vqc.csv` (shared columns + task, optimizer).
Run: `python VQC/runExperiments.py --stage all` (resumable).
