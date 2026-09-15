"""Self-test: build a tiny weighted 5-snapshot dataset with 3 planted communities
and train on it, to check the install and the code without the real data.

    python selftest.py
    python train.py --dataset synth --src synth --years 1990-1994 \
        --epochs 60 --batch_size 20 --patience 10 \
        --structural_head_config 4 --structural_layer_config 32 \
        --temporal_head_config 4 --temporal_layer_config 32

Expect: loss starts ~4.05 (= log 57, uniform), val perplexity drops to ~36,
early stop around epoch 30, beta ~1.03.

Files are written as `synth/adj_<year>.npz`, the same layout notebook 01 uses
for the real arenas, so train.py reads both with one code path.
"""

import os

import numpy as np
import scipy.sparse as sp

rng = np.random.RandomState(0)
N, T = 60, 5
os.makedirs("synth", exist_ok=True)
comm = np.repeat([0, 1, 2], 20)

for t in range(T):
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            same = comm[i] == comm[j]
            if rng.rand() < (0.45 + 0.05 * t if same else 0.04):
                A[i, j] = A[j, i] = rng.gamma(2.0, 0.5 if same else 0.2)
    if t < 2:                                   # a few isolated nodes in early years
        A[:3, :] = 0
        A[:, :3] = 0
    sp.save_npz("synth/adj_{}.npz".format(1990 + t), sp.csr_matrix(A))

print("wrote synth/adj_1990.npz .. adj_{}.npz".format(1990 + T - 1))
print("now run:  python train.py --dataset synth --src synth --years 1990-1994 \\")
print("              --epochs 60 --batch_size 20 --patience 10 \\")
print("              --structural_head_config 4 --structural_layer_config 32 \\")
print("              --temporal_head_config 4 --temporal_layer_config 32")
