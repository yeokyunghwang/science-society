"""Self-test: build a synthetic weighted multi-snapshot graph with planted communities
and write it as `synth/adj_<year>.npz`, the layout notebook 01 uses for the real
arenas, so train.py reads both with one code path. Checks the install and both
attention variants without the real data.

    python selftest.py                       # 60 nodes, 5 snapshots: the reference case
    python selftest.py --nodes 3000 --degree 50   # closer to the real graphs' density

then, for the reference case,

    python train.py --dataset synth --src synth --years 1990-1994 \
        --epochs 60 --batch_size 20 --patience 10 \
        --structural_head_config 4 --structural_layer_config 32 \
        --temporal_head_config 4 --temporal_layer_config 32

Reference numbers (60 nodes, defaults): loss starts ~4.05 (= log 57, uniform);
gat   early stop at epoch 44, best epoch 34, val 3.6122
gatv2 early stop at epoch 44, best epoch 34, val 3.6032 (--share_weights=false)
Other sizes give other numbers; the check there is that both variants train and
val loss falls well below log(n_active).
"""

import argparse
import os

import numpy as np
import scipy.sparse as sp


def build(nodes, snapshots, communities, degree, seed, first_year=1990, out="synth"):
    rng = np.random.RandomState(seed)
    comm = np.arange(nodes) * communities // nodes            # equal-size communities
    same = comm[:, None] == comm[None, :]

    # Edge probabilities. The defaults (0.45 same / 0.04 cross) reproduce the reference
    # case; --degree rescales both, keeping their ratio, to hit a target mean degree.
    p_same, p_cross = 0.45, 0.04
    if degree is not None:
        c = nodes / communities
        p_cross = degree / ((c - 1) * (p_same / p_cross) + (nodes - c))
        p_same = min(0.95, p_cross * 0.45 / 0.04)

    n_isolated = max(3, nodes // 20)                          # isolated in early years
    os.makedirs(out, exist_ok=True)
    for t in range(snapshots):
        p = np.where(same, min(0.95, p_same + 0.05 * t * (p_same / 0.45)), p_cross)
        draw = rng.rand(nodes, nodes) < p
        w = np.where(same, rng.gamma(2.0, 0.5, (nodes, nodes)),
                     rng.gamma(2.0, 0.2, (nodes, nodes)))
        A = np.triu(draw * w, k=1)
        A = A + A.T
        if t < 2:
            A[:n_isolated, :] = 0
            A[:, :n_isolated] = 0
        A = sp.csr_matrix(A)
        sp.save_npz(os.path.join(out, "adj_{}.npz".format(first_year + t)), A)
        deg = np.asarray((A > 0).sum(1)).ravel()
        print("{}  nodes {:>6}  edges {:>9,}  mean degree {:6.1f}  isolated {}".format(
            first_year + t, nodes, A.nnz // 2, deg[deg > 0].mean(), int((deg == 0).sum())))
    return first_year, first_year + snapshots - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, default=60)
    ap.add_argument("--snapshots", type=int, default=5)
    ap.add_argument("--communities", type=int, default=3)
    ap.add_argument("--degree", type=float, default=None,
                    help="target mean degree; default keeps the reference densities")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    y0, y1 = build(a.nodes, a.snapshots, a.communities, a.degree, a.seed)
    print("\nwrote synth/adj_{}.npz .. adj_{}.npz".format(y0, y1))

    years = "{}-{}".format(y0, y1)
    if a.nodes <= 200:
        print("now run (reference config):")
        print("  python train.py --dataset synth --src synth --years {} --attn_variant gat \\".format(years))
        print("      --epochs 60 --batch_size 20 --patience 10 \\")
        print("      --structural_head_config 4 --structural_layer_config 32 \\")
        print("      --temporal_head_config 4 --temporal_layer_config 32")
    else:
        bs = min(512, max(32, a.nodes // 8))
        print("now run (default model, batch scaled to the graph):")
        print("  python train.py --dataset synth --src synth --years {} --attn_variant gat   --batch_size {}".format(years, bs))
        print("  python train.py --dataset synth --src synth --years {} --attn_variant gatv2 --batch_size {}".format(years, bs))


if __name__ == "__main__":
    main()
