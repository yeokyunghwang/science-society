#!/usr/bin/env python3
"""Smoke test for the bundled sample data.

Loads the embedding and the co-occurrence networks, then reports where the
embedding ranks concepts that actually co-occur. Run from the repository root.

    python check_setup.py
"""
import csv
import os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

YEARS = {1995: 0, 2010: 1, 2023: 2}

vocab = {}
with open(os.path.join(ROOT, "results", "tables", "backbone_vocab.csv")) as fh:
    rd = csv.DictReader(fh); k = rd.fieldnames
    for row in rd:
        vocab[int(row[k[0]])] = row[k[1]]

E = np.load(os.path.join(ROOT, "data", "processed", "embeddings", "news_embeddings_sample.npy"))
print(f"Embedding {E.shape}   (3 years x 29,312 concepts x 32 dims)")
print(f"Vocabulary {len(vocab):,}\n")

rng = np.random.default_rng(0)
for year, t in YEARS.items():
    z = np.load(os.path.join(ROOT, "data", "processed", "networks_emb", f"news_{year}_adj_f.npz"),
                allow_pickle=True)
    data, ind, indptr = z["data"], z["indices"], z["indptr"]
    deg = np.diff(indptr)
    active = np.where(deg > 0)[0]
    Z = np.ascontiguousarray(E[t][active])

    # 200 well-connected concepts; for each, try to recover one of its neighbours
    src = rng.choice(active[deg[active] >= 5], size=200, replace=False)
    ranks = []
    for u in src:
        v = int(ind[indptr[u] + rng.integers(deg[u])])
        scores = Z @ E[t][u]                       # inner product, as used in training
        target = np.searchsorted(active, v)
        ranks.append(int((scores > scores[target]).sum()) + 1)

    print(f"{year}  active concepts {len(active):>6,}   "
          f"median rank of the true neighbour {int(np.median(ranks)):>6,}   "
          f"(chance: {len(active)//2:,})")

u = next(i for i, n in vocab.items() if n == "automation")
z = np.load(os.path.join(ROOT, "data", "processed", "networks_emb", "news_2023_adj_f.npz"), allow_pickle=True)
active = np.where(np.diff(z["indptr"]) > 0)[0]
scores = E[2][active] @ E[2][u]
print("\nNearest concepts to 'automation' in 2023:")
for j in np.argsort(-scores)[1:8]:
    print(f"   {scores[j]:6.2f}   {vocab.get(int(active[j]), active[j])}")
