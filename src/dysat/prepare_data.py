"""Convert the per-year adjacency matrices into DySAT's `<source>/graphs.npz`.

Reads   <networks>/<source>/adj_<year>.npz     (notebooks/01_network_construction.ipynb)
Writes  <dysat_input>/<source>/graphs.npz      (consumed by train.py --dataset <source>)

Both roots come from `scisoc.config.paths`; `--src` / `--out` override them.

    python prepare_data.py --source news
    python prepare_data.py --source paper --years 1990 2023
    python make_synth.py && python prepare_data.py --source synth --src synth --years 1990 1994

Every snapshot keeps ALL N nodes (isolated ones included) so that node ids line
up across years and with the vocabulary in `node_info.pkl`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # <repo>/src -> scisoc
from scisoc.config import paths


def build(src: Path, out: Path, years: range) -> Path:
    graphs, N = [], None
    for y in years:
        f = src / "adj_{}.npz".format(y)
        if not f.is_file():
            raise FileNotFoundError(
                "{} not found. Run notebooks/01_network_construction.ipynb first, "
                "or point --src at the directory that holds adj_<year>.npz.".format(f)
            )
        A = sp.load_npz(f).tocsr()
        A.setdiag(0)
        A.eliminate_zeros()
        if N is None:
            N = A.shape[0]
        assert A.shape == (N, N), "fixed vocabulary expected: {} has shape {}".format(f, A.shape)
        G = (nx.from_scipy_sparse_array(A) if hasattr(nx, "from_scipy_sparse_array")
             else nx.from_scipy_sparse_matrix(A))
        G.add_nodes_from(range(N))
        graphs.append(G)
        print(y, "nodes", G.number_of_nodes(), "edges", G.number_of_edges())

    out.mkdir(parents=True, exist_ok=True)
    # nx.Graph is iterable, so np.array(graphs) would unpack the nodes.
    arr = np.empty(len(graphs), dtype=object)
    for i, g in enumerate(graphs):
        arr[i] = g
    dest = out / "graphs.npz"
    np.savez(dest, graph=arr)
    print("wrote", dest, "T =", len(graphs), "N =", N)
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="news | paper | synth")
    p.add_argument("--years", nargs=2, type=int, default=[1990, 2023], metavar=("FIRST", "LAST"))
    p.add_argument("--src", default=None,
                   help="directory holding adj_<year>.npz (default: <networks>/<source>)")
    p.add_argument("--out", default=None,
                   help="output directory (default: <dysat_input>/<source>)")
    a = p.parse_args()

    src = Path(a.src).expanduser() if a.src else paths.networks / a.source
    out = Path(a.out).expanduser() if a.out else paths.dysat_input / a.source
    build(src, out, range(a.years[0], a.years[1] + 1))


if __name__ == "__main__":
    main()
