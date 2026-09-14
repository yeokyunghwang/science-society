# Science–Society: Co-evolution of Conceptual Structure

How does the conceptual structure of scientific literature relate to that of news
coverage? This repository builds concept co-occurrence networks from research papers
and from news articles over **1990–2023** on one fixed vocabulary, and measures three
things on them: **core–periphery structure**, a **dynamic graph embedding** of every
concept in every year, and the **perplexity of persistent concept chains** read off
that embedding.

Everything runs from the numbered notebooks plus one training package. There are no
hard-coded absolute paths: every location resolves through
[`src/scisoc/config.py`](src/scisoc/config.py).

---

## Layout

```
notebooks/                 Three stages, run in numbered order
  01_network_construction.ipynb
  02_core_periphery.ipynb
  03_backbone_perplexity.ipynb
src/
  scisoc/                  Shared configuration and loaders
    config.py              paths.* — the only place a directory is named
    io.py                  node_info / adjacency / embedding loaders
  dysat/                   DySAT, representation-only full-softmax variant
    README.md              what was changed relative to the upstream repo
    prepare_data.py train.py export_probs.py make_synth.py
    flags.py tf_compat.py _repo.py
    models/DySAT/  utils/
data/
  raw/                     Subject terms per document per year — the only irreducible input
  processed/
    networks/<source>/     adj_<year>.npz, node_info.pkl, vocab.npy, active.npy   (01)
    dysat/<source>/        graphs.npz — DySAT input                               (prepare_data)
    embeddings/            <source>_E.npz — E [N, T, F], active [N, T]            (train)
    backbone/              <source>_<year>_backbone.parquet, alpha*/…             (03)
results/
  cp_results/              Core–periphery indices, per year and per run           (02)
  figures/                 Manuscript figures                                     (01–03)
  dysat/                   Training logs and checkpoints (not tracked)
old/                       Superseded material, kept locally, not tracked
```

`src/dysat/` used to live in a separate repository (`dysat-pytorch`). It is part of this
one now, so the trainer and the notebooks share a single path configuration and a single
vocabulary.

---

## Pipeline

| # | Stage | Reads | Writes |
|---|---|---|---|
| 01 | [`01_network_construction`](notebooks/01_network_construction.ipynb) | `data/raw/<source>_subject_by_year.pkl` | `networks/<source>/adj_<year>.npz`, `node_info.pkl`, descriptive figure |
| 02 | [`02_core_periphery`](notebooks/02_core_periphery.ipynb) | `networks/` | `cp_results/{grid,main}/`, `<source>_{grid,main,band}.csv`, CP index figure |
| E | [`src/dysat`](src/dysat/) | `networks/` | `dysat/<source>/graphs.npz`, `embeddings/<source>_E.npz` |
| 03 | [`03_backbone_perplexity`](notebooks/03_backbone_perplexity.ipynb) | `networks/`, `embeddings/` | `backbone/`, `persistent_paths.pkl`, backbone + persistence figures |

```
01 ──┬──→ 02  (core–periphery indices)
     │
     ├──→ E   (DySAT: adj_<year>.npz → graphs.npz → E [N, T, F])
     │          │
     └──→ 03 ←─┘   backbone → persistent edges → persistent chains → perplexity
```

**One vocabulary throughout.** Notebook 01 fixes a single concept index per arena and
writes every yearly matrix on it, so node id `i` means the same concept in 1990 and in
2023, in the core–periphery step and in the embedding. DySAT keeps all `N` nodes in
every snapshot, isolated ones included, for the same reason.

### 01 — Network construction

Each document contributes its subject terms. A document with `d ≥ 2` terms adds weight
`1/(d−1)` to each of its pairs, built as `Xᵀ X` with `X` holding `√(1/(d−1))`, so a long
subject list does not dominate. `node_info.pkl` carries the vocabulary, the per-year
document frequency `[T, V]`, and the yearly document/edge counts used downstream.

### 02 — Core–periphery

`cpnet.KM_ER` (Kojaku–Masuda) on each yearly matrix, after a concept filter on the
full-period share. The grid sweep over (min share, max share) is scored by median `Q_cp`;
the selected specification is then re-run with `n_runs = 10` to give the four indices
with run-to-run bands:

| index | meaning |
|---|---|
| `C_t` | churn of core nodes against the previous year |
| `R_t` | core nodes as a share of active nodes |
| `S_t` | number of cores |
| `H_t` | concentration (Herfindahl over core sizes) |

Per-cell results are pickled, so an interrupted sweep resumes where it stopped.

### E — DySAT embedding

`src/dysat/` is a modified DySAT: negative sampling is replaced by a **full softmax over
the nodes active in that year**, so that

```
p_t(u | v) = softmax_u ⟨E[v, t], E[u, t]⟩          u ≠ v, u active at t
```

is a normalised conditional distribution and its perplexity is directly interpretable.
Edge weights enter the structural attention as an additive log-bias,
`e_uv = LeakyReLU(f1_u + f2_v) + β·log A_uv`, with `β` a learned scalar. Training holds
out a fraction of edges per snapshot and early-stops on the loss over those held-out
pairs. [`src/dysat/README.md`](src/dysat/README.md) lists every change against the
upstream repository, including two bugs found there.

### 03 — Backbone, persistent chains, perplexity

The disparity filter (Serrano et al.) scores every edge; the backbone keeps the
`3·N_active` strongest, plus the maximum spanning tree so the backbone stays connected.
`α ∈ {0.01, 0.05, 0.1, 0.2}` variants are written alongside for sensitivity, and the
backbone is compared against an RCA filter on strength preservation, year-to-year
stability and weight retention.

An edge is **persistent** when it sits in the backbone for at least 20 consecutive
years. A **persistent chain** is a path of length 8 through persistent edges. Each
chain is scored per year by masking one concept at a time and predicting it from the
mean of the others, against all concepts active that year — a pseudo-perplexity that
falls as the chain's concepts settle into a stable neighbourhood.

---

## Running it

```bash
git clone https://github.com/yeokyunghwang/science-society.git
cd science-society
pip install -e .              # puts `scisoc` on the path; see pyproject.toml
```

The raw subject tables are not in the repository. Point at them once, either by
exporting the variable or by writing a `.env` file in the repository root:

```bash
cp .env.example .env          # then edit SCISOC_RAW
```

| variable | default | what it is |
|---|---|---|
| `SCISOC_ROOT` | inferred from `src/scisoc/config.py` | repository root |
| `SCISOC_RAW` | `<root>/data/raw` | `news_subject_by_year.pkl`, `paper_subject_by_year.pkl` |
| `SCISOC_DATA` | `<root>/data/processed` | processed data (point elsewhere if disk is tight) |
| `SCISOC_RESULTS` | `<root>/results` | figures and tables |

Then:

```bash
jupyter lab notebooks/01_network_construction.ipynb     # ~30 min per arena
jupyter lab notebooks/02_core_periphery.ipynb           # grid sweep is the long part

cd src/dysat                                            # TF 1.x or 2.x, see its README
python prepare_data.py --source news --years 1990 2023
python train.py --dataset news --time_steps 34
cd ../..

jupyter lab notebooks/03_backbone_perplexity.ipynb
```

Each notebook's first cell prints the resolved paths, so a wrong `SCISOC_RAW` shows up
immediately rather than three cells later.

### Dependencies

`requirements.txt` for the notebooks; TensorFlow is only needed for `src/dysat/` and is
listed separately there because it pins `numpy < 2`:

```bash
pip install -r requirements.txt
pip install "tensorflow-cpu>=2.16" "numpy<2"       # only for src/dysat
```

---

## What is and is not tracked

Tracked: notebooks, `src/`, `results/cp_results/` (the core–periphery indices and the
per-year pickles behind them), `results/figures/`.

Not tracked: `data/` (rebuilt by 01 and the trainer), `results/dysat/` (checkpoints),
`results/cp_results/grid/` (1,700 files the grid sweep regenerates), `old/`, and
archives. The paths the notebooks read from are the paths the `.gitignore` describes, so
dropping the raw files into `data/raw/` is all it takes to reproduce everything.
