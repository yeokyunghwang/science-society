# Science–Society: Co-evolution of Conceptual Structure

How does the conceptual structure of scientific literature relate to that of news
coverage? This repository compares concept co-occurrence networks built from research
papers and from news articles over 1990–2023, tracking core–periphery structure,
latent position through dynamic graph embedding, period boundaries, and the
stabilisation of concept chains.

**Start here.** [Results](#results) summarises what each stage found and what it cannot
support. For the code, read [`notebooks/`](notebooks/) in numbered order.

---

## Layout

```
notebooks/        Five stages, run in numbered order
data/
  raw/            Subject terms per document per year — the only irreducible input
  processed/
    networks_cp/  Yearly graphs on that year's own vocabulary (notebook 01)
    networks_emb/ The same graphs on one vocabulary fixed across all 34 years (02)
    lexical/ embeddings/ backbone/
models/           Trained DySAT checkpoints
results/
  figures/        Figures
  cp/             Core–periphery indices, temporal identification, periodisation
  tables/         Backbone, vocabulary and persistence tables
  sequences/      Sequence perplexity outputs
check_setup.py    One-command environment check against the bundled sample
```

Results are summarised in this file.

## Pipeline

| # | Notebook | What it does |
|---|---|---|
| 01 | [`01_core_periphery`](notebooks/01_core_periphery.ipynb) | Extracts **core–periphery structure** from the yearly co-occurrence networks with the Kojaku–Masuda algorithm and computes four structural indices (C, R, S, H) |
| 02 | [`02_embedding_pipeline`](notebooks/02_embedding_pipeline.ipynb) | Trains **LexicalDySAT**, a DySAT variant with a lexical-neighbour attention layer inserted before the co-occurrence layer. Output: `[34 years × 29,312 concepts × 32 dims]` |
| 03 | [`03_temporal_identification`](notebooks/03_temporal_identification.ipynb) | Assembles the indices from 01 and 02 into **yearly state vectors**, screens for change points through three independent layers, tests cross-arena ordering and persistence, and **fixes the period boundaries** (§3.8) — model selection, agreement across specifications, robustness, bootstrap stability |
| 04 | [`04_figures`](notebooks/04_figures.ipynb) | Draws the **manuscript figures**. Runs no new analysis; period shading is read from 03's output rather than hard-coded |
| 05 | [`05_backbone_sequence_perplexity`](notebooks/05_backbone_sequence_perplexity.ipynb) | **Backbone → edge persistence → persistent concept chains → yearly perplexity.** One continuous process, so one notebook |

### How the stages depend on each other

**No output is produced by more than one notebook.** Stages depend on each other but do
not overlap.

```
01 ─┐
    ├──→ 03 (state vectors → diagnostics → period boundaries) ──→ 04 (figures)
02 ─┘                                                              ↑
    └──────────────────────────────────────────────────────────────┘
    └────→ 05 (backbone → chains → perplexity)
```

03 builds the state vectors, screens them, and — in §3.8, reading nothing but the state
vectors it just wrote — settles the period boundaries. Notebook 04 reads that
`periodization_main.csv` and shades the periods. **Period boundaries are decided in one
place and nowhere else** — 04 holds no values of its own.

```python
# 04_figures.ipynb — nothing hard-coded
PERIOD_CSV     = Path("../results/cp/periodization/periodization_main.csv")
PERIOD_VARIANT = "detrended"   # "raw" | "detrended"
PERIOD_SPEC    = "joint"    # "joint" | "news" | "paper"
AI_PERIODS     = load_periods()
```

If §3.8 has not been run, notebook 04 stops with a message saying so.

### What a persistent concept chain is

A chain `v₁ – v₂ – … – v_K` counts as persistent when **all K−1 edges sit in the backbone
at the same time** for at least L consecutive years (L = 20 by default). It is not enough
for the individual edges to be long-lived; the whole chain has to hold together over the
same window. If `robot–control` runs 1990–2010 and `control–power` runs 2005–2023, they
overlap for only six years and the chain does not qualify.

Chains are measured at two lengths, **K = 6 and K = 8**, and the two are compared against
each other; yearly ranks differ by 1.3–1.6% on average, so the results are insensitive to
the choice.

---

## Data

Nothing under `data/` or `models/` is tracked. Only the raw subject-term files are
irreducible input; given those, **notebooks 01 and 02 rebuild everything** — both sets of
co-occurrence networks, the lexical resources, the model checkpoints and the embedding
tensors — and notebook 05 rebuilds the backbone.

```
data/raw/{news,paper}_subject_by_year.pkl          supply this yourself
    ↓ notebook 01
data/processed/networks_cp/                            136 files, 302 MB
    ↓ notebook 02
data/processed/networks_emb/                           149 files, 589 MB
data/processed/lexical/
data/processed/embeddings/{news,paper}_embeddings.pt   122 MB each
models/{news,paper}_lexical_dysat.pt
    ↓ notebook 05
data/processed/backbone/
```

The raw files are large and are usually kept on an external drive. There is no need to copy
them in: set `SCISOC_RAW` to wherever they live and notebooks 01 and 02 will read from there.

```bash
SCISOC_RAW=/Volumes/MyDrive/scisoc_raw jupyter lab
```

Notebooks 03 to 05 need no raw input at all; they run off `data/processed/`. Retraining takes
a GPU and several hours, so copying `data/processed/` across is usually easier than rebuilding
it. Ask the author for either the raw input or the derived files.

### Two sets of networks

The same co-occurrence data is stored twice, under different vocabularies, because the two
analyses need different things from it.

`networks_cp/` indexes each year by the subject terms that appear in that year alone, so
1995 news is a 2,666 x 2,666 matrix and 2023 news is larger. Core–periphery structure is
recovered year by year and never compared index to index, so a vocabulary that tracks the
year is the natural one.

`networks_emb/` indexes every year by one vocabulary of 29,312 concepts held fixed across
1990–2023, giving a 29,312 x 29,312 matrix for each year. A dynamic embedding has to carry
the same node through time, so row *i* must mean the same concept in every year. Notebook 02
first writes the unfiltered version (`*_adj.npz`, 34,403 concepts) and then the filtered one
the model trains on (`*_adj_f.npz`).

Edge counts match between the two: news 1995 has 82,574 non-zero entries either way. Only the
index space differs.

### Sample data

A three-year sample of the news corpus ships with the repository, **at the exact paths the
notebooks read from**. Clone and the notebooks find data immediately; drop the full files
into the same folders and the same code runs on everything.

```
data/processed/embeddings/news_embeddings_sample.npy      [3, 29312, 32] — 1995, 2010, 2023
data/processed/networks_emb/news_{1995,2010,2023}_adj_f.npz
data/processed/networks_emb/global_subjects_filtered.pkl  concept vocabulary
data/processed/lexical/news_{1995,2010,2023}_lex_adj_f.npz
data/processed/lexical/base_{vocab,idx}.pkl               lexical base vocabulary
data/processed/lexical/subject_lexical_summary.csv        concept → lexical bases
models/{news,paper}_lexical_dysat_train_history.csv       training curves
```

About 20 MB in total, most of it the embedding sample. The DySAT checkpoints themselves
are 8.9 MB each and notebook 02 rebuilds them, so only the training curves beside them are
tracked.

Concept names and edge persistence windows are in [`results/tables/`](results/tables/) in
full. Check the environment:

```bash
python check_setup.py
```

```
Embedding (3, 29312, 32)   (3 years x 29,312 concepts x 32 dims)
Vocabulary 29,312

1995  active concepts  2,496   median rank of the true neighbour    438   (chance: 1,248)
2010  active concepts  5,617   median rank of the true neighbour    513   (chance: 2,808)
2023  active concepts  7,560   median rank of the true neighbour    380   (chance: 3,780)
```

Notebook 05 loads the full embedding when present and falls back to the sample with a
warning when not. **The sample cannot reproduce the published numbers.**

Notebooks resolve paths relative to `notebooks/`, so run them from there:

```bash
cd notebooks && jupyter lab
```

## Results

### Period identification — notebook 03, §3.6–3.7

Change points survive all three screening layers at **1998 and 2017 in news** and at
**2002 and 2013 in papers**.
Whether those points delimit periods rather than cut a trend is tested in §3.8 below; the
short answer is that news weakly supports a period structure and papers do not.

Two results look stronger than they are and should not be reported at face value.

- **No lead–lag ordering is identified at the aggregate level.** The cross-correlation of
  the two PC1 series peaks at lag 0 with r = 0.935 and a bootstrap p below 0.001, but both
  series are near-deterministic trends (correlation with year: news −0.932, paper −0.991)
  and |r| moves by only 0.043 across the whole ±5-year window. The block-bootstrap null
  destroys the trend, so any two trending series beat it. After first differencing no lag
  is significant (p = 0.862). Ordering claims have to rest on the concept-level Cox
  time-varying analysis instead.
- **The micropathway labels flip with one parameter.** Both observed paper–news gaps are
  exactly four years, which falls between the pairing gate (±2) and the classifier's own
  gate (±6). At `pair_tol` ≥ 4 both breakpoints relabel from latent accumulation and
  public-first reconfiguration to ordered cross-arena alignment. Report `pair_tol = 2` as
  the primary result and the other as sensitivity.

The persistence permutation test is anti-conservative, because the breakpoint years were
chosen by PELT to maximise between-segment separation. It answers "does this shift
persist beyond one year", not "is this breakpoint significant" — CUSUM already answered
the second question, in advance.

### Periodisation — notebook 03, §3.8

Boundaries are fitted to the 1991–2023 state vector by exact dynamic programming under a
piecewise-constant model, K chosen by BIC with penalty β = σ̂²·d·log T and segments held to
at least five years. Six specifications run: each arena alone (d = 6) and the two jointly
(d = 12), on the raw series and on the series with a global linear trend removed.

```
variant     spec     K   boundaries
raw         news     3   1998, 2009, 2017
raw         paper    5   1996, 2002, 2007, 2013, 2019
raw         joint    5   1998, 2003, 2009, 2014, 2019
detrended   news     3   1998, 2010, 2017
detrended   paper    2   2003, 2016
detrended   joint    3   1998, 2009, 2017
```

**The reported partition is the detrended joint fit**, and the figures shade it:

```
P1  1991–1997      P2  1998–2008      P3  2009–2016      P4  2017–2023
```

Four things pick that specification out. Bootstrap resampling puts K = 3 at 0.908, the
highest agreement any specification reaches, with boundary frequencies of 0.93, 0.86 and
0.71. It reproduces the news-only fit almost exactly — Hausdorff 1.0, Rand 0.966, F1 1.0
within ±2 years, the best agreement in the table. It does not move when the minimum segment
length is set to 5, 6 or 7 years. And the quadratic and Gaussian-kernel costs return
1998, 2009, 2017 and 1998, 2010, 2017. The raw joint fit is not stable in the same way:
BIC selects K = 5 there while the modal bootstrap K is 4
(0.51 against 0.384). Only 1998 survives everything — it appears in four of the six
specifications and never below frequency 0.92 in a bootstrap that contains it.

**This is a descriptive partition, not evidence of discontinuity.** The six measures are
badly collinear with the year itself: M correlates 0.995 with year in news and 1.000 in
papers, and five of the six clear |r| ≥ 0.85 on the paper side. Comparing models on the
state vector directly, papers are best explained by a plain linear trend — it beats the
piecewise-constant fit by 88 BIC points and a trend-with-shifts model by 40 — while news
prefers trend-with-shifts, leaving the piecewise-constant fit 45 points behind. A parametric
bootstrap against a trend-only null rejects for news at p = 0.048, which is barely, and not
for papers at p = 0.208. Detrending removes the CUSUM significance altogether (news
0.014 → 0.261, papers 0.008 → 0.183), and a calibration run shows why: against a pure trend
of moderate slope the CUSUM test fires 87% of the time. At the network level, where no
derived indicator is involved, news prefers a three-block model over distance decay
(BIC −181 against −28) while papers prefer distance decay by a wide margin (−646 against
−71).

The four segments therefore summarise the news trajectory and are a convenience for the
paper one. Period shading in the figures is an axis annotation, not a claim that something
broke in 1998, 2009 or 2017.

### Backbone — notebook 05, first stage

Edges are kept by the disparity filter (Serrano, Boguñá and Vespignani 2009) under a
density rule rather than a fixed α: each year keeps |E| = 3N edges, unioned with the
maximum spanning tree so the backbone stays connected. The filter is doing real work —
against an equal-sized top-weight cut the Jaccard overlap is 0.58–0.71, so 30 to 40% of
the edges differ.

Three residual issues to carry into any reading of the backbone results:

1. **The news retained-weight share is unstable**, swinging 48.4–67.8% (sd 0.037) and
   correlating −0.52 with that year's edge density. Papers are steady at 52.0–56.4%
   (sd 0.010), so this is a news-side confound.
2. **News 1991 is an outlier** — 1,538 nodes and 14,386 edges against 2,260 and 40,084 in
   1990 and 2,306 and 38,797 in 1992, and ten components even after the spanning tree. This
   looks like a collection artefact and the year should be handled separately.
3. **Node counts differ from the core–periphery tables**, because the backbone runs on the
   29,312-concept filtered vocabulary to stay index-aligned with the embedding. News 1990
   is 2,260 nodes here and 2,440 there.

A published reference implementation was used to check the α computation. It agrees on the
integral but does not apply the OR rule on undirected edges — it overwrites α as it walks
each endpoint, so the result depends on node order. Only 83.0% of edges matched
min(α_ij, α_ji) on this data.

### Persistent paths and path perplexity — notebook 05, second stage

An edge counts as present in a year if it is in that year's backbone proper; the MST
additions are connectivity repair, not significant edges, so they are excluded. For each
edge the longest unbroken run is recorded as `[run_start, run_end]`, and the threshold is
L = 20 years.

```
threshold L        news     paper     ratio
20 years          2,155    22,997     10.7x
25 years          1,495    16,799     11.2x
30 years            393    12,422     31.6x
34 years (all)      163     9,674     59.3x
```

The two persistent networks are shaped differently, not only sized differently. The news
one is a star: 863 nodes, and `automation` alone is adjacent to 774 of them, with a median
degree of 1. The paper one has 4,379 nodes at median degree 3.

A chain `v₁ – … – v_K` is a **persistent path** when the *intersection* of its K−1 edge
windows spans at least L years. Each edge lasting a long time separately is not enough —
the whole chain has to be alive over one common window.

#### Anchor selection

Comparing trajectories across corpora is confounded if each corpus walks its own chains: a
difference could be the arena or it could be the chain. Building paths on the union of the
two persistent networks is worse, because it walks edges one corpus does not have. Instead
one edge is shared and the rest of each path comes from that corpus's own graph.

An **anchor** is an edge that persists ≥ 20 years in *both* backbones. It is placed at the
centre of the path — `· · A B · ·` for K = 6, `· · · A B · · ·` for K = 8 — and each side is
extended by `K//2 − 1` steps through that corpus's own persistent graph, re-intersecting the
persistence window at every step.

The two networks share 111 concepts but only twelve edges that persist twenty years or more
on both sides, and **eleven of the twelve have `software` as an endpoint**. Those eleven are
usable; `medical imaging – tomography` yields no news path, because `tomography` has degree 1
in the news persistent graph and its single neighbour is the anchor itself.

Exhaustively counting the centred K = 6 persistent paths gives 776,000 to 1,453,000 per
anchor for news and more than 2,000,000 for paper (counting was stopped there), so 500 paths
per anchor and corpus are sampled rather than scored exhaustively. The sampler draws
uniformly among the neighbours that satisfy the window condition, which is not a uniform
sample over paths — it passes through high-degree nodes often.

#### Perplexity

The year-t embedding is asked, one step at a time, for the probability of the next concept:
context `c_k = Σ_{i≤k} w_i·z_{v_i,t}` with `w_i ∝ λ^(k−i)` and λ = 0.5, score
`s(u) = c_k·z_{u,t}`, and `p = softmax_u s(u)` over that year's active concepts with visited
nodes excluded. The conditioning set is only the preceding concepts, so the chain rule holds
and this is a true perplexity rather than a pseudo-perplexity. The network is undirected, so
forward and reverse passes are computed and combined by geometric mean. The embedding is the
existing LexicalDySAT `[34 × 29,312 × 32]`, used without retraining.

Three metrics are reported together, because the candidate set differs by corpus and grows
at different rates (news 1,538 → 7,560 active concepts, paper 12,577 → 18,200): perplexity
rises mechanically as the candidate set grows, normalised rank falls mechanically, and raw
rank is unaffected by either.

Confidence intervals are clustered in two stages — paths are averaged within an anchor
first, then the eleven anchor means are the sample (t, df = 10). Paths sharing an anchor are
not independent, and treating all 5,500 as the sample would shrink the interval spuriously.

#### Results

```
2009–2023 log slope, %/yr        news              paper
K=6  perplexity                  +0.20 ± 0.27      −0.33 ± 0.40
K=6  rank                        −0.66 ± 0.48      −0.22 ± 0.33
K=6  normalised rank             −2.88 ± 0.48      −0.49 ± 0.33
K=8  normalised rank             −3.13 ± 0.42      −0.51 ± 0.24
```

- **Holding the relation fixed, news chains are still consolidating and paper chains have
  stopped.** On normalised rank the paired difference is −2.38 %/yr (95% CI −2.82 to −1.95,
  t(10) = −12.2), and the sign is the same for all eleven anchors; at K = 8, −2.62
  (−3.05 to −2.18). On raw rank, which needs no normalisation, the difference is −0.44
  (p = 0.048) at K = 6 and −0.67 (p = 0.007) at K = 8 — so the direction does not depend on
  the normalisation choice, though the magnitude does. On the anchor mean the two cross
  around 2016: news falls from 20.11% of active concepts in 1990 to 1.78% in 2023, paper from
  9.17% to 2.09%, and paper's 2010 value is already 2.19%.
- **The metric disagreement survives the anchoring.** Perplexity moves the opposite way from
  the rank metrics (paired difference +0.53, p = 0.005). Fixing the chains rules out "the two
  arenas walk different chains" as the cause and leaves the growing candidate set: news gains
  25.9% active concepts over 2009–2023 against paper's 4.1%.
- **Same relation, different surrounding vocabulary.** With the middle two positions
  identical by construction, news fills the remaining four slots with business and IT
  discourse (`alliances`, `chief executive officers`, `acquisitions & mergers`,
  `customer services`) and paper with abstract method terms (`image (mathematics)`,
  `function (biology)`, `point (geometry)`, `cluster analysis`). The paper side is dominated
  by polysemous subject headings, which any qualitative reading has to flag.

The anchors are all `software` neighbours, which is forced by there being only twelve shared
persistent edges. The result describes the two arenas' entire shared persistent structure; it
does not generalise to other subject areas.

An earlier version of this stage sampled 2,000 random-walk sequences per corpus from the
whole persistent network and compared the two trajectories directly. It was dropped: the two
corpora walked different chains, so the arena and the chain could not be separated, and the
random walk over-sampled hubs. The anchor design replaces it.

## Environment

`numpy`, `pandas`, `scipy` and `matplotlib` throughout. Beyond those:

| notebook | also needs |
|---|---|
| 01 | `cpnet`, `lifelines`, `joblib`, `seaborn`, `tqdm`, `tqdm_joblib` |
| 02 | `torch`, `scikit-learn`, `spacy` with `en_core_web_sm`, `nltk`, `joblib`, `seaborn`, `tqdm` |
| 03 | `torch`, `ruptures`, `changepoint` |
| 04 | `torch`, `seaborn`, `tqdm` |
| 05 | `networkx` |

Notebook 05 reads the `.pt` embeddings as zip archives of raw float32, so it needs no
`torch` of its own.
