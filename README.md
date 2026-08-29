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
  tables/         Backbone, core, main-path and persistence tables
  sequences/      Sequence perplexity outputs
check_setup.py    One-command environment check against the bundled sample
```

Results are summarised in this file. The long-form reports they condense are kept
locally in `_superseded/` and are not tracked.

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
PERIOD_VARIANT = "raw"      # "raw" | "detrended"
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

Notebooks 03 to 06 need no raw input at all; they run off `data/processed/`. Retraining takes
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

Two megabytes in total. The DySAT checkpoints themselves are 8.9 MB each and notebook 02
rebuilds them, so only the training curves beside them are tracked.

Concept names and edge persistence windows are in [`results/tables/`](results/tables/) in
full. Check the environment:

```bash
python check_setup.py
```

```
Embedding (3, 29312, 32)   vocabulary 29,312

1995  active concepts  2,496   median rank of the true neighbour   438   (chance: 1,248)
2010  active concepts  5,617                                       513   (chance: 2,808)
2023  active concepts  7,560                                       380   (chance: 3,780)
```

Notebook 06 loads the full embedding when present and falls back to the sample with a
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
1998, 2009, 2017 and 1998, 2010, 2017. The raw joint fit, which the figures used before, is
not stable in the same way: BIC selects K = 5 there while the modal bootstrap K is 4
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

1. **The news retained-weight share is unstable**, swinging 49.4–71.1% (sd 0.041) and
   correlating −0.62 with that year's edge density. Papers are steady at 52.7–56.9%
   (sd 0.008), so this is a news-side confound.
2. **News 1991 is an outlier** — 1,538 nodes and 14,386 edges against roughly 2,300 and
   38,000 in the surrounding years, and ten components even after the spanning tree. This
   looks like a collection artefact and the year should be handled separately.
3. **Node counts differ from the core–periphery tables**, because the backbone runs on the
   29,312-concept filtered vocabulary to stay index-aligned with the embedding. News 1990
   is 2,260 nodes here and 2,440 there.

A published reference implementation was used to check the α computation. It agrees on the
integral but does not apply the OR rule on undirected edges — it overwrites α as it walks
each endpoint, so the result depends on node order. Only 83.0% of edges matched
min(α_ij, α_ji) on this data.

### Sequence perplexity — notebook 05

- **Concept chains in the scientific literature stop moving around 2009.** Over
  2009–2023 the slopes are −0.64 %/yr for perplexity, −0.31 %/yr for rank and
  −0.58 %/yr for normalised rank: all three agree, and all three are near zero.
- **No stabilisation point can be identified for news.** Over the same window the three
  slopes are −0.10, −1.29 and −3.51 %/yr — a 35-fold disagreement. The news vocabulary
  grows by 25.9% over that period while the paper vocabulary grows by 4.1%, and the two
  normalisations respond to that growth in opposite directions.
- **The two arenas share almost no persistent relational structure.** Only twelve edges
  persist for twenty years or more in both corpora, and eleven of them have `software` as
  an endpoint.

## Environment

`numpy`, `pandas`, `scipy`, `matplotlib`, `networkx`. Notebook 01 additionally needs
`cpnet`, 02 needs `torch`, and 04 needs `ruptures`. Nothing else needs `torch` — `.pt`
files are read as zip archives of raw float32.
