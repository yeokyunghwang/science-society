# Science–Society: Co-evolution of Conceptual Structure

How does the conceptual structure of scientific literature relate to that of news
coverage? This repository compares concept co-occurrence networks built from research
papers and from news articles over 1990–2023, tracking core–periphery structure,
latent position through dynamic graph embedding, period boundaries, and the
stabilisation of concept chains.

**Start here.** [Results](#results) — 개념 연쇄의 안정화(notebook 05)가 먼저 오고, 시기
구분(notebook 03)이 뒤따른다. 코드는 [`notebooks/`](notebooks/)를 번호 순서로 읽으면 된다.

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

### 개념 연쇄의 안정화 — notebook 05

#### Backbone network 추출

disparity filter (Serrano, Boguñá & Vespignani 2009). 고정 α 대신 **density rule** — 매년
`|E| = 3N` 개를 남기고 최대 신장나무를 합집합한다. 연도·코퍼스마다 밀도가 달라 같은 α가
전혀 다른 밀도의 백본을 만들기 때문이다.

34년 전체에서 나온 서로 다른 백본 엣지: **news 160,112 / paper 196,246**.

각 엣지의 최장 연속 구간 길이별 누적 개수:

| 최장 연속 구간 | news | paper | 비 |
|---|---|---|---|
| ≥ 34년 (전 기간) | 163 | 9,674 | 59.3× |
| ≥ 30년 | 393 | 12,422 | 31.6× |
| ≥ 25년 | 1,495 | 16,799 | 11.2× |
| **≥ 20년 (사용)** | **2,155** | **22,997** | **10.7×** |
| ≥ 15년 | 3,302 | 29,570 | 9.0× |
| ≥ 10년 | 6,361 | 38,297 | 6.0× |
| ≥ 5년 | 16,367 | 56,797 | 3.5× |
| ≥ 1년 | 160,112 | 196,246 | 1.2× |

임계를 올릴수록 격차가 1.2배에서 59.3배로 벌어진다. 뉴스의 지속 엣지 중 34년을 다 버틴
것은 7.6%, 논문은 42.1%다.

백본 자체에 대해 남는 이슈 넷:

1. **뉴스의 잔존 가중치 비중이 불안정하다** — 48.4–67.8%(sd 0.037)로 흔들리고 그해 엣지
   밀도와 −0.52로 상관한다. 논문은 52.0–56.4%(sd 0.010)로 안정적이므로 뉴스 쪽 교란 요인이다.
2. **news 1991은 이상치** — 노드 1,538 · 엣지 14,386으로, 1990년(2,260 · 40,084)과
   1992년(2,306 · 38,797)과 크게 다르고 신장나무를 붙인 뒤에도 성분이 10개다. 수집
   아티팩트로 보이며 별도 취급해야 한다.
3. **노드 수가 core–periphery 표와 다르다** — 백본은 임베딩과 인덱스를 맞추려고 29,312개
   필터 어휘 위에서 돌기 때문이다. news 1990이 여기서는 2,260개, 저기서는 2,440개다.
4. **참조 구현과 17%가 어긋난다** — `aekpalakorn/python-backbone-network`는 적분값은
   일치하나 무방향 엣지에 OR rule을 적용하지 않는다(양 끝을 돌며 α를 덮어써 노드 순서에
   의존). 이 데이터에서 `min(α_ij, α_ji)`와 일치한 엣지는 83.0%다.

필터가 실제로 일하고 있는지는 같은 엣지 수의 단순 가중치 상위컷과 비교해 확인했다 —
Jaccard 0.58–0.71로 엣지의 30~40%가 다르다.

#### Persistent path 추출

`longest_run ≥ L`인 엣지만 남긴 **지속망**을 만들고 그 위에서 경로를 뽑는다.

사슬 `v₁ – … – v_K`가 지속적이려면 **K−1개 엣지의 지속 구간을 교집합한 길이가 L년 이상**
이어야 한다. 엣지가 각각 오래간 것으로는 부족하고, 사슬 전체가 같은 창 안에서 함께 살아
있어야 한다.

기준은 **L = 20년, K = 6**(주 분석)과 **K = 8**(길이 민감도). 실제로 뽑힌 경로의 유지
기간은 중앙값 21–22년으로 대부분 임계 바로 위에 몰려 있다.

![Figure 1](results/figures/fig1_structure.png)

**(a)** 지속 임계 L별 생존 백본 엣지 수.
**(b)** 지속 경로를 한 칸 걸었을 때 조건을 만족하는 다음 칸이 몇 개인가 — 방향 엣지
전수(news 4,310 / paper 45,994)에 대해 센 값이므로 표본이 아니다.

| | 중앙값 분기 | 막다른 길 |
|---|---|---|
| news | 15 | 10.3% |
| paper | 48 | 3.1% |

뉴스 지속망은 `automation` 하나가 863개 노드 중 774개와 연결된 **별 모양**이고 연결선수
중앙값이 1이다. 논문은 4,379개 노드가 중앙값 3으로 고르게 얽혀 있다. 사슬을 이어가기가
뉴스에서 어려운 이유가 이것이다.

> **이전 판에서 고친 것.** (b)는 원래 "랜덤워크가 길이 K의 지속 연쇄를 만들 확률"이었다
> (K=8에서 news 10% 대 paper 30%). 그 숫자는 그래프가 아니라 표본추출기의 성질이었고
> 희소함을 과장했다 — **길이 K의 지속 경로가 존재하는 노드는 K=20까지도 양쪽 다 99%**다.
> 실제 차이는 경로의 존재가 아니라 분기에 있으므로 분기를 직접 센다.

#### 뉴스 vs 논문 — 비교 대상 선정

두 코퍼스가 각자 자기 경로를 걸으면 궤적 차이가 **영역 때문인지 경로 때문인지** 구분되지
않는다. 두 지속망의 합집합으로 "공통 그래프"를 만드는 것은 더 나쁘다 — 한쪽에 실제로 없는
엣지를 걷게 된다.

그래서 관계 **하나만** 공유시킨다. **앵커** = 양쪽 코퍼스 모두에서 20년 이상 지속인 엣지.

| 키워드 1 | 키워드 2 | news | paper |
|---|---|---|---|
| architecture | software | 1998–2023 | 1990–2023 |
| automation | software | 1990–2023 | 1990–2023 |
| data collection | software | 1997–2023 | 2001–2023 |
| data processing | software | 1998–2023 | 1990–2023 |
| interoperability | software | 1998–2023 | 1991–2023 |
| medical imaging | software | 1998–2023 | 1997–2019 |
| ~~medical imaging~~ | ~~tomography~~ | ~~2000–2023~~ | ~~1993–2023~~ |
| project management | software | 1996–2023 | 1995–2023 |
| software | systems design | 1998–2017 | 1998–2023 |
| software | usability | 1998–2023 | 1994–2023 |
| software | user interface | 1992–2023 | 1990–2023 |
| software | visualization | 1998–2023 | 1990–2023 |

`medical imaging – tomography`는 제외했다. 뉴스 지속망에서 `tomography`의 연결선수가 1이고
그 유일한 이웃이 앵커의 반대쪽 끝(`medical imaging`)이라, 앵커를 가운데 두는 경로를
**하나도 만들 수 없다**. 논문에서는 500개가 나오지만 짝이 없으므로 양쪽에서 함께 뺐다.
**사용 11개.**

두 지속망이 공유하는 개념은 111개인데 그 사이에서 양쪽 모두 20년 이상 지속되는 관계는
12개뿐이고, **쓸 수 있는 11개가 전부 `software`를 한쪽 끝으로 갖는다**. 이 결과는 두 영역이
공유하는 지속 구조 전체에 대한 것이지, 임의의 주제 영역으로 일반화되지 않는다.

**경로 표본.** 앵커를 한가운데 고정하고(`· · A B · ·`) 양옆으로 `K//2 − 1`칸씩 뻗되, 한 칸
뻗을 때마다 구간 교집합이 20년 이상인 이웃 중에서만 고른다. 앵커·코퍼스·K당 500개. 앵커를
가운데 둔 K=6 지속 경로는 news 776,000–1,453,000개, paper 200만 개 이상이므로 전수 채점은
불가능하다.

**신뢰구간은 2단계.** ① 앵커 안에서 500개 경로를 평균 → 앵커·연도 값 ② 앵커 11개를 표본
단위로 평균·SD·95% CI (t, df = 10). 같은 앵커를 공유하는 경로는 독립이 아니므로 5,500개를
표본으로 세면 CI가 실제보다 좁아진다.

#### Perplexity 측정

연도 t의 임베딩에게 경로를 한 칸씩 물어본다. 문맥 `c_k = Σ_{i≤k} w_i·z_{v_i,t}`
(`w_i ∝ λ^(k−i)`, λ = 0.5), 점수 `s(u) = c_k·z_{u,t}`, 확률 `p = softmax_u s(u)`. 후보는
그해 활성 개념 전체이고 지나온 노드는 분모에서도 뺀다. 조건이 앞선 개념들로만 이루어지므로
pseudo-perplexity가 아닌 **진짜 perplexity**다. 무방향이므로 정·역 양방향의 기하평균.
임베딩은 기존 LexicalDySAT `[34 × 29,312 × 32]`를 재학습 없이 쓴다.

지표 셋: **PP**, **rank**(정답이 후보 중 몇 등), **rank_norm**(rank ÷ 그해 활성 개념 수).

![Figure 2](results/figures/fig2_anchor_mean.png)

앵커 11개의 평균과 앵커 수준 95% CI (K = 6). 1990–92는 causal mask 때문에 참조할 과거가
거의 없는 warm-up 구간이라 해석하지 않는다.

| rank_norm (%) | news | paper |
|---|---|---|
| 1990 | 20.11 [17.79, 22.42] | 9.17 [7.33, 11.01] |
| 2000 | 4.31 [3.52, 5.09] | 3.17 [2.53, 3.82] |
| 2010 | 2.71 [2.20, 3.23] | 2.19 [1.71, 2.67] |
| 2016 | 2.27 [1.79, 2.75] | 2.14 [1.67, 2.61] |
| 2023 | 1.78 [1.49, 2.06] | 2.09 [1.61, 2.57] |

**논문은 2010년에 멈춘다** — 2.19% → 2.09%, 13년 동안 0.10%p. 원순위로는 383등 → 378등이다.
**뉴스는 계속 내려간다** — 2.71% → 1.78%. 2016년경 두 곡선이 만나고 이후 역전된다.

![Figure 3](results/figures/fig3_anchor_panels.png)

앵커마다 한 칸. 선은 그 앵커 500개 경로의 평균, 띠는 경로 간 사분위 범위.

| 2009–2023 기울기 (%/년) | news | paper |
|---|---|---|
| K=6 PP | +0.20 ± 0.27 | −0.33 ± 0.40 |
| K=6 rank | −0.66 ± 0.48 | −0.22 ± 0.33 |
| K=6 rank_norm | −2.88 ± 0.48 | −0.49 ± 0.33 |
| K=8 rank_norm | −3.13 ± 0.42 | −0.51 ± 0.24 |

대응표본 검정 — **news − paper = −2.38 %/년 [−2.82, −1.95], 11/11 앵커에서 부호 동일,
t(10) = −12.2, p < 0.0001** (K=8: −2.62 [−3.05, −2.18]).

#### 읽을 때 주의할 것

**수준은 비교할 수 없다.** 뉴스와 논문은 각자 다른 임베딩 모형(`news_embeddings.pt`,
`paper_embeddings.pt`)으로 채점되고 후보 풀도 다르다. "2023년 뉴스 134등 < 논문 378등이니
뉴스가 더 굳었다"는 성립하지 않는다. **비교 가능한 것은 기울기뿐이다.**

**PP는 인용하지 않는다.** 임베딩이 양성 1 : 음성 1로 학습되어 softmax가 보정되어 있지 않다.
순위는 서열만 쓰므로 이 문제에 면역이지만 PP는 아니다. PP의 쓸모는 값이 아니라 **불일치
탐지기**다 — PP가 순위와 반대로 움직이면 후보 풀이 빠르게 자라고 있다는 신호다.

**크기와 방향을 분리한다.** 정규화가 필요 없는 원순위에서도 뉴스가 논문보다 후기에 더
움직인다(K=6 p = 0.048, K=8 p = 0.007). **방향**은 정규화 선택에 기대지 않는 결론이다.
다만 −2.38 %/년이라는 **크기**는 후보 풀 증가분(news +25.9%, paper +4.1%)을 포함한다.
원순위(−0.66)와 정규화 순위(−2.88)가 진짜 값을 위아래로 감싼다고 읽는 것이 정확하다.

**대조군이 없다.** 실재하지 않는 사슬을 같은 방식으로 채점해 그 대비 비율로 보고해야 순위
하락이 사슬의 성질인지 허브 효과인지 구별된다. 이것이 갖춰지면 세 지표가 하나로 줄고
코퍼스 간 비교도 열린다. 현재 가장 큰 미완 부분이다.

#### 경로 예시

앵커·코퍼스별로 2023년 rank_norm이 가장 낮은 경로. 가운데 두 자리는 설계상 항상 앵커이므로
두 코퍼스가 다른 것은 나머지 네 자리뿐이다.

```
automation – software
  news   0.08%   alliances → advisors → [software – automation] → chief executive officers → information systems
  paper  0.29%   image (mathematics) → cluster analysis → [software – automation] → function (biology) → point (geometry)

medical imaging – software
  news   1.11%   marketing → internet → [software – medical imaging] → automation → wireless networks
  paper  0.93%   function (biology) → focus (optics) → [software – medical imaging] → segmentation → pattern recognition (psychology)

software – visualization
  news   0.58%   software industry → executives → [software – visualization] → automation → trademarks
  paper  0.27%   constraint (computer-aided design) → function (biology) → [software – visualization] → work (physics) → image (mathematics)
```

뉴스는 남은 자리를 기업·경영 담론(`alliances`, `chief executive officers`,
`acquisitions & mergers`, `customer services`)으로, 논문은 추상적 방법 어휘
(`image (mathematics)`, `function (biology)`, `point (geometry)`, `cluster analysis`)로
채운다. 관계는 공유하는데 그 관계를 둘러싸는 어휘가 갈린다.

단, 논문 쪽 이웃은 WoS 주제어의 다의어(`point`, `function`, `image`, `class`, `key`)가
지배한다. 어휘 자체의 성질이므로 질적으로 인용할 때 고지해야 한다.

전체 44개 경로는 `results/sequences/anchor_examples.csv`에 있다.

---

### 시기 구분 — notebook 03

#### Period identification — notebook 03, §3.6–3.7

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

#### Periodisation — notebook 03, §3.8

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
