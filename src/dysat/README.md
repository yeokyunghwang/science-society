# DySAT → representation-only, full-softmax variant

원본: https://github.com/aravindsankar28/DySAT (TF 1.11 / Py 2.7).
이 패키지: Python 3 + TensorFlow 1.x **또는** 2.x(`compat.v1`)에서 동작. TF 2.16.2 CPU로 검증.

목적: 고정 어휘 N개 노드 × T개 연도 스냅샷을 한 번에 학습하고, 학습 후
`p_t(u | v) = softmax_u <E[v,t], E[u,t]>` (그 해 활성 노드, 자기 자신 제외) 가 정규화된 조건부 분포가 되게 함.

## 위치

이 패키지는 본 저장소의 `src/dysat/` 에 있다 (예전에는 별도 저장소였다).
경로는 전부 `src/scisoc/config.py` 의 `paths` 에서 온다 — 하드코딩된 절대경로는 없다.

| 무엇 | 어디 |
|---|---|
| 입력 스냅샷 | `data/processed/networks/<source>/adj_<year>.npz` (노트북 01 생성) |
| 최종 임베딩 | `data/processed/embeddings/<source>_E.npz` (노트북 03 이 읽는 위치) |
| 로그·체크포인트 | `results/dysat/DySAT_default/{log,model}/` (추적하지 않음) |

## 실행

`src/dysat/` 에서 실행한다 (`from flags import *`, `from models...` 가 상대 import 이므로).

```bash
pip install "tensorflow-cpu>=2.16" "numpy<2" scipy            # 또는 TF 1.15 + numpy<1.20

cd src/dysat
python train.py --dataset news --years 1990-2023          # 기본값: batch 512, heads 8, dim 128
# 결과: data/processed/embeddings/news_E.npz  →  E [N,34,128], active [N,34], beta, ...
```

`--src` 를 주면 다른 디렉터리의 `adj_<year>.npz` 를 읽는다. 스냅샷 개수는 `--years` 에서 나온다.

진행 상황은 배치마다 stdout 으로 나온다(`--log_every`, 기본 5; `0` 이면 끔). stdout 은 줄 단위로
흘려보내므로 로그 수집기 뒤에서도 `python -u` 없이 바로 보인다. 배치별 손실은 그와 별개로
`results/dysat/<run>/log/` 에도 쌓인다.

```
Building the TF graph (T = 5 snapshots x 8 heads)
  graph built in 3.5s; initializing variables
  epoch   0  batch   5/31  loss 8.1234  6.2s/batch  ~3.2 min/epoch
Epoch   0  train 7.95  val 8.11  (PP_val ~ 3336.7)  beta 1.011  time 190.1s
```

## Attention 변형: GAT / GATv2

`--attn_variant` 로 고른다. 구조 attention의 **점수 함수 한 곳**만 바뀌고 나머지는 전부 같다.

| | 점수 함수 | 출처 |
|---|---|---|
| `gat` (기본) | `e(h_i,h_j) = LeakyReLU( a·[W h_i ‖ W h_j] )` | Veličković et al., ICLR 2018 |
| `gatv2` | `e(h_i,h_j) = a·LeakyReLU( W [h_i ‖ h_j] )` | Brody et al., ICLR 2022 |

GAT는 `a` 를 비선형 **앞**에서 각 변에 따로 적용한다. LeakyReLU가 단조증가라, 이웃 j 의
**순위가 모든 질의 노드 i 에 대해 같다** (static attention, Brody et al. Theorem 1).
GATv2는 비선형을 `a` 와 내적하기 **전**으로 옮겨 두 변이 상호작용하게 하고, 그래서 i 마다
다른 순위를 낼 수 있다 (dynamic attention, Theorem 2).

구현은 논문 식 (7)을 그대로 옮긴 것이다. 외부 라이브러리는 쓰지 않는다:

- `W = [W' ‖ W']` 로 묶었다 (논문 Table 18의 실험 설정). 여기서 `W'` 가 곧 one-hot →
  128차원 임베딩 테이블이라, 묶으면 테이블이 하나로 유지되고 `E` 의 의미가 안 바뀐다.
  따라서 `W [h_i ‖ h_j] = seq_fts[i] + seq_fts[j]`.
- 엣지마다 다시 계산하지 않고 미리 구한 `seq_fts` 행을 gather 한다 (Appendix G.1).
- 파라미터: GAT는 헤드당 `a1, a2` 2 × d′(+bias), GATv2는 `a` 하나 d′(bias 없음 — 참조 구현의
  `(x * att).sum(-1)` 과 동일). 실측 4헤드 d′=8 기준 **72 vs 32**.
- `--share_weights` 로 `W_dst = W_src` 여부를 고른다. 기본값 `False` (PyG `GATv2Conv` 와 동일).
  `True` 는 논문 Table 18 의 실험 설정으로 임베딩 테이블 하나를 양쪽에 쓴다. one-hot 입력이라
  `False` 면 `[N, d′]` 테이블이 하나 더 생기고, 그건 attention 점수로만 학습된다.
- 비용: GAT는 엣지당 스칼라, GATv2는 엣지당 d′차원 벡터를 물질화한다. 엣지 항이 d′배다.
  news(445K 엣지, 5스냅샷, 헤드 8, d′=16) 기준 순전파 ~1.1 GB, 역전파까지 ~3.4 GB.
  paper(200만 엣지)는 ~15 GB라 헤드를 줄이거나 GPU가 필요하다.

`β·log A_uv` 항은 두 변형 모두 동일하게 비선형 **바깥**에 붙는다.

출력이 갈린다:

| variant | 파일 |
|---|---|
| `gat` | `<embeddings>/<dataset>_E.npz` |
| `gatv2` | `<embeddings>/<dataset>_gatv2_E.npz` |

```bash
python train.py --dataset news --years 2019-2023                      # GAT
python train.py --dataset news --years 2019-2023 --attn_variant gatv2  # GATv2
```

노트북에서 읽을 때는 `load_embeddings("news", variant="gatv2")`.

합성 데이터 자체 테스트(노드 60, 커뮤니티 3개 심음, 기준 설정)에서:

| variant | best epoch | best val loss |
|---|---|---|
| gat | 21 | 3.3690 |
| gatv2 `--share_weights=false` (기본) | 22 | 3.3654 |
| gatv2 `--share_weights=true` | 30 | **3.3594** |

self-test (TF 설치와 두 attention 변형 확인용):

```bash
python selftest.py                            # 노드 60, 스냅샷 5 — 기준 케이스, 1분
python selftest.py --nodes 3000 --degree 50   # 실데이터 밀도에 가깝게, epoch당 12~17초
python train.py --dataset synth --src synth --years 1990-1994 --attn_variant gat   [--batch_size ...]
python train.py --dataset synth --src synth --years 1990-1994 --attn_variant gatv2 [--batch_size ...]
```

`selftest.py` 가 그래프를 만들고 나서 그 크기에 맞는 `train.py` 명령을 찍어준다. `--nodes`,
`--snapshots`, `--communities`, `--degree`(목표 평균 degree) 로 크기를 조절한다.

경로 단위 perplexity 는 `notebooks/03_backbone_perplexity.ipynb` 에서 계산한다.

## 파일별 변경

### 유지 · 수정

| 파일 | 변경 |
|---|---|
| `flags.py` | `epochs` 200, `structural_head_config` '8', `temporal_drop` 0.0. 신설 `max_positive`, `val_frac`, `val_pairs_per_step`, `patience`, `beta_init`, `binary_adj`. 삭제 `neg_sample_size`, `neg_weight`, `walk_len`, `test_freq`, `csv_dir`. `log_dir`→`log_subdir` (absl 충돌) |
| `train.py` | **제거:** JSON 플래그 덮어쓰기, `get_context_pairs`, `get_evaluation_data`, 마지막 스냅샷 엣지 덮어쓰기(126–135행), `evaluate_classifier`·csv, `[:, T−2, :]` 슬라이싱. **추가:** 스냅샷별 `split_edges` → 학습 인접행렬 + held-out 쌍, `active` placeholder, 고정 검증 부분집합, 검증 손실 조기종료, best checkpoint, 전체 인접행렬로 최종 `E [N,T,F]` 계산·저장(+ β, temporal attn 평균, 위치 임베딩, 손실 이력) |
| `utils/preprocess.py` | `preprocess_features`: `.todense()` 제거. `sparse_to_tuple`: canonical 정렬. **신설** `adj_with_selfloop_raw` (원가중치 + self-loop=행평균, GCN 정규화 없음), `split_edges`. **삭제** `normalize_graph_gcn`, `get_context_pairs*`, `get_evaluation_data`, `create_data_splits`, random_walk import |
| `utils/minibatch.py` | 컨텍스트 쌍 → `sample_pairs`: 학습 인접행렬 행에서 이웃을 가중치 비례로 `max_positive`개 추출. `active` 마스크 생성. `pairs_feed_dict` (검증용). `max_positive`가 `neg_sample_size`에서 분리 |
| `models/layers.py` | **수정①** 구조 attention: `e_uv = LeakyReLU(f1_u+f2_v) + β·log A_uv` (원본은 `LeakyReLU(A_uv·(f1_u+f2_v))`). `β` 스칼라 변수(층당 1개, `binary_adj`면 0 고정). 인덱스 정렬을 `tf.gather`로 보장. `tf.contrib.*` → `tf.glorot_uniform_initializer`, `tf.linalg.band_part`. `tf.layers.conv1d(kernel=1)` → `_dense1`(동일 연산). `tf.layers.dropout(training=False)`(원본에서 무효였음) → `tf.nn.dropout(rate=)`. `if coef_drop != 0.0` 텐서-불리언 제거 |
| `models/models.py` | **수정②** `_loss`: BCE+음성표본 → 활성 노드 전체 softmax cross-entropy, 자기·비활성 마스크, 스냅샷 평균. `num_time_steps_train = T` (원본 T−1). `fixed_unigram_candidate_sampler` 제거. 생성자 인자 `degrees` → `num_nodes`. `map()`→`list(map())`. `graph_loss_per_t` 노출 |
| `models/inits.py` | import만 (`tf_compat`) |

### 신설

`tf_compat.py`(TF1/2 shim), `selftest.py`(합성 데이터 자체 테스트).

### 2026-09 저장소 통합 시 수정

| 파일 | 변경 |
|---|---|
| `prepare_data.py` | **삭제.** scipy 희소행렬 → networkx → 다시 scipy 로 되돌리는 왕복이었고, `load_graphs` 가 반환하던 networkx 객체는 `train.py` 에서 한 번도 쓰이지 않았다(원본에서 random walk 와 link-prediction 평가가 쓰던 것으로, 둘 다 제거됨). 중간 산출물 `graphs.npz` 와 networkx 의존성도 같이 없어졌다 |
| `utils/preprocess.py` | `load_graphs` 가 `adj_<year>.npz` 를 직접 읽는다. 작업 디렉터리 기준 하드코딩 제거, 연도마다 어휘가 다르면 예외. `load_feats`(`features.npz` 는 생성된 적 없음)와 `dataset_dir` 삭제 |
| `train.py` | `--src`/`--years` 로 스냅샷을 직접 읽고 시간 스텝 수를 거기서 정한다(`--time_steps` 플래그 삭제). 출력 경로를 `paths` 로 통일(`<embeddings>/<source>_E.npz`). 검증 손실이 한 번도 개선되지 않으면 체크포인트가 없어 `saver.restore` 에서 죽던 문제를 경고 후 마지막 파라미터로 내보내도록 처리. 검증 쌍이 0개인 경우를 assert 로 조기 진단 |
| `export_probs.py` | 해당 연도에 활성 노드가 없을 때 `max()` 가 빈 배열에서 죽던 문제, `pairs` 가 비었을 때 `mean()` 이 NaN 경고를 내던 문제 처리. `load_export(source)` 추가 |
| `flags.py` | `save_dir` 제거(출력 위치는 `paths.embeddings` 하나로 고정) |
| `make_synth.py` → `selftest.py` | 합성 데이터도 실제 데이터와 같은 `adj_<year>.npz` 이름을 쓰게 하고, 자체 테스트임이 이름에 드러나게 함 |
| `export_probs.py` | **삭제.** 쌍 단위 `p(u\|v)` 를 꺼내는 코드였으나 어디서도 쓰지 않았다. 경로 단위 perplexity 는 노트북 03 이 계산한다 |
| `models/` | `models/DySAT/` 한 층을 없애고 `models/{inits,layers,models}.py` 로 폄 |
| `models/layers.py` | **수정③** `--attn_variant` 로 GAT / GATv2 점수 함수 선택. 위 절 참조 |
| 경로 | 진입점마다 `sys.path.insert(..., <repo>/src)` 두 줄로 `scisoc.config.paths` 를 불러온다 |

### 삭제

`utils/random_walk.py`, `utils/utilities.py`, `eval/`, `run_script.py`, `train_incremental.py`, `utils/incremental_minibatch.py`, `models/IncSAT/`.

## 원본에서 발견한 것

- `tf.layers.dropout(outputs, rate=...)`은 `training=False` 기본값 → 시간 attention dropout이 **한 번도 적용된 적 없음**. `temporal_drop=0.5`가 무효였음. 기본값 0.0으로 두어 원본 동작 유지, 플래그는 이제 실제 작동.
- 시간 attention 점수 스케일이 `√T`이지 `√(F/h)`가 아님. 원본 그대로 둠.
- `max_positive = neg_sample_size` 커플링. 분리.

## 학습 흐름 (한 스텝)

1. 블록 1~4: T개 스냅샷 전부 → `E [N,T,F]` (배치 무관)
2. 배치 노드 512개 × T년 × `max_positive`쌍에 대해 `logits = E[v,t]·E[:,t]ᵀ` → 마스크 → CE
3. 역전파 (Adam, lr 0.001, clip 1.0)

검증: 스냅샷마다 `val_frac` 엣지를 인코더·샘플링 양쪽에서 제거하고, 그 held-out 쌍에서 같은 손실을 계산. `patience` epoch 동안 개선 없으면 중단. 최종 `E`는 best 파라미터로 **전체 인접행렬**에서 한 번 더 forward.

## 메모리 (N=29,312, T=34, F=128, heads 8, batch 512, paper 200만 엣지)

구조 attention 계수 ~2 GB + `E` 계열 ~1.5 GB + 손실 logits `[512,N]×34×3` ~6 GB ≈ **9.5 GB**. 8 GB GPU면 `--batch_size 256`.

## 미결정 (기본값으로 시작)

`adj_with_selfloop_raw(self_loop='mean')`, `beta_init 1.0`, 백본 path 쌍을 학습 엣지에서 제외할지 여부(현재 미제외).
