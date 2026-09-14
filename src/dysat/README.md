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
| DySAT 입력 | `data/processed/dysat/<source>/graphs.npz` (`prepare_data.py` 생성) |
| 최종 임베딩 | `data/processed/embeddings/<source>_E.npz` (노트북 03 이 읽는 위치) |
| 로그·체크포인트 | `results/dysat/DySAT_default/{log,model}/` (추적하지 않음) |

## 실행

`src/dysat/` 에서 실행한다 (`from flags import *`, `from models.DySAT...` 가 상대 import 이므로).

```bash
pip install "tensorflow-cpu>=2.16" "numpy<2" scipy networkx     # 또는 TF 1.15 + numpy<1.20

cd src/dysat
python prepare_data.py --source news --years 1990 2023
python train.py --dataset news --time_steps 34            # 기본값: batch 512, heads 8, dim 128
# 결과: data/processed/embeddings/news_E.npz  →  E [N,34,128], active [N,34], beta, ...
```

self-test (TF 설치 확인용, 1분):

```bash
python make_synth.py
python prepare_data.py --source synth --src synth --years 1990 1994
python train.py --dataset synth --time_steps 5 --epochs 60 --batch_size 20 --patience 10 \
    --structural_head_config 4 --structural_layer_config 32 \
    --temporal_head_config 4 --temporal_layer_config 32
```

확률/perplexity: `export_probs.py` 의 `load_export(source)`, `cond_prob(E, active, v, t)`,
`perplexity(E, active, pairs, t)`.

## 파일별 변경

### 유지 · 수정

| 파일 | 변경 |
|---|---|
| `flags.py` | `time_steps` 34, `epochs` 200, `structural_head_config` '8', `temporal_drop` 0.0. 신설 `max_positive`, `val_frac`, `val_pairs_per_step`, `patience`, `beta_init`, `binary_adj`. 삭제 `neg_sample_size`, `neg_weight`, `walk_len`, `test_freq`, `csv_dir`. `log_dir`→`log_subdir` (absl 충돌) |
| `train.py` | **제거:** JSON 플래그 덮어쓰기, `get_context_pairs`, `get_evaluation_data`, 마지막 스냅샷 엣지 덮어쓰기(126–135행), `evaluate_classifier`·csv, `[:, T−2, :]` 슬라이싱. **추가:** 스냅샷별 `split_edges` → 학습 인접행렬 + held-out 쌍, `active` placeholder, 고정 검증 부분집합, 검증 손실 조기종료, best checkpoint, 전체 인접행렬로 최종 `E [N,T,F]` 계산·저장(+ β, temporal attn 평균, 위치 임베딩, 손실 이력) |
| `utils/preprocess.py` | `preprocess_features`: `.todense()` 제거. `sparse_to_tuple`: canonical 정렬. **신설** `adj_with_selfloop_raw` (원가중치 + self-loop=행평균, GCN 정규화 없음), `split_edges`. **삭제** `normalize_graph_gcn`, `get_context_pairs*`, `get_evaluation_data`, `create_data_splits`, random_walk import |
| `utils/minibatch.py` | 컨텍스트 쌍 → `sample_pairs`: 학습 인접행렬 행에서 이웃을 가중치 비례로 `max_positive`개 추출. `active` 마스크 생성. `pairs_feed_dict` (검증용). `max_positive`가 `neg_sample_size`에서 분리 |
| `models/DySAT/layers.py` | **수정①** 구조 attention: `e_uv = LeakyReLU(f1_u+f2_v) + β·log A_uv` (원본은 `LeakyReLU(A_uv·(f1_u+f2_v))`). `β` 스칼라 변수(층당 1개, `binary_adj`면 0 고정). 인덱스 정렬을 `tf.gather`로 보장. `tf.contrib.*` → `tf.glorot_uniform_initializer`, `tf.linalg.band_part`. `tf.layers.conv1d(kernel=1)` → `_dense1`(동일 연산). `tf.layers.dropout(training=False)`(원본에서 무효였음) → `tf.nn.dropout(rate=)`. `if coef_drop != 0.0` 텐서-불리언 제거 |
| `models/DySAT/models.py` | **수정②** `_loss`: BCE+음성표본 → 활성 노드 전체 softmax cross-entropy, 자기·비활성 마스크, 스냅샷 평균. `num_time_steps_train = T` (원본 T−1). `fixed_unigram_candidate_sampler` 제거. 생성자 인자 `degrees` → `num_nodes`. `map()`→`list(map())`. `graph_loss_per_t` 노출 |
| `models/DySAT/inits.py` | import만 (`tf_compat`) |

### 신설

`tf_compat.py`(TF1/2 shim), `_repo.py`(저장소 `src/` 를 `sys.path` 에 올려 `scisoc.config.paths` 사용),
`prepare_data.py`(adj_<year>.npz → graphs.npz), `export_probs.py`, `make_synth.py`.

### 2026-09 저장소 통합 시 수정

| 파일 | 변경 |
|---|---|
| `prepare_data.py` | 입력 파일명이 `{source}_{year}_adj_f.npz` 로 되어 있어 노트북 01 의 실제 산출물(`networks/<source>/adj_<year>.npz`)과 맞지 않았다. 실제 레이아웃으로 맞추고 `--src`/`--out` 기본값을 `paths` 에서 가져오게 함 |
| `utils/preprocess.py` | `load_graphs` 가 프로세스 작업 디렉터리 기준 `data/<dataset>/` 를 하드코딩했다. `paths.dysat_input` 기준으로 변경, 파일이 없을 때 재현 명령을 알려주는 예외 추가. `nx.adjacency_matrix` 결과를 `csr_matrix` 로 명시 변환(networkx 3.x 는 sparse *array* 를 반환) |
| `train.py` | 출력 경로를 `paths` 로 통일(`<embeddings>/<source>_E.npz`). 검증 손실이 한 번도 개선되지 않으면 체크포인트가 없어 `saver.restore` 에서 죽던 문제를 경고 후 마지막 파라미터로 내보내도록 처리. 검증 쌍이 0개인 경우를 assert 로 조기 진단 |
| `export_probs.py` | 해당 연도에 활성 노드가 없을 때 `max()` 가 빈 배열에서 죽던 문제, `pairs` 가 비었을 때 `mean()` 이 NaN 경고를 내던 문제 처리. `load_export(source)` 추가 |
| `flags.py` | `save_dir` 제거(출력 위치는 `paths.embeddings` 하나로 고정) |
| `make_synth.py` | 합성 데이터도 실제 데이터와 같은 `adj_<year>.npz` 이름을 쓰게 함 |

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
