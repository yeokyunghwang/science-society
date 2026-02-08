# predict_neighbors_by_distance.py

import os
import argparse
from collections import defaultdict

import torch
from transformers import BertForMaskedLM, AutoTokenizer

SPECIAL_TOKENS = {"[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"}


def iter_corpus(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line.split()


def read_node_vocab(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def build_node_token_ids(tokenizer, node_tokens):
    ids = set()
    missing = 0
    for t in node_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid is None or tid == tokenizer.unk_token_id:
            missing += 1
            continue
        ids.add(int(tid))
    if missing:
        print(f"[WARN] {missing} node tokens mapped to UNK (ignored). e.g. {node_tokens[:5]}")
    return ids


@torch.inference_mode()
def collect_predictions_node_only(
    model,
    tokenizer,
    corpus_path,
    target_token,
    distance,
    topk,
    max_contexts,
    device,
    node_token_ids,  # set[int]
    debug: bool = False,
):
    """
    ✅ 핵심 변경점:
    - " ".join(...) -> tokenizer(...) 로 재토큰화하지 않음
    - 코퍼스가 이미 토큰 리스트이므로, convert_tokens_to_ids로 "lookup"만 수행
    - 따라서 pos(코퍼스 인덱스) == mask_idx(모델 인덱스에서 [CLS] 보정 포함) 보장
    - 후보는 node_token_ids로만 제한하여(node-only softmax/topk) BERT 기본 토큰 배제
    """
    scores = defaultdict(list)
    used = 0

    mask_token = tokenizer.mask_token
    mask_id = tokenizer.mask_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    unk_id = tokenizer.unk_token_id

    if cls_id is None or sep_id is None or mask_id is None:
        raise ValueError("Tokenizer must have [CLS]/[SEP]/[MASK] token ids.")

    # node 후보 id 텐서 (node vocab 내부 softmax/topk용)
    node_ids = torch.tensor(sorted(list(node_token_ids)), dtype=torch.long, device=device)

    for tokens in iter_corpus(corpus_path):
        for i, tok in enumerate(tokens):
            if tok != target_token:
                continue

            pos = i + distance
            if pos < 0 or pos >= len(tokens):
                continue

            # --- (A) 코퍼스 토큰 기준으로 마스킹 ---
            masked = tokens.copy()
            masked[pos] = mask_token

            # --- (B) "재토큰화" 없이, 토큰 -> id lookup만 수행 ---
            ids = tokenizer.convert_tokens_to_ids(masked)
            # convert_tokens_to_ids가 list를 받을 때 None이 섞일 수 있으니 방어
            ids = [unk_id if (x is None) else int(x) for x in ids]

            # --- (C) 모델 입력 구성: [CLS] + ids + [SEP] ---
            input_ids = torch.tensor([[cls_id] + ids + [sep_id]], device=device, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)

            # pos는 코퍼스 인덱스(0-based), 모델 입력에서는 [CLS] 때문에 +1
            mask_idx = pos + 1

            # sanity: 우리가 의도한 위치가 진짜 [MASK]인지 확인(디버그용)
            if debug and used == 0:
                mid = int(input_ids[0, mask_idx].item())
                print(f"[DEBUG-ONE] target={target_token} distance={distance} len(tokens)={len(tokens)} i={i} pos={pos}")
                print(f"[DEBUG-ONE] mask_idx={mask_idx} input_id@mask_idx={mid} (mask_id={mask_id})")
                if mid != mask_id:
                    # 이 경우는 거의 항상 tokenizer가 mask_token을 진짜 mask_id로 매핑 못 했다는 뜻
                    print("[DEBUG-ONE][WARN] input_id at mask_idx is not [MASK]! Check mask_token string and tokenizer setup.")

            if debug and used < 3:
                unk_cnt = sum(1 for x in ids if x == unk_id)
                print(f"[DEBUG-CTX] used={used} len(ids)={len(ids)} unk_cnt={unk_cnt} unk_rate={unk_cnt/len(ids):.3f}")
                print("[DEBUG-CTX] sample tokens:", masked[:10])
                print("[DEBUG-CTX] sample ids   :", ids[:10])
                print("[DEBUG-CTX] target_token_id:", tokenizer.convert_tokens_to_ids(target_token))

            # --- (D) forward ---
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [1, seq, vocab]

            # --- (E) node vocab 로짓만 추출 -> node vocab 내부 확률 ---
            node_logits = logits[0, mask_idx].index_select(0, node_ids)  # [n_nodes]
            node_probs = torch.softmax(node_logits, dim=-1)

            k = min(int(topk), int(node_probs.numel()))
            top = torch.topk(node_probs, k)

            top_node_ids = node_ids[top.indices].tolist()
            top_node_ps = top.values.tolist()

            for tid, p in zip(top_node_ids, top_node_ps):
                t = tokenizer.convert_ids_to_tokens(int(tid))
                if t is None:
                    continue
                if t in SPECIAL_TOKENS:
                    continue
                scores[t].append(float(p))

            used += 1
            if used >= max_contexts:
                break

        if used >= max_contexts:
            break

    ranked = sorted(
        ((t, sum(ps) / len(ps)) for t, ps in scores.items()),
        key=lambda x: -x[1]
    )

    if debug:
        print(f"[DEBUG] used={used} unique_scores={len(scores)} ranked_len={len(ranked)} topk={topk}")

    return ranked


def build_ckpt(ckpt_root, domain, year):
    return os.path.join(ckpt_root, f"{domain}_annual", str(year))


def build_corpus(corpus_root, domain, year):
    return os.path.join(corpus_root, f"{domain}_{year}.txt")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--domains", nargs="+", choices=["news", "paper"], default=["news", "paper"])

    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--corpus_root", required=True)

    ap.add_argument("--node_vocab_txt", required=True,
                    help="Node vocabulary (one token per line), e.g. concept_vocab_encoded.txt")

    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--distance", type=int, required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--max_contexts", type=int, default=1000)

    ap.add_argument("--out_root", required=True)
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    node_tokens = read_node_vocab(args.node_vocab_txt)
    print(f"[INFO] loaded node vocab: {len(node_tokens)}")

    for year in args.years:
        for target in args.targets:
            rows = []

            for domain in args.domains:
                ckpt = build_ckpt(args.ckpt_root, domain, year)
                corpus = build_corpus(args.corpus_root, domain, year)

                if not os.path.isdir(ckpt) or not os.path.isfile(corpus):
                    print(f"[SKIP] {domain} {year} missing data")
                    continue

                # ✅ tokenizer/model은 ckpt에서 로드 (id 체계 동일)
                tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
                model = BertForMaskedLM.from_pretrained(ckpt).to(device)
                model.eval()

                node_token_ids = build_node_token_ids(tokenizer, node_tokens)
                if not node_token_ids:
                    print(f"[SKIP] {domain} {year} node_token_ids empty (tokenizer mismatch?)")
                    continue

                ranked = collect_predictions_node_only(
                    model=model,
                    tokenizer=tokenizer,
                    corpus_path=corpus,
                    target_token=target,
                    distance=args.distance,
                    topk=args.topk,
                    max_contexts=args.max_contexts,
                    device=device,
                    node_token_ids=node_token_ids,
                    debug=args.debug,
                )

                for rank, (tok, prob) in enumerate(ranked[:args.topk], start=1):
                    rows.append({
                        "rank": rank,
                        f"{domain}_neighbor": tok,
                        f"{domain}_avg_prob": prob,
                    })

            if not rows:
                continue

            # ---- merge by rank ----
            merged = {}
            for r in rows:
                k = r["rank"]
                merged.setdefault(k, {"rank": k})
                merged[k].update(r)

            table = [merged[k] for k in sorted(merged)]

            out_dir = os.path.join(
                args.out_root,
                f"dist={args.distance}",
                f"year={year}",
            )
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, f"target={target}.tsv")

            cols = ["rank"]
            for d in args.domains:
                cols += [f"{d}_neighbor", f"{d}_avg_prob"]

            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\t".join(cols) + "\n")
                for r in table:
                    f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

            print(f"[OK] saved {out_path}")


if __name__ == "__main__":
    main()
