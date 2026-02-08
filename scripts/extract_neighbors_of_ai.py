import os
import json
import argparse
import random
import torch
from collections import defaultdict
from transformers import PreTrainedTokenizerFast, BertForMaskedLM


def load_tokenizer(tokenizer_json: str):
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def iter_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield ln


def find_ai_occurrences(tokens, target):
    return [i for i, t in enumerate(tokens) if t == target]


def mask_neighbor(tokens, ai_idx, direction, mask_token):
    """
    direction: 'left' or 'right'
    returns (masked_sentence, masked_token_original) or (None, None)
    """
    if direction == "left":
        j = ai_idx - 1
    else:
        j = ai_idx + 1

    if j < 0 or j >= len(tokens):
        return None, None

    original = tokens[j]
    masked = tokens.copy()
    masked[j] = mask_token
    return " ".join(masked), original


@torch.no_grad()
def topk_for_mask(model, tokenizer, sent: str, topk: int, device: str):
    enc = tokenizer(sent, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    mask_pos = (input_ids[0] == mask_id).nonzero(as_tuple=False)
    if mask_pos.numel() == 0:
        return None

    outputs = model(input_ids=input_ids, attention_mask=attn)
    logits = outputs.logits
    pos = mask_pos[0].item()
    probs = torch.softmax(logits[0, pos], dim=-1)

    vals, idxs = torch.topk(probs, k=topk)
    return idxs.cpu().tolist(), vals.cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_json", required=True)
    ap.add_argument("--checkpoint_dir", required=True)      # e.g., ./checkpoints/annual_node/news/year=1995/stage2
    ap.add_argument("--corpus_path", required=True)         # e.g., ./corpus/news_1995.txt
    ap.add_argument("--target", required=True)              # "artificial intelligence"
    ap.add_argument("--direction", choices=["left", "right", "both"], default="both")
    ap.add_argument("--num_contexts", type=int, default=500)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)

    tokenizer = load_tokenizer(args.tokenizer_json)
    model = BertForMaskedLM.from_pretrained(args.checkpoint_dir).to(device)
    model.eval()

    # 1) corpus에서 target 포함 문장 샘플링
    candidates = []
    for ln in iter_lines(args.corpus_path):
        toks = ln.split()
        if args.target in toks:
            candidates.append(ln)

    rng.shuffle(candidates)
    candidates = candidates[: min(args.num_contexts, len(candidates))]

    # 2) AI의 좌/우 이웃을 [MASK]로 바꾸고 예측 분포 누적
    agg_left = defaultdict(float)
    agg_right = defaultdict(float)
    used_left = 0
    used_right = 0

    for ln in candidates:
        toks = ln.split()
        ai_idxs = find_ai_occurrences(toks, args.target)
        if not ai_idxs:
            continue

        # 한 문장에 여러 번 나오면 첫 번째 occurrence만 사용(원하면 변경 가능)
        ai_idx = ai_idxs[0]

        if args.direction in ("left", "both"):
            masked_sent, original_neighbor = mask_neighbor(toks, ai_idx, "left", tokenizer.mask_token)
            if masked_sent is not None:
                res = topk_for_mask(model, tokenizer, masked_sent, args.topk, device)
                if res is not None:
                    idxs, vals = res
                    for tid, p in zip(idxs, vals):
                        agg_left[tid] += float(p)
                    used_left += 1

        if args.direction in ("right", "both"):
            masked_sent, original_neighbor = mask_neighbor(toks, ai_idx, "right", tokenizer.mask_token)
            if masked_sent is not None:
                res = topk_for_mask(model, tokenizer, masked_sent, args.topk, device)
                if res is not None:
                    idxs, vals = res
                    for tid, p in zip(idxs, vals):
                        agg_right[tid] += float(p)
                    used_right += 1

    def make_topk(agg, used):
        items = sorted(agg.items(), key=lambda x: x[1], reverse=True)[: args.topk]
        return [{"token": tokenizer.convert_ids_to_tokens(tid), "avg_prob": s / max(1, used)} for tid, s in items]

    out = {
        "target": args.target,
        "checkpoint_dir": args.checkpoint_dir,
        "corpus_path": args.corpus_path,
        "direction": args.direction,
        "num_contexts_requested": args.num_contexts,
        "num_contexts_used_left": used_left,
        "num_contexts_used_right": used_right,
        "topk_left": make_topk(agg_left, used_left),
        "topk_right": make_topk(agg_right, used_right),
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved -> {args.out_json}")
    print(f"used_left={used_left}, used_right={used_right}")

    # 콘솔에 top10도 같이 출력
    if args.direction in ("left", "both"):
        print("\n[TOP10 LEFT neighbor predictions]")
        for r in out["topk_left"][:10]:
            print(r["token"], r["avg_prob"])
    if args.direction in ("right", "both"):
        print("\n[TOP10 RIGHT neighbor predictions]")
        for r in out["topk_right"][:10]:
            print(r["token"], r["avg_prob"])


if __name__ == "__main__":
    main()
