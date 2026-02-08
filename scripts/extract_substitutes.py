import os
import json
import argparse
import random
import torch
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


def sample_contexts(corpus_path: str, target: str, n: int, seed: int):
    rng = random.Random(seed)
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # target이 포함된 문장만
    cand = [ln for ln in lines if f" {target} " in f" {ln} "]
    if len(cand) == 0:
        return []

    rng.shuffle(cand)
    return cand[: min(n, len(cand))]


def mask_first_occurrence(text: str, target: str, mask_token: str):
    parts = text.split()
    for i, tok in enumerate(parts):
        if tok == target:
            parts[i] = mask_token
            return " ".join(parts), i
    return None, None


@torch.no_grad()
def topk_for_masked_sentence(model, tokenizer, sent: str, topk: int, device: str):
    enc = tokenizer(sent, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    # mask position 찾기
    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    mask_pos = (input_ids[0] == mask_id).nonzero(as_tuple=False)
    if mask_pos.numel() == 0:
        return None

    outputs = model(input_ids=input_ids, attention_mask=attn)
    logits = outputs.logits  # (1, seq, vocab)
    pos = mask_pos[0].item()
    probs = torch.softmax(logits[0, pos], dim=-1)

    vals, idxs = torch.topk(probs, k=topk)
    return idxs.cpu().tolist(), vals.cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_json", type=str, required=True)
    ap.add_argument("--checkpoint_dir", type=str, required=True, help="e.g., ./checkpoints/annual/news/year=1991/stage2")
    ap.add_argument("--corpus_path", type=str, required=True, help="e.g., ./corpus/news_1991.txt")
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--num_contexts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer(args.tokenizer_json)
    model = BertForMaskedLM.from_pretrained(args.checkpoint_dir).to(device)
    model.eval()

    contexts = sample_contexts(args.corpus_path, args.target, args.num_contexts, args.seed)
    if len(contexts) == 0:
        print(f"[WARN] no contexts found for target={args.target} in {args.corpus_path}")
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"target": args.target, "n_contexts": 0, "topk": []}, f, ensure_ascii=False, indent=2)
        return

    # aggregate: token_id -> sum_prob
    agg = {}

    used = 0
    for ln in contexts:
        masked, _ = mask_first_occurrence(ln, args.target, tokenizer.mask_token)
        if masked is None:
            continue
        res = topk_for_masked_sentence(model, tokenizer, masked, args.topk, device)
        if res is None:
            continue
        idxs, vals = res
        for tid, p in zip(idxs, vals):
            agg[tid] = agg.get(tid, 0.0) + float(p)
        used += 1

    # 평균 확률
    items = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    top = items[: args.topk]

    out = {
        "target": args.target,
        "checkpoint_dir": args.checkpoint_dir,
        "corpus_path": args.corpus_path,
        "n_contexts_requested": args.num_contexts,
        "n_contexts_used": used,
        "topk": [
            {"token": tokenizer.convert_ids_to_tokens(tid), "avg_prob": s / max(1, used)}
            for tid, s in top
        ],
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved substitutes -> {args.out_json} (used={used})")


if __name__ == "__main__":
    main()
