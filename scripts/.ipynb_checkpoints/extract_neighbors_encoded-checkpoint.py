import os, json, argparse, random
from collections import defaultdict

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


def iter_token_lists(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield ln.split()  # 코퍼스는 "노드 토큰"을 공백으로 구분


@torch.no_grad()
def predict_mask_topk_from_tokens(model, tokenizer, tokens, mask_index, topk, device):
    """
    핵심 변경:
    - 문자열로 join해서 다시 토크나이즈하지 않고,
    - 이미 노드 토큰으로 split된 tokens를 tokenizer에 직접 투입한다.
    - fast tokenizer는 `is_split_into_words=True`로 받으면 토큰 경계를 유지하려고 한다.
    """
    toks = tokens.copy()
    toks[mask_index] = tokenizer.mask_token

    enc = tokenizer(
        toks,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    pos = (input_ids[0] == mask_id).nonzero(as_tuple=False)
    if pos.numel() == 0:
        return None

    pos = pos[0].item()
    logits = model(input_ids=input_ids, attention_mask=attn).logits
    probs = torch.softmax(logits[0, pos], dim=-1)
    vals, idxs = torch.topk(probs, k=topk)
    return idxs.cpu().tolist(), vals.cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_json", required=True)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--corpus_path", required=True)
    ap.add_argument("--target_token", required=True)
    ap.add_argument("--side", choices=["left", "right", "both"], default="both")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--num_contexts", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)

    tokenizer = load_tokenizer(args.tokenizer_json)
    model = BertForMaskedLM.from_pretrained(args.checkpoint_dir).to(device)
    model.eval()

    # target 포함 문장 샘플링
    candidates = []
    for toks in iter_token_lists(args.corpus_path):
        if args.target_token in toks:
            candidates.append(toks)

    rng.shuffle(candidates)
    candidates = candidates[: min(args.num_contexts, len(candidates))]

    agg_left = defaultdict(float)
    agg_right = defaultdict(float)
    used_left = 0
    used_right = 0

    for toks in candidates:
        i = toks.index(args.target_token)

        if args.side in ("left", "both") and i - 1 >= 0:
            res = predict_mask_topk_from_tokens(model, tokenizer, toks, i - 1, args.topk, device)
            if res is not None:
                idxs, vals = res
                for tid, p in zip(idxs, vals):
                    agg_left[tid] += float(p)
                used_left += 1

        if args.side in ("right", "both") and i + 1 < len(toks):
            res = predict_mask_topk_from_tokens(model, tokenizer, toks, i + 1, args.topk, device)
            if res is not None:
                idxs, vals = res
                for tid, p in zip(idxs, vals):
                    agg_right[tid] += float(p)
                used_right += 1

        
        SPECIAL_TOKENS = {
        tokenizer.pad_token,
        tokenizer.unk_token,
        tokenizer.cls_token,
        tokenizer.sep_token,
        tokenizer.mask_token,
    }
    
    def finalize(agg, used):
        items = sorted(agg.items(), key=lambda x: x[1], reverse=True)
        out = []
        for tid, s in items:
            tok = tokenizer.convert_ids_to_tokens(tid)
            if tok in SPECIAL_TOKENS:
                continue
            out.append({"token": tok, "avg_prob": s / max(1, used)})
            if len(out) >= args.topk:
                break
        return out
    
    # def finalize(agg, used):
    #     items = sorted(agg.items(), key=lambda x: x[1], reverse=True)[: args.topk]
    #     return [{"token": tokenizer.convert_ids_to_tokens(tid), "avg_prob": s / max(1, used)} for tid, s in items]

    out = {
        "target_token": args.target_token,
        "checkpoint_dir": args.checkpoint_dir,
        "corpus_path": args.corpus_path,
        "side": args.side,
        "num_contexts_requested": args.num_contexts,
        "used_left": used_left,
        "used_right": used_right,
        "topk_left": finalize(agg_left, used_left),
        "topk_right": finalize(agg_right, used_right),
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved -> {args.out_json}")
    print(f"used_left={used_left}, used_right={used_right}")


if __name__ == "__main__":
    main()
