import os
import json
import argparse
from collections import defaultdict

import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast


SPECIAL_TOKENS = {"[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"}


def load_tokenizer(tokenizer_json: str) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def iter_corpus(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line.split()


@torch.inference_mode()
def collect_predictions(
    model,
    tokenizer,
    corpus_path,
    target_token,
    distance,
    topk,
    max_contexts,
    device,
):
    scores = defaultdict(list)
    used = 0

    mask_token = tokenizer.mask_token
    mask_id = tokenizer.mask_token_id

    for tokens in iter_corpus(corpus_path):
        for i, tok in enumerate(tokens):
            if tok != target_token:
                continue

            pos = i + distance
            if pos < 0 or pos >= len(tokens):
                continue

            masked = tokens.copy()
            masked[pos] = mask_token

            inputs = tokenizer(" ".join(masked), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            logits = model(**inputs).logits
            mask_idx = (inputs["input_ids"][0] == mask_id).nonzero(as_tuple=False)[0].item()

            probs = torch.softmax(logits[0, mask_idx], dim=-1)
            top = torch.topk(probs, topk)

            for idx, p in zip(top.indices.tolist(), top.values.tolist()):
                tok = tokenizer.convert_ids_to_tokens(idx)
                if tok in SPECIAL_TOKENS:
                    continue
                scores[tok].append(float(p))

            used += 1
            if used >= max_contexts:
                break

        if used >= max_contexts:
            break

    ranked = sorted(
        ((t, sum(ps) / len(ps)) for t, ps in scores.items()),
        key=lambda x: -x[1]
    )

    return ranked


def build_ckpt(ckpt_root, domain, year, stage):
    return os.path.join(ckpt_root, domain, f"year={year}", stage)


def build_corpus(corpus_root, domain, year):
    return os.path.join(corpus_root, f"{domain}_{year}.txt")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--domains", nargs="+", choices=["news", "paper"], default=["news", "paper"])

    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--stage", default="stage2")
    ap.add_argument("--corpus_root", required=True)
    ap.add_argument("--tokenizer_json", required=True)

    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--distance", type=int, required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--max_contexts", type=int, default=1000)

    ap.add_argument("--out_root", required=True)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.tokenizer_json)

    for year in args.years:
        for target in args.targets:
            rows = []

            for domain in args.domains:
                ckpt = build_ckpt(args.ckpt_root, domain, year, args.stage)
                corpus = build_corpus(args.corpus_root, domain, year)

                if not os.path.isdir(ckpt) or not os.path.isfile(corpus):
                    print(f"[SKIP] {domain} {year} missing data")
                    continue

                model = BertForMaskedLM.from_pretrained(ckpt).to(device)
                model.eval()

                ranked = collect_predictions(
                    model=model,
                    tokenizer=tokenizer,
                    corpus_path=corpus,
                    target_token=target,
                    distance=args.distance,
                    topk=args.topk,
                    max_contexts=args.max_contexts,
                    device=device,
                )

                for rank, (tok, prob) in enumerate(ranked[:args.topk], start=1):
                    rows.append({
                        "rank": rank,
                        f"{domain}_neighbor": tok,
                        f"{domain}_avg_prob": prob,
                    })

            if not rows:
                continue

            # ---- merge news/paper by rank ----
            merged = {}
            for r in rows:
                k = r["rank"]
                merged.setdefault(k, {"rank": k})
                merged[k].update(r)

            table = [merged[k] for k in sorted(merged)]

            # ---- save TSV ----
            out_dir = os.path.join(
                args.out_root,
                f"dist={args.distance}",
                f"year={year}"
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
