import os
import argparse
import json

def token_set_from_corpus(path: str, min_count: int = 1):
    counts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            for tok in line.strip().split():
                if not tok:
                    continue
                counts[tok] = counts.get(tok, 0) + 1
    # 너무 희귀한 토큰 제외(옵션)
    return {t for t, c in counts.items() if c >= min_count}, counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_root", required=True)
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--min_count", type=int, default=5, help="filter tokens with count < min_count in each domain")
    ap.add_argument("--out_dir", default="./outputs/common_targets")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for y in args.years:
        news_path = os.path.join(args.corpus_root, f"news_{y}.txt")
        paper_path = os.path.join(args.corpus_root, f"paper_{y}.txt")

        if not os.path.isfile(news_path) or not os.path.isfile(paper_path):
            print(f"[SKIP] missing year={y}: {news_path} or {paper_path}")
            continue

        news_set, news_counts = token_set_from_corpus(news_path, args.min_count)
        paper_set, paper_counts = token_set_from_corpus(paper_path, args.min_count)

        common = sorted(news_set & paper_set)

        out = {
            "year": y,
            "min_count": args.min_count,
            "n_news_tokens": len(news_set),
            "n_paper_tokens": len(paper_set),
            "n_common": len(common),
            "common_tokens": common,
        }

        out_path = os.path.join(args.out_dir, f"common_targets_{y}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        # also save txt for easy --targets paste
        out_txt = os.path.join(args.out_dir, f"common_targets_{y}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            for t in common:
                f.write(t + "\n")

        print(f"[OK] year={y} common={len(common)} saved: {out_txt}")

if __name__ == "__main__":
    main()
