import os
import argparse
from datasets import Dataset, DatasetDict

def read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_root", type=str, default="./corpus")
    ap.add_argument("--out_dir", type=str, default="./datasets")
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--domains", type=str, nargs="+", default=["news", "paper"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for y in args.years:
        for d in args.domains:
            in_path = os.path.join(args.corpus_root, f"{d}_{y}.txt")
            if not os.path.exists(in_path):
                raise FileNotFoundError(in_path)

            ds = Dataset.from_dict({"text": read_lines(in_path)})
            out_path = os.path.join(args.out_dir, f"{d}_{y}")
            ds.save_to_disk(out_path)
            print(f"[OK] saved {d}_{y} -> {out_path} (n={len(ds)})")

if __name__ == "__main__":
    main()
