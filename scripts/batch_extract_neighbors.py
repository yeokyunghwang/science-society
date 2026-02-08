import os
import argparse
import subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--domains", nargs="+", choices=["news", "paper"], default=["news", "paper"])
    ap.add_argument("--targets", nargs="+", required=True,
                    help="List of target tokens (encoded).")
    ap.add_argument("--tokenizer_json", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--corpus_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--num_contexts", type=int, default=1000)
    ap.add_argument("--side", choices=["left", "right", "both"], default="both")
    ap.add_argument("--stage", choices=["stage1", "stage2"], default="stage2")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    ran, skipped, failed = 0, 0, 0

    for domain in args.domains:
        for y in args.years:
            ckpt_dir = os.path.join(args.ckpt_root, domain, f"year={y}", args.stage)
            corpus_path = os.path.join(args.corpus_root, f"{domain}_{y}.txt")

            if not os.path.isdir(ckpt_dir):
                print(f"[SKIP] missing checkpoint: {ckpt_dir}")
                skipped += 1
                continue
            if not os.path.isfile(corpus_path):
                print(f"[SKIP] missing corpus: {corpus_path}")
                skipped += 1
                continue

            for target in args.targets:
                out_json = os.path.join(
                    args.out_root,
                    f"{domain}_{y}_{target}_neighbors.json"
                )

                cmd = [
                    "python", "-u", "scripts/extract_neighbors_encoded.py",
                    "--tokenizer_json", args.tokenizer_json,
                    "--checkpoint_dir", ckpt_dir,
                    "--corpus_path", corpus_path,
                    "--target_token", target,
                    "--side", args.side,
                    "--topk", str(args.topk),
                    "--num_contexts", str(args.num_contexts),
                    "--out_json", out_json,
                ]

                print("[RUN]", " ".join(cmd))
                try:
                    subprocess.run(cmd, check=True)
                    ran += 1
                except subprocess.CalledProcessError as e:
                    print(f"[FAIL] domain={domain} year={y} target={target}")
                    failed += 1

    print(f"[DONE] ran={ran}, skipped={skipped}, failed={failed}")
    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
