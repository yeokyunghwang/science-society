import os
import json
import argparse
from collections import defaultdict

SPECIAL = {"[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"}

def load_one(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize(dist):
    s = sum(dist.values())
    if s <= 0:
        return {}
    return {k: v/s for k, v in dist.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--domains", nargs="+", choices=["news", "paper"], default=["news", "paper"])
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--in_root", required=True)   # ./outputs/neighbor_mask
    ap.add_argument("--out_root", required=True)  # ./outputs/family_aggregate
    ap.add_argument("--side", choices=["left", "right", "both"], default="both")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--weighting", choices=["mean", "contexts"], default="contexts",
                    help="mean=equal weight across targets, contexts=weight by used_left/right")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    sides = ["left", "right"] if args.side == "both" else [args.side]

    for domain in args.domains:
        for y in args.years:
            for side in sides:
                agg = defaultdict(float)
                total_w = 0.0
                per_target_meta = []

                for t in args.targets:
                    fp = os.path.join(args.in_root, f"{domain}_{y}_{t}_neighbors.json")
                    if not os.path.isfile(fp):
                        per_target_meta.append({"target": t, "missing": True})
                        continue

                    obj = load_one(fp)
                    used = obj["used_left"] if side == "left" else obj["used_right"]
                    per_target_meta.append({"target": t, "missing": False, "used": used})

                    # 이 타깃에서 추출된 분포
                    key = "topk_left" if side == "left" else "topk_right"
                    dist = {r["token"]: float(r["avg_prob"]) for r in obj.get(key, [])}
                    # special token 제거
                    dist = {k:v for k,v in dist.items() if k not in SPECIAL}
                    # dist는 top-k만 있으니 그대로 합치되, 나중에 normalize
                    if not dist:
                        continue

                    if args.weighting == "mean":
                        w = 1.0
                    else:
                        w = float(used)

                    if w <= 0:
                        continue

                    for tok, p in dist.items():
                        agg[tok] += w * p
                    total_w += w

                # normalize
                agg = dict(agg)
                if total_w <= 0 or not agg:
                    out = {
                        "domain": domain, "year": y, "side": side,
                        "targets": args.targets, "weighting": args.weighting,
                        "note": "empty_aggregate",
                        "per_target": per_target_meta,
                        "topk": []
                    }
                else:
                    # total_w로 스케일 (mean이면 그냥 평균, contexts면 컨텍스트 가중 평균)
                    for tok in list(agg.keys()):
                        agg[tok] = agg[tok] / total_w

                    # 확률 다시 정규화(선택이지만 추천: top-k truncated여서 합이 1이 아닐 수 있음)
                    agg = normalize(agg)

                    items = sorted(agg.items(), key=lambda x: x[1], reverse=True)[: args.topk]
                    out = {
                        "domain": domain, "year": y, "side": side,
                        "targets": args.targets, "weighting": args.weighting,
                        "per_target": per_target_meta,
                        "topk": [{"token": tok, "prob": p} for tok, p in items]
                    }

                out_path = os.path.join(args.out_root, f"{domain}_{y}_AI_FAMILY_{side}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                print("[OK]", out_path)

if __name__ == "__main__":
    main()
