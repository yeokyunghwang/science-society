import os, json, csv, math, argparse

def load_obj(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_dist(path, side):
    obj = load_obj(path)
    key = "topk_left" if side == "left" else "topk_right"
    d = {row["token"]: float(row["avg_prob"]) for row in obj.get(key, [])}
    s = sum(d.values())
    if s <= 0:
        return {}
    return {k: v/s for k, v in d.items()}

def kl(p, q, eps=1e-12):
    keys = set(p) | set(q)
    out = 0.0
    for k in keys:
        pk = p.get(k, 0.0)
        if pk <= 0:
            continue
        qk = q.get(k, 0.0)
        out += pk * math.log((pk + eps) / (qk + eps))
    return out

def js(p, q, eps=1e-12):
    keys = set(p) | set(q)
    m = {k: 0.5*(p.get(k,0.0)+q.get(k,0.0)) for k in keys}
    return 0.5*kl(p,m,eps) + 0.5*kl(q,m,eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True)          # ./outputs/neighbor_mask
    ap.add_argument("--out_csv", required=True)          # ./outputs/divergence/xxx.csv
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--news_target", required=True)      # 예: artificial_intelligence
    ap.add_argument("--paper_target", required=True)     # 예: Applications_of_artificial_intelligence
    ap.add_argument("--side", choices=["left","right","both"], default="both")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    rows = []
    for y in args.years:
        news_path  = os.path.join(args.in_root, f"news_{y}_{args.news_target}_neighbors.json")
        paper_path = os.path.join(args.in_root, f"paper_{y}_{args.paper_target}_neighbors.json")

        for side in (["left","right"] if args.side=="both" else [args.side]):

            if not os.path.exists(news_path) or not os.path.exists(paper_path):
                rows.append({
                    "year": y, "side": side,
                    "js": "", "kl_news_paper": "", "kl_paper_news": "",
                    "note": "missing_json"
                })
                continue

            p = load_dist(news_path, side)
            q = load_dist(paper_path, side)

            if not p or not q:
                rows.append({
                    "year": y, "side": side,
                    "js": "", "kl_news_paper": "", "kl_paper_news": "",
                    "note": "empty_dist"
                })
                continue

            rows.append({
                "year": y,
                "side": side,
                "js": js(p, q),
                "kl_news_paper": kl(p, q),
                "kl_paper_news": kl(q, p),
                "note": ""
            })

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["year","side","js","kl_news_paper","kl_paper_news","note"])
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] wrote -> {args.out_csv} (rows={len(rows)})")

if __name__ == "__main__":
    main()
