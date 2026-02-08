import os, argparse, pickle, csv
from node_text_codec import encode_node, save_mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--out_vocab", required=True)
    ap.add_argument("--out_mapping", required=True, help="원본 노드 → 토큰 매핑 (dict)")
    ap.add_argument("--out_common_dir", required=True, help="Directory to save per-year common node CSVs")
    args = ap.parse_args()

    vocab, mapping = set(), {}

    os.makedirs(os.path.dirname(args.out_vocab), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_mapping), exist_ok=True)
    os.makedirs(args.out_common_dir, exist_ok=True)

    for y in args.years:
        pkl_news = f"{args.data_root}/news/news_network_{y}_m0.0-M1.0.pkl"
        pkl_paper = f"{args.data_root}/paper/paper_network_{y}_m0.0-M0.5.pkl"

        G_news = pickle.load(open(pkl_news, "rb"))["G"]
        G_paper = pickle.load(open(pkl_paper, "rb"))["G"]

        news_nodes = set(G_news.nodes())
        paper_nodes = set(G_paper.nodes())

        common_nodes = sorted(news_nodes & paper_nodes, key=lambda x: str(x))

        out_csv = os.path.join(args.out_common_dir, f"common_nodes_{y}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["year", "node_raw", "node_encoded"])
            for n in common_nodes:
                w.writerow([y, str(n), encode_node(n)])

        for n in (news_nodes | paper_nodes):
            e = encode_node(n)
            vocab.add(e)
            mapping[str(n)] = e

    with open(args.out_vocab, "w", encoding="utf-8") as f:
        for v in sorted(vocab):
            f.write(v + "\n")

    save_mapping(mapping, args.out_mapping)

if __name__ == "__main__":
    main()
