import os, argparse, pickle, random, math
import networkx as nx
from node_text_codec import encode_node, save_mapping

def load_graph(pkl_path: str):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    # pkl 포맷: {"G": networkx.Graph}
    return obj["G"]

def random_walk(G: nx.Graph, start, L: int, rng: random.Random):
    w = [start]
    for _ in range(L - 1):
        nbrs = list(G.neighbors(w[-1]))
        if not nbrs:
            break
        w.append(rng.choice(nbrs))
    return w

# def allocate_walks_proportional(G: nx.Graph, total_walks: int, mode: str):
#     """
#     mode: "degree" or "logdegree"
#     Return dict: node -> k_i, where sum(k_i)=total_walks and k_i ∝ weight(node).
#     weight(node) = degree(node)            if mode=="degree"
#                  = log1p(degree(node))     if mode=="logdegree"
#     """
#     nodes = list(G.nodes())
#     deg = dict(G.degree())

#     if mode == "degree":
#         w = [max(0.0, float(deg.get(n, 0))) for n in nodes]
#     elif mode == "logdegree":
#         w = [max(0.0, math.log1p(float(deg.get(n, 0)))) for n in nodes]
#     else:
#         raise ValueError(f"Unknown alloc mode: {mode}")

#     S = sum(w)

#     # 엣지 거의 없거나 degree/logdegree 합이 0인 경우: 모든 노드 균등 배분 fallback
#     if S <= 0:
#         base = total_walks // max(1, len(nodes))
#         rem = total_walks - base * len(nodes)
#         out = {n: base for n in nodes}
#         for n in nodes[:rem]:
#             out[n] += 1
#         return out

#     # 1) floor 배분
#     raw = [total_walks * (wi / S) for wi in w]
#     ks = [int(math.floor(x)) for x in raw]
#     cur = sum(ks)
#     rem = total_walks - cur

#     # 2) largest remainder로 rem개를 소수점 큰 순서대로 +1
#     frac = [(raw[i] - ks[i], i) for i in range(len(nodes))]
#     frac.sort(reverse=True)
#     for j in range(rem):
#         _, i = frac[j]
#         ks[i] += 1

#     return {nodes[i]: ks[i] for i in range(len(nodes))}

def allocate_walks_pos_min1(G: nx.Graph, total_walks: int, mode: str):
    deg = dict(G.degree())
    nodes = [n for n, d in deg.items() if d > 0]
    M = len(nodes)

    if M == 0:
        return {}

    # 최소 1개씩
    alloc = {n: 1 for n in nodes}
    remaining = total_walks - M

    if remaining <= 0:
        return alloc

    # weight
    if mode == "degree":
        w = {n: float(deg[n]) for n in nodes}
    elif mode == "logdegree":
        w = {n: math.log1p(float(deg[n])) for n in nodes}
    else:
        raise ValueError(mode)

    S = sum(w.values())
    raw = {n: remaining * (w[n] / S) for n in nodes}
    ks = {n: int(math.floor(raw[n])) for n in nodes}
    rem = remaining - sum(ks.values())

    # tie-break 결정성 확보
    frac = sorted(
        ((raw[n] - ks[n], str(n), n) for n in nodes),
        reverse=True,
    )
    for i in range(rem):
        _, _, n = frac[i]
        ks[n] += 1

    for n in nodes:
        alloc[n] += ks[n]

    return alloc

    
def build_start_sampler(G: nx.Graph, start_mode: str):
    nodes = list(G.nodes())

    if start_mode == "uniform":
        # 모든 노드 균등
        weights = None
        return nodes, weights

    if start_mode == "degree":
        # degree 비례 샘플링(각 walk마다 시작노드 샘플)
        deg = dict(G.degree())
        weights = [max(0.0, float(deg.get(n, 0.0))) for n in nodes]
        if sum(weights) <= 0:
            weights = None
        return nodes, weights

    if start_mode == "pagerank":
        # pagerank 비례 샘플링(각 walk마다 시작노드 샘플)
        pr = nx.pagerank(G)
        weights = [max(0.0, float(pr.get(n, 0.0))) for n in nodes]
        if sum(weights) <= 0:
            weights = None
        return nodes, weights

    # ✅ 모든 노드를 대상으로 "노드별 walk 개수"를 배분해서 생성
    if start_mode == "all_degree":
        return nodes, "ALL_DEGREE"  # sentinel
    if start_mode == "all_logdegree":
        return nodes, "ALL_LOGDEGREE"  # sentinel

    raise ValueError(f"Unknown start_mode: {start_mode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--walk_length", type=int, default=20)
    ap.add_argument("--num_walks", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--start_mode",
        choices=["uniform", "degree", "pagerank", "all_degree", "all_logdegree"],
        default="uniform",
        help=(
            "uniform/degree/pagerank: sample start node per-walk. "
            "all_degree: generate walks for every node with count ∝ degree. "
            "all_logdegree: generate walks for every node with count ∝ log1p(degree)."
        ),
    )

    ap.add_argument("--mapping_json", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    G = load_graph(args.pkl)

    deg = dict(G.degree())
    pos_nodes = [n for n, d in deg.items() if d > 0]
    M = len(pos_nodes)
    
    if args.start_mode in ("all_degree", "all_logdegree"):
        if args.num_walks < M:
            print(
                f"[WARN] num_walks={args.num_walks} < #deg>0 nodes={M}. "
                f"Clamping num_walks to {M}."
            )
            args.num_walks = M


    # mapping 저장(노드 문자열 -> 인코딩 토큰)
    mapping = {}
    if args.mapping_json:
        for n in G.nodes():
            mapping[str(n)] = encode_node(str(n))
        os.makedirs(os.path.dirname(args.mapping_json), exist_ok=True)
        save_mapping(mapping, args.mapping_json)

    # 시작 노드 샘플러 준비
    start_nodes, start_weights = build_start_sampler(G, args.start_mode)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        if start_weights in ("ALL_DEGREE", "ALL_LOGDEGREE"):
            alloc_mode = "degree" if start_weights == "ALL_DEGREE" else "logdegree"
            alloc = allocate_walks_pos_min1(G, args.num_walks, mode=alloc_mode)

            for n in start_nodes:
                k = alloc.get(n, 0)
                for _ in range(k):
                    walk = random_walk(G, n, args.walk_length, rng)
                    enc = [encode_node(str(x)) for x in walk]
                    f.write(" ".join(enc) + "\n")

            total_written = sum(alloc.values())
            print(f"[OK] start_mode={args.start_mode} wrote {total_written} walks to {args.out}")

        else:
            # 기존 방식: walk마다 시작노드 샘플링
            for _ in range(args.num_walks):
                start = rng.choices(start_nodes, weights=start_weights, k=1)[0]
                walk = random_walk(G, start, args.walk_length, rng)
                enc = [encode_node(str(x)) for x in walk]
                f.write(" ".join(enc) + "\n")

            print(f"[OK] start_mode={args.start_mode} wrote {args.num_walks} walks to {args.out}")

if __name__ == "__main__":
    main()
