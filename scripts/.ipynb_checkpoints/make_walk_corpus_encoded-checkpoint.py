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


def build_start_sampler(G: nx.Graph, start_mode: str):
    nodes = list(G.nodes())

    if start_mode == "uniform":
        return nodes, None

    if start_mode == "degree":
        deg = dict(G.degree())
        weights = [max(0.0, float(deg.get(n, 0.0))) for n in nodes]
        if sum(weights) <= 0:
            weights = None
        return nodes, weights

    if start_mode == "pagerank":
        pr = nx.pagerank(G)
        weights = [max(0.0, float(pr.get(n, 0.0))) for n in nodes]
        if sum(weights) <= 0:
            weights = None
        return nodes, weights

    raise ValueError(f"Unknown start_mode: {start_mode}")


def _weights_for_per_node_scheme(G: nx.Graph, nodes, scheme: str):
    """
    per_node에서 '추가 배분'할 때 쓸 가중치.
    - uniform: None (균등)
    - degree: degree
    - logdegree: log(1+degree)
    """
    if scheme == "uniform":
        return None

    deg = dict(G.degree())
    if scheme == "degree":
        w = [max(0.0, float(deg.get(n, 0.0))) for n in nodes]
    elif scheme == "logdegree":
        w = [max(0.0, math.log1p(float(deg.get(n, 0.0)))) for n in nodes]
    else:
        raise ValueError(f"Unknown per_node_scheme: {scheme}")

    if sum(w) <= 0:
        return None
    return w


def build_per_node_plan(G: nx.Graph, num_walks: int, rng: random.Random, scheme: str):
    """
    모든 노드에 대해 walk 개수를 '정수로' 배분해서 start 리스트를 만든다.

    핵심:
    - coverage를 위해 노드당 최소 1개를 보장해야 하므로,
      per_node에서는 num_walks가 최소 N(노드수) 이상이어야 한다.
    - 만약 입력 num_walks < N이면 자동으로 num_walks = N으로 올린다.
    - 그 다음 남는 R = num_walks - N을 scheme에 따라 추가 배분.
    """
    nodes = list(G.nodes())
    N = len(nodes)
    if N == 0:
        return [], 0  # plan, effective_num_walks

    effective_num_walks = int(num_walks)
    if effective_num_walks < N:
        effective_num_walks = N  # ✅ 자동 최소 보정 (노드당 1개)

    # 1) 모든 노드에 1개씩 배정
    counts = {n: 1 for n in nodes}

    # 2) 남은 walk를 scheme에 따라 추가 배분
    R = effective_num_walks - N
    if R > 0:
        weights = _weights_for_per_node_scheme(G, nodes, scheme)
        # weights=None이면 uniform으로 처리됨
        extra_nodes = rng.choices(nodes, weights=weights, k=R)
        for n in extra_nodes:
            counts[n] += 1

    # 3) start plan 리스트로 펼치고 섞기
    plan = []
    for n in nodes:
        plan.extend([n] * counts[n])
    rng.shuffle(plan)

    return plan, effective_num_walks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--walk_length", type=int, default=20)
    ap.add_argument("--num_walks", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--start_mode",
        choices=["uniform", "degree", "pagerank", "per_node"],
        default="uniform",
        help="How to sample starting nodes for random walks."
    )

    # ✅ per_node 전용: 배분 방식
    ap.add_argument(
        "--per_node_scheme",
        choices=["uniform", "degree", "logdegree"],
        default="uniform",
        help="(per_node only) How to allocate walk counts across nodes."
    )

    ap.add_argument("--mapping_json", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    G = load_graph(args.pkl)

    # mapping 저장(노드 문자열 -> 인코딩 토큰)
    mapping = {}
    if args.mapping_json:
        for n in G.nodes():
            mapping[str(n)] = encode_node(str(n))
        os.makedirs(os.path.dirname(args.mapping_json), exist_ok=True)
        save_mapping(mapping, args.mapping_json)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ✅ per_node: 노드별로 walk 개수를 배분해서 생성
    if args.start_mode == "per_node":
        start_plan, effective_num_walks = build_per_node_plan(
            G, args.num_walks, rng, scheme=args.per_node_scheme
        )

        with open(args.out, "w", encoding="utf-8") as f:
            for start in start_plan:
                walk = random_walk(G, start, args.walk_length, rng)
                enc = [encode_node(str(x)) for x in walk]
                f.write(" ".join(enc) + "\n")

        if effective_num_walks != args.num_walks:
            print(f"[WARN] per_node requires num_walks >= #nodes. Auto-adjusted: {args.num_walks} -> {effective_num_walks}")

        print(f"[OK] start_mode=per_node scheme={args.per_node_scheme} wrote {effective_num_walks} walks to {args.out}")
        return

    # 기존 모드: 확률적 샘플링
    start_nodes, start_weights = build_start_sampler(G, args.start_mode)

    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.num_walks):
            start = rng.choices(start_nodes, weights=start_weights, k=1)[0]
            walk = random_walk(G, start, args.walk_length, rng)
            enc = [encode_node(str(x)) for x in walk]
            f.write(" ".join(enc) + "\n")

    print(f"[OK] start_mode={args.start_mode} wrote {args.num_walks} walks to {args.out}")


if __name__ == "__main__":
    main()
