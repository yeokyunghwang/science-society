# scripts/check_conditional_from_corpus.py
import argparse
from collections import Counter, defaultdict
import math

def iter_corpus(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line.split()

def normalize(counter: Counter):
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}

def jsd(p: dict, q: dict, eps: float = 1e-12):
    # Jensen–Shannon divergence (base e)
    keys = set(p) | set(q)
    m = {}
    for k in keys:
        m[k] = 0.5 * p.get(k, 0.0) + 0.5 * q.get(k, 0.0)

    def kl(a, b):
        s = 0.0
        for k in keys:
            ak = a.get(k, 0.0)
            if ak <= 0:
                continue
            bk = b.get(k, 0.0)
            s += ak * math.log(ak / max(bk, eps))
        return s

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def topk_list(counter: Counter, k: int):
    return [t for t, _ in counter.most_common(k)]

def overlap(a_list, b_list):
    a = set(a_list)
    b = set(b_list)
    return len(a & b), len(a), len(b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="e.g. ./corpus/news_1991.txt")
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--distance", type=int, default=1, help="relative position from target (e.g. +1, -1, +10)")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--max_lines", type=int, default=0, help="0 means no limit")
    args = ap.parse_args()

    # (A) prior: 전체 토큰 분포
    unigram = Counter()

    # (B) conditional: target별로 (i+distance) 위치 토큰 카운트
    cond = {t: Counter() for t in args.targets}
    n_hits = defaultdict(int)     # target 등장 횟수(조건 만족 contexts 수)
    n_valid = defaultdict(int)    # distance 위치가 유효했던 횟수

    for li, toks in enumerate(iter_corpus(args.corpus)):
        if args.max_lines and li >= args.max_lines:
            break

        unigram.update(toks)

        # 각 target별로 라인 내 모든 occurrence 스캔
        for i, tok in enumerate(toks):
            if tok not in cond:
                continue
            n_hits[tok] += 1
            pos = i + args.distance
            if 0 <= pos < len(toks):
                cond[tok][toks[pos]] += 1
                n_valid[tok] += 1

    prior_p = normalize(unigram)
    prior_top = topk_list(unigram, args.topk)
    prior_top1 = prior_top[0] if prior_top else None

    print(f"\n[CORPUS] {args.corpus}")
    print(f"[DIST] distance={args.distance}")
    print(f"[PRIOR] total_tokens={sum(unigram.values())} unique={len(unigram)}")
    print(f"[PRIOR] top{args.topk}[:10]={prior_top[:10]}")
    print()

    for t in args.targets:
        c = cond[t]
        cp = normalize(c)
        ctop = topk_list(c, args.topk)
        ctop1 = ctop[0] if ctop else None

        j = jsd(cp, prior_p) if cp else float("nan")
        inter, la, lb = overlap(ctop, prior_top)
        topk_overlap = inter / max(1, args.topk)
        top1_same = (ctop1 == prior_top1)

        print(f"===== TARGET={t} =====")
        print(f"occurrences_in_lines={n_hits[t]}  valid_positions={n_valid[t]}")
        print(f"cond_unique={len(c)}")
        print(f"cond_top{args.topk}[:10]={ctop[:10]}")
        print(f"top1_same_as_prior={top1_same} (cond_top1={ctop1}, prior_top1={prior_top1})")
        print(f"top{args.topk}_overlap_with_prior={topk_overlap:.3f} (|∩|={inter})")
        print(f"JSD(cond || prior)={j:.6f}")
        print()

if __name__ == "__main__":
    main()
