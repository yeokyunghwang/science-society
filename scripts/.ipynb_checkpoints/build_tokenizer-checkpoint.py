import os
import json
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_txt", type=str, required=True, help="concept_vocab.txt path")
    ap.add_argument("--out_dir", type=str, default="./tokenizer")
    args = ap.parse_args()

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    with open(args.vocab_txt, "r", encoding="utf-8") as f:
        concepts = [line.strip() for line in f if line.strip()]

    # Node level vocab: token -> id
    vocab = {}
    idx = 0
    for tok in special_tokens:
        vocab[tok] = idx
        idx += 1

    for tok in concepts:
        if tok not in vocab:
            vocab[tok] = idx
            idx += 1

    os.makedirs(args.out_dir, exist_ok=True)

    # minimal tokenizer.json compatible with PreTrainedTokenizerFast(tokenizer_file=...)
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "[UNK]"
        }
    }

    out_path = os.path.join(args.out_dir, "tokenizer.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False)

    print(f"[OK] tokenizer.json saved_to={out_path}")
    print(f"[OK] vocab_size={len(vocab)} (incl specials={len(special_tokens)})")

if __name__ == "__main__":
    main()
