#train_mlm_preinit.py

import os
import argparse
from typing import List, Set, Optional, Dict

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from collator import NodeOnlyMLMCollator


def read_node_vocab(path: str) -> List[str]:
    nodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            nodes.append(t)
    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for t in nodes:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def build_node_token_ids(tokenizer, node_tokens: List[str]) -> Set[int]:
    ids = set()
    for t in node_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid is None or tid == tokenizer.unk_token_id:
            continue
        ids.add(int(tid))
    return ids


def init_added_node_embeddings_mean_of_subwords(
    model: BertForMaskedLM,
    base_tokenizer,
    extended_tokenizer,
    node_tokens: List[str],
):
    """
    extended_tokenizer: base_tokenizer에 node_tokens를 add_tokens() 한 토크나이저
    base_tokenizer: bert-base-uncased 원본 (node 토큰이 없는) 토크나이저
    """
    emb = model.get_input_embeddings().weight.data  # [vocab, hidden]
    unk_id = base_tokenizer.unk_token_id
    unk_vec = emb[unk_id].clone()

    # 안전장치: base_tokenizer vocab size (원본)
    base_vocab_size = len(base_tokenizer)

    n_inited = 0
    n_fallback = 0

    for node in node_tokens:
        new_id = extended_tokenizer.convert_tokens_to_ids(node)
        if new_id is None or new_id == extended_tokenizer.unk_token_id:
            continue

        # add_tokens로 추가된 토큰만 초기화 (기존 BERT vocab 영역은 건드리지 않음)
        if int(new_id) < base_vocab_size:
            continue

        phrase = node.replace("_", " ").strip()
        # 원본 base_tokenizer로 subword 분해
        sub_tokens = base_tokenizer.tokenize(phrase)
        if not sub_tokens:
            emb[int(new_id)] = unk_vec
            n_fallback += 1
            continue

        sub_ids = base_tokenizer.convert_tokens_to_ids(sub_tokens)
        sub_ids = [sid for sid in sub_ids if sid is not None and sid != base_tokenizer.unk_token_id]
        if not sub_ids:
            emb[int(new_id)] = unk_vec
            n_fallback += 1
            continue

        vec = emb[torch.tensor(sub_ids, dtype=torch.long)].mean(dim=0)
        emb[int(new_id)] = vec
        n_inited += 1

    print(f"[init] initialized added node token embeddings: {n_inited}, fallback_to_UNK: {n_fallback}")

def tokenize_dataset(ds, tokenizer, max_len: int):
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)
    return ds.map(tok, batched=True, remove_columns=["text"])


def make_train_eval(ds, eval_ratio: float, seed: int):
    if eval_ratio <= 0:
        return ds, None
    split = ds.train_test_split(test_size=eval_ratio, seed=seed, shuffle=True)
    return split["train"], split["test"]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_dir", type=str, required=True, help="load_from_disk() 경로 (text 컬럼 필요)")
    ap.add_argument("--node_vocab_txt", type=str, required=True, help="node vocab 파일 (1줄 1토큰)")
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--bert_name", type=str, default="bert-base-uncased")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--eval_ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--mlm_prob", type=float, default=0.15)
    ap.add_argument("--mask_replace_prob", type=float, default=0.9)
    ap.add_argument("--random_replace_prob", type=float, default=0.15)
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=32)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--logging_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--save_total_limit", type=int, default=2)

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 데이터 로드
    ds = load_from_disk(args.dataset_dir)
    if "text" not in ds.column_names:
        raise ValueError(f"Dataset must have 'text' column. columns={ds.column_names}")

    # 2) 노드 vocab
    node_tokens = read_node_vocab(args.node_vocab_txt)
    print(f"[vocab] loaded node tokens: {len(node_tokens)}")

    # 3) base tokenizer & extended tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(args.bert_name, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name, use_fast=True)

    # node tokens를 “추가 토큰”으로 등록 (단일 토큰 유지)
    added = tokenizer.add_tokens(node_tokens, special_tokens=False)
    print(f"[tokenizer] added tokens: {added} (new vocab size={len(tokenizer)})")

    # 4) 모델 로드 & resize
    model = BertForMaskedLM.from_pretrained(args.bert_name)
    model.resize_token_embeddings(len(tokenizer))

    # 5) 추가된 노드 토큰 임베딩을 subword 평균으로 초기화
    init_added_node_embeddings_mean_of_subwords(
        model=model,
        base_tokenizer=base_tokenizer,
        extended_tokenizer=tokenizer,
        node_tokens=node_tokens,
    )

    # 6) 토크나이즈
    tokenized = tokenize_dataset(ds, tokenizer, max_len=args.max_len)
    train_ds, eval_ds = make_train_eval(tokenized, args.eval_ratio, args.seed)

    # 7) node-only 15% MLM collator
    node_token_ids = build_node_token_ids(tokenizer, node_tokens)
    print(f"[mask] node_token_ids in tokenizer: {len(node_token_ids)}")
    collator = NodeOnlyMLMCollator(
        tokenizer=tokenizer,
        node_token_ids=node_token_ids,
        mlm_probability=args.mlm_prob,
        mask_replace_prob=args.mask_replace_prob,
        random_replace_prob=args.random_replace_prob,
    )

    # 8) Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        seed=args.seed,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 9) 저장 (annual에서 그대로 로드)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] baseline saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
