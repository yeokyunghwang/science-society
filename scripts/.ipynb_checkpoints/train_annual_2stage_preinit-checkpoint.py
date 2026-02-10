# train_annual_2stage_preinit.py

import os
import argparse
from typing import List, Tuple, Dict, Set, Optional

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
)
from collator import NodeControlledMLMCollator


def read_node_vocab(path: str) -> List[str]:
    nodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                nodes.append(t)
    # dedup, keep order
    seen = set()
    uniq = []
    for t in nodes:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def build_node_token_ids(tokenizer, node_tokens: List[str]) -> Set[int]:
    ids = set()
    missing = 0
    for t in node_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid is None or tid == tokenizer.unk_token_id:
            missing += 1
            continue
        ids.add(int(tid))
    if missing > 0:
        print(f"[warn] node tokens missing in tokenizer (mapped to UNK): {missing}")
    return ids


def tokenize_dataset(ds, tokenizer, max_len: int):
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)
    return ds.map(tok, batched=True, remove_columns=["text"])


def make_train_eval(ds, eval_ratio: float, seed: int):
    if eval_ratio <= 0:
        return ds, None
    split = ds.train_test_split(test_size=eval_ratio, seed=seed, shuffle=True)
    return split["train"], split["test"]


def parse_year_datasets_arg(s: str) -> List[Tuple[int, str]]:
    """
    "1991=pathA 1992=pathB" 형태(공백 구분) 또는
    "1991=pathA,1992=pathB" 형태(콤마 구분) 모두 지원.
    """
    items = []
    if not s:
        return items
    # allow commas or spaces
    parts = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend(chunk.split())
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Invalid --year_datasets item: {p} (expected YEAR=DIR)")
        y_str, d = p.split("=", 1)
        y = int(y_str)
        items.append((y, d))
    items.sort(key=lambda x: x[0])
    return items


def resolve_plan(
    years: Optional[List[int]],
    dataset_template: Optional[str],
    year_datasets: Optional[str],
) -> List[Tuple[int, str]]:
    """
    Returns sorted list of (year, dataset_dir).
    Priority:
      1) if year_datasets is provided -> use it
      2) else use years + dataset_template
    """
    if year_datasets:
        plan = parse_year_datasets_arg(year_datasets)
        if not plan:
            raise ValueError("--year_datasets was provided but parsed empty.")
        return plan

    if not years or not dataset_template:
        raise ValueError("Provide either --year_datasets OR (--years AND --dataset_template).")

    plan = []
    for y in years:
        ds_dir = dataset_template.format(year=y)
        plan.append((int(y), ds_dir))
    plan.sort(key=lambda x: x[0])
    return plan


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--init_dir", type=str, required=True, help="baseline 또는 이전연도 체크포인트 디렉토리(시작점)")
    ap.add_argument("--node_vocab_txt", type=str, required=True, help="node vocab txt (baseline과 동일)")

    # 옵션 1: years + template
    ap.add_argument("--years", type=int, nargs="*", default=None, help="예: --years 1991 1992 1993")
    ap.add_argument("--dataset_template", type=str, default=None, help="예: ./dataset/news")

    # 옵션 2: year=dir 명시 매핑
    ap.add_argument(
        "--year_datasets",
        type=str,
        default=None,
        help='예: --year_datasets "1991=./ds1991 1992=./ds1992" 또는 "1991=./ds1991,1992=./ds1992"',
    )

    ap.add_argument("--output_base_dir", type=str, required=True, help="연도별 출력 루트. 예: ./ckpt/news_annual")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--eval_ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    
    ap.add_argument("--mask_replace_prob", type=float, default=0.9)
    ap.add_argument("--random_replace_prob", type=float, default=0.05)

    # 학습 하이퍼
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=32)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--logging_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--save_total_limit", type=int, default=2)

    # controlled masking
    ap.add_argument("--exact_k_masks", type=int, default=1, help="샘플당 마스킹할 노드 개수 (기본=1)")

    args = ap.parse_args()
    os.makedirs(args.output_base_dir, exist_ok=True)

    # 학습 계획(연도->데이터셋 경로)
    plan = resolve_plan(args.years, args.dataset_template, args.year_datasets)
    print("[plan]")
    for y, d in plan:
        print(f"  - {y}: {d}")

    # node vocab
    node_tokens = read_node_vocab(args.node_vocab_txt)
    print(f"[vocab] node tokens: {len(node_tokens)}")

    # chaining init_dir
    current_init_dir = args.init_dir

    for year, dataset_dir in plan:
        print(f"\n==============================")
        print(f"[year {year}] init_dir = {current_init_dir}")
        print(f"[year {year}] dataset_dir = {dataset_dir}")

        year_out_dir = os.path.join(args.output_base_dir, str(year))
        os.makedirs(year_out_dir, exist_ok=True)

        # 1) load model/tokenizer from previous checkpoint (baseline or last year)
        tokenizer = AutoTokenizer.from_pretrained(current_init_dir, use_fast=True)
        model = BertForMaskedLM.from_pretrained(current_init_dir)

        # 2) node ids (mask candidates)
        node_token_ids = build_node_token_ids(tokenizer, node_tokens)
        if not node_token_ids:
            raise ValueError(
                f"[year {year}] node_token_ids is empty. "
                f"Check that baseline tokenizer contains your node tokens."
            )
        print(f"[year {year}] node_token_ids in tokenizer: {len(node_token_ids)}")

        # 3) dataset
        ds = load_from_disk(dataset_dir)
        if "text" not in ds.column_names:
            raise ValueError(f"[year {year}] Dataset must have 'text' column. columns={ds.column_names}")

        tokenized = tokenize_dataset(ds, tokenizer, max_len=args.max_len)
        train_ds, eval_ds = make_train_eval(tokenized, args.eval_ratio, args.seed)

        # 4) collator: exactly K node masks per sample (default=1)
        collator = NodeControlledMLMCollator(
            tokenizer=tokenizer,
            node_token_ids=node_token_ids,
            exact_k_masks=args.exact_k_masks,
            mask_replace_prob=args.mask_replace_prob,
            random_replace_prob=args.random_replace_prob,
        )


        # 5) training args (연도별로 새 output_dir)
        training_args = TrainingArguments(
            output_dir=year_out_dir,
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

        # 6) save year checkpoint
        trainer.save_model(year_out_dir)
        tokenizer.save_pretrained(year_out_dir)
        print(f"[year {year}] saved: {year_out_dir}")

        # 7) chain to next year
        current_init_dir = year_out_dir

    print("\n[done] all years finished.")
    print(f"[done] final checkpoint: {current_init_dir}")


if __name__ == "__main__":
    main()
