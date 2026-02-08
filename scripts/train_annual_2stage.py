import os
import math
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    PreTrainedTokenizerFast,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from node_controlled_collator import NodeControlledMLMCollator

def load_tokenizer(tokenizer_json: str) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def tokenize_dataset(ds, tokenizer, max_len: int):
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)
    return ds.map(tok, batched=True, remove_columns=["text"])


def make_train_eval(ds, eval_ratio: float, seed: int):
    if eval_ratio <= 0:
        return ds, None
    split = ds.train_test_split(test_size=eval_ratio, seed=seed, shuffle=True)
    return split["train"], split["test"]


def train_one_stage(
    model,
    train_ds,
    eval_ds,
    tokenizer,
    out_dir,
    batch_size,
    epochs,
    lr,
    seed,
    collator,
    logging_steps: int,
    save_steps: int,
    weight_decay: float,
):
    os.makedirs(out_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=logging_steps,
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    if eval_ds is not None:
        eval_results = trainer.evaluate()
        loss = eval_results.get("eval_loss", None)
        ppl = math.exp(loss) if loss is not None else float("nan")
        print(f"[EVAL] {out_dir}  eval_loss={loss:.4f}  ppl={ppl:.2f}")

    return model


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--domain", choices=["news", "paper"], required=True)
    ap.add_argument("--dataset_root", type=str, default="./datasets")
    ap.add_argument("--tokenizer_json", type=str, default="./tokenizer/tokenizer.json")
    ap.add_argument("--ckpt_root", type=str, default="./checkpoints/annual_node")
    ap.add_argument("--years", type=int, nargs="+", required=True)

    # starting checkpoint (e.g., 1990 seed)
    ap.add_argument("--seed_ckpt", type=str, required=True)

    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)

    # eval
    ap.add_argument("--eval_ratio", type=float, default=0.05)
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    # control: optionally skip stage1 after seed
    ap.add_argument("--skip_stage1_after_seed", action="store_true")

    # Stage 1 (random MLM)
    ap.add_argument("--stage1_mlm_prob", type=float, default=0.15)
    ap.add_argument("--stage1_epochs", type=int, default=1)
    ap.add_argument("--stage1_batch", type=int, default=32)
    ap.add_argument("--stage1_lr", type=float, default=5e-4)

    # Stage 2 (controlled node masking)
    ap.add_argument("--stage2_epochs", type=int, default=1)
    ap.add_argument("--stage2_batch", type=int, default=32)
    ap.add_argument("--stage2_lr", type=float, default=2e-4)
    ap.add_argument("--stage2_exact_k_masks", type=int, default=1)
    ap.add_argument("--stage2_min_masks", type=int, default=1)
    ap.add_argument("--stage2_max_masks", type=int, default=1)

    args = ap.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_json)
    model = BertForMaskedLM.from_pretrained(args.seed_ckpt)

    for y in args.years:
        if y == 1990:
            continue

        ds_dir = os.path.join(args.dataset_root, f"{args.domain}_{y}")
        if not os.path.exists(ds_dir):
            raise FileNotFoundError(ds_dir)

        ds = load_from_disk(ds_dir)
        tokenized = tokenize_dataset(ds, tokenizer, args.max_len)
        train_tok, eval_tok = make_train_eval(tokenized, args.eval_ratio, args.seed)

        # ---- Stage 1: random 15% MLM (optional) ----
        if not args.skip_stage1_after_seed:
            stage1_out = os.path.join(args.ckpt_root, args.domain, f"year={y}", "stage1")
            stage1_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=args.stage1_mlm_prob,
            )
            model = train_one_stage(
                model=model,
                train_ds=train_tok,
                eval_ds=eval_tok,
                tokenizer=tokenizer,
                out_dir=stage1_out,
                batch_size=args.stage1_batch,
                epochs=args.stage1_epochs,
                lr=args.stage1_lr,
                seed=args.seed,
                collator=stage1_collator,
                logging_steps=args.logging_steps,
                save_steps=args.save_steps,
                weight_decay=args.weight_decay,
            )

        # ---- Stage 2: controlled node masking ----
        stage2_out = os.path.join(args.ckpt_root, args.domain, f"year={y}", "stage2")
        stage2_collator = NodeControlledMLMCollator(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.0,  # 안전: 확률 기반 마스킹 끄고 controlled만 쓰기
            exact_k_masks=args.stage2_exact_k_masks,
            min_masks=args.stage2_min_masks,
            max_masks=args.stage2_max_masks,
        )
        model = train_one_stage(
            model=model,
            train_ds=train_tok,
            eval_ds=None, # eval_tok
            tokenizer=tokenizer,
            out_dir=stage2_out,
            batch_size=args.stage2_batch,
            epochs=args.stage2_epochs,
            lr=args.stage2_lr,
            seed=args.seed,
            collator=stage2_collator,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            weight_decay=args.weight_decay,
        )

        print(f"[OK] finished year {y}: stage2->{stage2_out}")

    print("[DONE] Annual 2-stage continual training finished.")


if __name__ == "__main__":
    main()
