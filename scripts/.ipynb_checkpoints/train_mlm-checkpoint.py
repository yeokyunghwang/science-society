import os
import math
import argparse

import torch
from datasets import load_from_disk
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
)

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_dir", type=str, required=True, help="Path like ./datasets/news_1990")
    ap.add_argument("--tokenizer_json", type=str, required=True, help="Path to tokenizer.json")
    ap.add_argument("--out_dir", type=str, required=True, help="Checkpoint output dir")

    ap.add_argument("--mlm_prob", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)

    # BERT-from-scratch hyperparams
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--intermediate_size", type=int, default=1024)  # FFN size
    ap.add_argument("--dropout", type=float, default=0.1)

    # eval
    ap.add_argument("--eval_ratio", type=float, default=0.05, help="fraction for eval split (0~1)")
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=500)

    # optional: push to hub
    ap.add_argument("--push_to_hub", action="store_true", help="Upload to Hugging Face Hub")
    ap.add_argument("--hub_model_id", type=str, default=None, help="e.g. username/model_name (optional)")
    ap.add_argument("--hub_private", action="store_true", help="Create private repo on Hub (if supported)")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Tokenizer (WordLevel tokenizer.json)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer_json,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    ds = load_from_disk(args.dataset_dir)

    # Split train/eval if requested
    if args.eval_ratio > 0:
        split = ds.train_test_split(test_size=args.eval_ratio, seed=args.seed, shuffle=True)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_len,
        )

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    eval_tok = eval_ds.map(tokenize, batched=True, remove_columns=["text"]) if eval_ds is not None else None

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_prob,
    )

    # Build BERT MLM from scratch (matches your custom vocab)
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=args.max_len,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForMaskedLM(config)

    # TrainingArguments (includes optional push_to_hub)
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,

        report_to="none",
        seed=args.seed,
        fp16=torch.cuda.is_available(),

        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Start training")
    trainer.train()

    # Save locally
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # Evaluate + perplexity
    if eval_tok is not None:
        eval_results = trainer.evaluate()
        ppl = math.exp(eval_results["eval_loss"]) if "eval_loss" in eval_results else float("nan")
        print(f"[OK] eval_loss={eval_results.get('eval_loss')}  perplexity={ppl:.2f}")

    # Push to hub if requested
    if args.push_to_hub:
        trainer.push_to_hub()

    print(f"[OK] trained & saved to {args.out_dir}")

if __name__ == "__main__":
    main()
