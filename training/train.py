"""
train.py
--------
QLoRA fine-tuning of Phi-3-mini-4k-instruct on medical triage dataset.
Designed to run on Kaggle T4 (16GB VRAM).

Usage:
    python training/train.py --config training/config.yaml
"""

import os
import json
import yaml
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_prompt(example: dict) -> str:
    """Format example into Phi-3 instruction format."""
    return (
        f"<|system|>\n{example['instruction']}<|end|>\n"
        f"<|user|>\n{example['input']}<|end|>\n"
        f"<|assistant|>\n{example['output']}<|end|>"
    )


def prepare_dataset(data_path: str) -> Dataset:
    """Load JSONL and format into HuggingFace Dataset."""
    examples = load_jsonl(data_path)
    texts = [format_prompt(ex) for ex in examples]
    return Dataset.from_dict({"text": texts})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"\n── Config loaded from {args.config} ──")
    print(json.dumps(cfg, indent=2))

    # ── 1. Quantisation config ─────────────────────────────────────────────
    print("\n── Loading model in 4-bit (QLoRA) ──")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── 2. Load base model ─────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Model loaded: {cfg['model_name']}")
    print(f"Trainable params before LoRA: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── 3. Apply LoRA ──────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params after LoRA: {trainable:,} "
          f"({100 * trainable / total:.2f}% of total)")

    # ── 4. Load dataset ────────────────────────────────────────────────────
    print(f"\n── Loading dataset from {cfg['train_data_path']} ──")
    train_dataset = prepare_dataset(cfg["train_data_path"])
    print(f"Training examples: {len(train_dataset)}")

    # ── 5. Training arguments ──────────────────────────────────────────────
    output_dir = cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        max_grad_norm=cfg["max_grad_norm"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler"],
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    # ── 6. Train ───────────────────────────────────────────────────────────
    print("\n── Starting training ──")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        tokenizer=tokenizer,
        packing=False,
    )

    trainer.train()

    # ── 7. Save adapter ────────────────────────────────────────────────────
    adapter_path = Path(output_dir) / "final_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\n✅ Adapter saved to {adapter_path}")

    # Save training metadata
    metadata = {
        "model_name":   cfg["model_name"],
        "lora_r":       cfg["lora_r"],
        "lora_alpha":   cfg["lora_alpha"],
        "num_epochs":   cfg["num_epochs"],
        "train_examples": len(train_dataset),
        "output_dir":   str(adapter_path),
    }
    with open(Path(output_dir) / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Training metadata saved.")


if __name__ == "__main__":
    main()