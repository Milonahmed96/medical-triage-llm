"""
evaluate.py
-----------
Evaluates fine-tuned model on the 100-question test set.
Computes triage accuracy, ROUGE-L, per-class accuracy,
and EMERGENCY recall.

Usage (after training on Kaggle):
    python evaluation/evaluate.py \
        --model_path outputs/phi3-medical-triage/final_adapter \
        --test_data data/dataset_test.jsonl
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from rouge_score import rouge_scorer
from sklearn.metrics import confusion_matrix, classification_report


VALID_LEVELS = ["EMERGENCY", "URGENT", "SEMI-URGENT", "NON-URGENT"]


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_triage_level(text: str) -> str:
    """Extract triage level label from model output."""
    text_upper = text.upper()
    # Try exact match first
    for level in VALID_LEVELS:
        if f"TRIAGE LEVEL: {level}" in text_upper:
            return level
    # Fallback: find level anywhere in text
    for level in VALID_LEVELS:
        if level in text_upper:
            return level
    return "UNKNOWN"


def generate_prediction(pipe, instruction: str, input_text: str,
                        max_new_tokens: int = 256) -> str:
    """Generate model prediction for a single example."""
    prompt = (
        f"<|system|>\n{instruction}<|end|>\n"
        f"<|user|>\n{input_text}<|end|>\n"
        f"<|assistant|>\n"
    )
    result = pipe(prompt, max_new_tokens=max_new_tokens,
                  do_sample=False, temperature=1.0)
    generated = result[0]["generated_text"]
    # Extract only the assistant response
    if "<|assistant|>" in generated:
        response = generated.split("<|assistant|>")[-1]
        response = response.replace("<|end|>", "").strip()
        return response
    return generated.strip()


def evaluate_model(model_path: str, test_data: list[dict],
                   base_model_name: str = None) -> dict:
    """Load model and evaluate on test set."""
    print(f"\n── Loading model from {model_path} ──")

    if base_model_name:
        # Load fine-tuned model with adapter
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        # Load model directly (base model or merged)
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    tokenizer.pad_token = tokenizer.eos_token
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # ── Generate predictions ───────────────────────────────────────────────
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    y_true, y_pred = [], []
    rouge_scores = []

    print(f"Evaluating {len(test_data)} test examples...")
    for i, ex in enumerate(test_data):
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test_data)}")

        prediction = generate_prediction(pipe, ex["instruction"], ex["input"])
        predicted_level = extract_triage_level(prediction)
        true_level = ex["triage_level"]

        y_true.append(true_level)
        y_pred.append(predicted_level)

        rouge = scorer.score(ex["output"], prediction)
        rouge_scores.append(rouge["rougeL"].fmeasure)

    # ── Compute metrics ────────────────────────────────────────────────────
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true)

    # Per-class accuracy
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1
    per_class_accuracy = {
        level: class_correct[level] / class_total[level]
        for level in VALID_LEVELS if class_total[level] > 0
    }

    # EMERGENCY recall
    emergency_true  = [t == "EMERGENCY" for t in y_true]
    emergency_pred  = [p == "EMERGENCY" for p in y_pred]
    emergency_tp    = sum(t and p for t, p in zip(emergency_true, emergency_pred))
    emergency_total = sum(emergency_true)
    emergency_recall = emergency_tp / emergency_total if emergency_total > 0 else 0

    avg_rouge = sum(rouge_scores) / len(rouge_scores)

    results = {
        "model_path":          model_path,
        "total_examples":      len(test_data),
        "correct":             correct,
        "triage_accuracy":     round(accuracy, 4),
        "rouge_l":             round(avg_rouge, 4),
        "emergency_recall":    round(emergency_recall, 4),
        "per_class_accuracy":  {k: round(v, 4) for k, v in per_class_accuracy.items()},
        "predictions":         list(zip(y_true, y_pred)),
    }

    print(f"\n── Results for {model_path} ──")
    print(f"  Triage Accuracy:   {accuracy:.1%}")
    print(f"  ROUGE-L:           {avg_rouge:.4f}")
    print(f"  EMERGENCY Recall:  {emergency_recall:.1%}")
    print(f"  Per-class accuracy: {per_class_accuracy}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  type=str, required=True)
    parser.add_argument("--test_data",   type=str,
                        default="data/dataset_test.jsonl")
    parser.add_argument("--output_path", type=str,
                        default="evaluation/results.json")
    parser.add_argument("--base_model",  type=str, default=None,
                        help="Base model name if loading adapter separately")
    args = parser.parse_args()

    test_data = load_jsonl(args.test_data)
    print(f"Loaded {len(test_data)} test examples")

    results = evaluate_model(args.model_path, test_data, args.base_model)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {args.output_path}")


if __name__ == "__main__":
    main()