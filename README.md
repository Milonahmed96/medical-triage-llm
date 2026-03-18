<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Georgia&size=32&duration=3000&pause=1000&color=1F3864&center=true&vCenter=true&width=600&lines=Medical+Triage+LLM;Domain+Fine-tuning+with+QLoRA" alt="Medical Triage LLM"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.12-1F3864?style=for-the-badge&logo=python&logoColor=white)
![Phi-3-mini](https://img.shields.io/badge/Phi--3--mini-3.8B-D97757?style=for-the-badge&logo=microsoft&logoColor=white)
![QLoRA](https://img.shields.io/badge/QLoRA-Fine--tuned-2E75B6?style=for-the-badge)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Published-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![CI](https://img.shields.io/github/actions/workflow/status/Milonahmed96/medical-triage-llm/tests.yml?style=for-the-badge&label=CI)

<br/>

**[🤗 Model on Hugging Face](https://huggingface.co/Milon96/phi3-medical-triage)** &nbsp;·&nbsp;
**[🚀 Live Demo](https://huggingface.co/spaces/Milon96/medical-triage-demo)** &nbsp;·&nbsp;
**[📊 Evaluation Results](#evaluation-results)** &nbsp;·&nbsp;
**[⚙️ Training](#training)** &nbsp;·&nbsp;
**[🚀 Quick Start](#quick-start)**

</div>

---

## Overview

Fine-tuned `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters) on a medical triage classification task using QLoRA — 4-bit quantisation plus LoRA adapters — entirely on a free Kaggle T4 GPU (16GB VRAM).

Given a patient symptom description, the model classifies the urgency level and provides a brief clinical rationale. The fine-tuned model outperforms the base model by 27 percentage points on triage accuracy and achieves 100% recall on life-threatening EMERGENCY cases.

**This is an educational portfolio project. Not for clinical use.**

---

## Evaluation Results

Evaluated on 100 held-out synthetic patient presentations (25 per triage level).

| Model | Triage Accuracy | ROUGE-L | EMERGENCY Recall |
|---|---|---|---|
| **Fine-tuned Phi-3-mini (this model)** | **64%** | **0.42** | **100%** |
| Base Phi-3-mini (zero-shot) | 37% | 0.21 | 100% |

**Key finding:** The fine-tuned model never misses a genuine emergency. The main remaining weakness is the URGENT class — frequently over-triaged to EMERGENCY — which is the clinically safer error direction.

### Per-Class Accuracy

| Triage Level | Meaning | Fine-tuned | Base Model |
|---|---|---|---|
| EMERGENCY | Life-threatening — call 999 | 100% | 100% |
| URGENT | Same-day care required | 4% | 0% |
| SEMI-URGENT | GP within 24–48 hours | 60% | 48% |
| NON-URGENT | Routine GP appointment | 92% | 0% |

---

## Architecture
```
Patient Symptom Description
        │
        ▼  Phi-3 instruction format (<|system|> / <|user|> / <|assistant|>)
Formatted Prompt
        │
        ▼  microsoft/Phi-3-mini-4k-instruct (3.8B, 4-bit NF4 quantised)
Frozen Base Model
        │
        ▼  LoRA adapters (r=16, target: qkv_proj + o_proj)
Domain-Adapted Model
        │
        ▼
Triage Level + Clinical Rationale
```

---

## Dataset

Training data: 200 synthetic patient presentations generated using Claude (Anthropic), covering realistic clinical scenarios across all four triage levels with varied ages, genders, and symptom descriptions.

| Split | Examples | Per Class |
|---|---|---|
| Training | 200 | 50 |
| Test | 100 | 25 |

The dataset was generated because MedQuAD (47,000 NIH Q&A pairs) uses a "What is X?" question format that does not match real patient presentation language. Synthetic generation using Claude produced significantly better training signal for this task — a key lesson documented in the project.

---

## Training

**Platform:** Kaggle Notebooks (T4 GPU, 16GB VRAM — free tier)

**Method:** QLoRA — 4-bit NF4 quantisation reduces the 3.8B parameter model from ~28GB to ~6GB VRAM, making fine-tuning feasible on a free GPU.

| Hyperparameter | Value |
|---|---|
| Base model | microsoft/Phi-3-mini-4k-instruct |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | qkv_proj, o_proj |
| Quantisation | 4-bit NF4 (bitsandbytes) |
| Epochs | 5 |
| Learning rate | 1e-4 |
| Batch size | 4 (grad accumulation × 4 = effective 16) |
| Trainable parameters | 9,437,184 (0.468% of total) |
| Training time | ~8 minutes on T4 |

---

## Quick Start
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_id = "Milon96/phi3-medical-triage"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

instruction = """You are a medical triage assistant. Classify the following
patient presentation into: EMERGENCY, URGENT, SEMI-URGENT, NON-URGENT.
EMERGENCY: Life-threatening — call 999 immediately.
URGENT: Serious — same-day care required.
SEMI-URGENT: Needs care within 24-48 hours.
NON-URGENT: Routine — GP appointment within a week."""

patient = "72-year-old male with sudden chest pain radiating to his left arm, sweating, nausea. Onset 15 minutes ago."

prompt = f"<|system|>\n{instruction}<|end|>\n<|user|>\n{patient}<|end|>\n<|assistant|>\n"
result = pipe(prompt, max_new_tokens=150, do_sample=False, temperature=1.0)
response = result[0]["generated_text"].split("<|assistant|>")[-1].replace("<|end|>","").strip()
print(response)
# Triage Level: EMERGENCY
# Symptoms are consistent with acute myocardial infarction...
```

---

## Local Setup
```bash
git clone git@github.com:Milonahmed96/medical-triage-llm.git
cd medical-triage-llm
python -m venv venv
.\venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

Regenerate the dataset (requires MedQuAD cloned to `~/MedQuAD`):
```bash
python data/process_medquad.py --medquad_path "C:/path/to/MedQuAD"
python -m pytest tests/test_dataset.py -v
```

---

## Project Structure
```
medical-triage-llm/
├── data/
│   ├── process_medquad.py        # MedQuAD filtering and processing
│   ├── validate_dataset.py       # Dataset schema validation
│   ├── dataset_train.jsonl       # 1,500 MedQuAD training examples
│   ├── dataset_test.jsonl        # 100 MedQuAD test examples
│   ├── synthetic_examples.jsonl  # 200 synthetic training examples
│   └── synthetic_test.jsonl      # 100 synthetic test examples
├── training/
│   ├── train.py                  # QLoRA fine-tuning script
│   └── config.yaml               # Hyperparameters
├── evaluation/
│   ├── evaluate.py               # Accuracy + ROUGE-L scorer
│   ├── compare_models.py         # 3-model comparison
│   └── results.json              # Benchmark results
├── notebooks/
│   └── medical_triage_finetune.ipynb  # Full Kaggle training notebook
├── tests/
│   ├── test_dataset.py           # 22 dataset validation tests
│   └── test_evaluate.py          # Evaluation unit tests
├── .github/workflows/tests.yml   # CI — green on every push
├── model_card.md                 # Hugging Face model card
└── requirements.txt
```

---

## Lessons Learned

Three genuine engineering insights from building this project:

**1. Data quality beats data quantity.** 200 high-quality synthetic patient presentations outperformed 1,500 MedQuAD examples. MedQuAD uses "What is X?" question format — the model learned to classify medical information questions, not patient presentations.

**2. LoRA target modules are architecture-specific.** Phi-3-mini uses `qkv_proj` and `o_proj` instead of the standard `q_proj`/`v_proj` split. Discovering this through error is exactly how you learn it permanently.

**3. Test set distribution must match training distribution.** Evaluating a model trained on synthetic patient presentations against MedQuAD-style questions gives artificially low scores. The evaluation dataset must reflect the real inference use case.

---

## Licence

MIT
