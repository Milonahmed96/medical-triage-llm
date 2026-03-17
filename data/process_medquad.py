"""
process_medquad.py
------------------
Downloads and processes MedQuAD XML files into instruction-tuning format
for medical triage classification.

Output:
    data/dataset_train.jsonl  -- 1,500 training examples (375 per class)
    data/dataset_test.jsonl   -- 100 test examples (25 per class)

Usage:
    python data/process_medquad.py --medquad_path ~/MedQuAD
"""

import os
import json
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

random.seed(42)

# ── Triage level keyword rules ─────────────────────────────────────────────

EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "myocardial", "stroke", "seizure",
    "unconscious", "unresponsive", "severe bleeding", "difficulty breathing",
    "shortness of breath", "anaphylaxis", "allergic reaction", "overdose",
    "poisoning", "severe head injury", "spinal injury", "meningitis",
    "subarachnoid", "pulmonary embolism", "aortic", "sepsis", "septic shock",
    "diabetic ketoacidosis", "severe burns", "drowning", "choking",
    "severe chest", "crushing chest", "cannot breathe", "stopped breathing",
]

URGENT_KEYWORDS = [
    "high fever", "fever above", "fever over", "confusion", "disoriented",
    "fracture", "broken bone", "severe pain", "dehydration", "vomiting blood",
    "blood in stool", "rectal bleeding", "severe abdominal", "appendicitis",
    "kidney stone", "urinary retention", "severe migraine", "sudden vision",
    "sudden hearing", "cellulitis", "deep wound", "laceration", "abscess",
    "severe infection", "pneumonia", "asthma attack", "severe rash",
    "high blood pressure", "hypertensive",
]

SEMI_URGENT_KEYWORDS = [
    "ear pain", "earache", "ear infection", "eye infection", "conjunctivitis",
    "urinary tract", "uti", "bladder infection", "moderate pain", "sprain",
    "mild fever", "sore throat", "tonsillitis", "skin infection", "wound",
    "minor cut", "insect bite", "rash", "back pain", "joint pain",
    "muscle pain", "headache", "migraine", "nausea", "vomiting",
    "diarrhoea", "diarrhea", "constipation", "indigestion", "heartburn",
]

NON_URGENT_KEYWORDS = [
    "cold", "runny nose", "sneezing", "mild cough", "fatigue", "tiredness",
    "routine", "prescription", "repeat prescription", "check-up", "checkup",
    "follow-up", "follow up", "vaccination", "vaccine", "immunisation",
    "minor", "mild", "slight", "general", "advice", "information",
    "prevention", "lifestyle", "diet", "exercise",
]

SYMPTOM_FILTER_KEYWORDS = [
    "symptom", "signs", "present", "feel", "pain", "ache", "fever",
    "breathing", "chest", "dizzy", "bleeding", "headache", "nausea",
    "vomit", "rash", "swelling", "infection", "injury", "hurt",
    "cough", "fatigue", "weakness", "vision", "hearing", "confusion",
    "unconscious", "severe", "acute", "sudden", "chronic",
]

INSTRUCTION = (
    "You are a medical triage assistant. Classify the following patient "
    "presentation into the correct triage level and provide a brief clinical "
    "rationale. Triage levels: EMERGENCY, URGENT, SEMI-URGENT, NON-URGENT.\n\n"
    "EMERGENCY: Life-threatening — call 999 immediately.\n"
    "URGENT: Serious — same-day care required.\n"
    "SEMI-URGENT: Needs care within 24-48 hours.\n"
    "NON-URGENT: Routine — GP appointment within a week."
)

RATIONALE_TEMPLATES = {
    "EMERGENCY": [
        "Symptoms indicate a potentially life-threatening emergency. "
        "Immediate emergency services required. Call 999 now.",
        "Presentation is consistent with a serious acute condition requiring "
        "immediate intervention. Do not wait — call 999.",
        "These symptoms require immediate emergency assessment. "
        "Call 999 or go directly to A&E.",
    ],
    "URGENT": [
        "Symptoms require same-day medical assessment. "
        "Attend A&E or urgent care today.",
        "This presentation needs prompt medical attention within hours. "
        "Contact your GP urgently or attend urgent care.",
        "Condition requires same-day review. "
        "Call 111 for advice or attend urgent care.",
    ],
    "SEMI-URGENT": [
        "Symptoms should be assessed by a GP within 24-48 hours. "
        "Call your GP surgery to arrange an appointment.",
        "This presentation requires medical review but is not immediately urgent. "
        "Book a GP appointment within the next 1-2 days.",
        "A GP appointment within 48 hours is recommended. "
        "Call 111 if symptoms worsen before then.",
    ],
    "NON-URGENT": [
        "Symptoms are mild and do not require urgent attention. "
        "Book a routine GP appointment at your convenience.",
        "This is a non-urgent presentation. "
        "A routine GP appointment within the week is appropriate.",
        "No immediate medical attention required. "
        "Book a routine appointment with your GP.",
    ],
}


def classify_triage(text: str) -> str | None:
    """Classify text into triage level using keyword matching."""
    text_lower = text.lower()

    if any(kw in text_lower for kw in EMERGENCY_KEYWORDS):
        return "EMERGENCY"
    if any(kw in text_lower for kw in URGENT_KEYWORDS):
        return "URGENT"
    if any(kw in text_lower for kw in SEMI_URGENT_KEYWORDS):
        return "SEMI-URGENT"
    if any(kw in text_lower for kw in NON_URGENT_KEYWORDS):
        return "NON-URGENT"
    return None


def is_symptom_related(text: str) -> bool:
    """Check if text is related to symptoms or presentations."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in SYMPTOM_FILTER_KEYWORDS)


def build_output(triage_level: str, answer_text: str) -> str:
    """Build the instruction-tuning output field."""
    rationale = random.choice(RATIONALE_TEMPLATES[triage_level])
    # Truncate answer text to keep output concise
    answer_snippet = answer_text[:300].strip()
    if len(answer_text) > 300:
        answer_snippet += "..."
    return (
        f"Triage Level: {triage_level}\n\n"
        f"{rationale}\n\n"
        f"Clinical context: {answer_snippet}"
    )


def parse_medquad_xml(xml_path: Path) -> list[dict]:
    """Parse a single MedQuAD XML file and return Q&A pairs."""
    pairs = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for qa_pair in root.findall(".//QAPair"):
            question_el = qa_pair.find("Question")
            answer_el   = qa_pair.find("Answer")

            if question_el is None or answer_el is None:
                continue

            question = (question_el.text or "").strip()
            answer   = (answer_el.text   or "").strip()

            if not question or not answer:
                continue
            if len(question) < 20 or len(answer) < 30:
                continue

            pairs.append({"question": question, "answer": answer})

    except ET.ParseError:
        pass

    return pairs


def collect_all_pairs(medquad_path: Path) -> list[dict]:
    """Walk MedQuAD directory and collect all XML Q&A pairs."""
    all_pairs = []
    xml_files = list(medquad_path.rglob("*.xml"))
    print(f"Found {len(xml_files)} XML files in {medquad_path}")

    for xml_file in xml_files:
        pairs = parse_medquad_xml(xml_file)
        all_pairs.extend(pairs)

    print(f"Extracted {len(all_pairs)} raw Q&A pairs")
    return all_pairs


def filter_and_classify(pairs: list[dict]) -> dict[str, list[dict]]:
    """Filter for symptom-related pairs and classify into triage levels."""
    classified = {
        "EMERGENCY":   [],
        "URGENT":      [],
        "SEMI-URGENT": [],
        "NON-URGENT":  [],
    }

    for pair in pairs:
        combined = pair["question"] + " " + pair["answer"]

        if not is_symptom_related(combined):
            continue

        level = classify_triage(combined)
        if level is None:
            continue

        classified[level].append(pair)

    for level, items in classified.items():
        print(f"  {level}: {len(items)} examples before balancing")

    return classified


def build_examples(classified: dict[str, list[dict]],
                   train_per_class: int = 375,
                   test_per_class: int = 25) -> tuple[list, list]:
    """Build balanced train and test sets."""
    train_examples = []
    test_examples  = []

    for level, pairs in classified.items():
        needed = train_per_class + test_per_class

        if len(pairs) < needed:
            print(f"  WARNING: {level} only has {len(pairs)} examples "
                  f"(need {needed}). Using all available.")
            sample = pairs
        else:
            sample = random.sample(pairs, needed)

        for pair in sample[:test_per_class]:
            test_examples.append({
                "instruction": INSTRUCTION,
                "input":       pair["question"],
                "output":      build_output(level, pair["answer"]),
                "triage_level": level,
            })

        for pair in sample[test_per_class:test_per_class + train_per_class]:
            train_examples.append({
                "instruction": INSTRUCTION,
                "input":       pair["question"],
                "output":      build_output(level, pair["answer"]),
                "triage_level": level,
            })

    random.shuffle(train_examples)
    random.shuffle(test_examples)
    return train_examples, test_examples


def write_jsonl(examples: list[dict], path: Path) -> None:
    """Write examples to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {path}")


def main():
    parser = argparse.ArgumentParser(description="Process MedQuAD into triage dataset")
    parser.add_argument("--medquad_path", type=str,
                        default=str(Path.home() / "MedQuAD"),
                        help="Path to cloned MedQuAD repository")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for JSONL files")
    parser.add_argument("--train_per_class", type=int, default=375)
    parser.add_argument("--test_per_class",  type=int, default=25)
    args = parser.parse_args()

    medquad_path = Path(args.medquad_path)
    output_dir   = Path(args.output_dir)

    if not medquad_path.exists():
        raise FileNotFoundError(
            f"MedQuAD not found at {medquad_path}. "
            f"Clone it with: git clone https://github.com/abachaa/MedQuAD.git"
        )

    print("\n── Step 1: Collecting Q&A pairs from MedQuAD ──")
    all_pairs = collect_all_pairs(medquad_path)

    print("\n── Step 2: Filtering and classifying ──")
    classified = filter_and_classify(all_pairs)

    print("\n── Step 3: Building balanced train/test sets ──")
    train_examples, test_examples = build_examples(
        classified,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
    )

    print("\n── Step 4: Writing JSONL files ──")
    write_jsonl(train_examples, output_dir / "dataset_train.jsonl")
    write_jsonl(test_examples,  output_dir / "dataset_test.jsonl")

    print(f"\n✅ Dataset ready.")
    print(f"   Training: {len(train_examples)} examples")
    print(f"   Test:     {len(test_examples)} examples")

    # Print class distribution
    from collections import Counter
    train_dist = Counter(ex["triage_level"] for ex in train_examples)
    test_dist  = Counter(ex["triage_level"] for ex in test_examples)
    print(f"\n   Train distribution: {dict(train_dist)}")
    print(f"   Test distribution:  {dict(test_dist)}")


if __name__ == "__main__":
    main()