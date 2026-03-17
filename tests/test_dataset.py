"""
test_dataset.py
---------------
Validates dataset_train.jsonl and dataset_test.jsonl before training.
Checks schema, balance, no leakage, and output format.

Run:
    pytest tests/test_dataset.py -v
"""

import json
import pytest
from pathlib import Path
from collections import Counter

TRAIN_PATH = Path("data/dataset_train.jsonl")
TEST_PATH  = Path("data/dataset_test.jsonl")

VALID_LEVELS = {"EMERGENCY", "URGENT", "SEMI-URGENT", "NON-URGENT"}
REQUIRED_FIELDS = {"instruction", "input", "output", "triage_level"}

TRAIN_PER_CLASS = 375
TEST_PER_CLASS  = 25


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def train_data():
    return load_jsonl(TRAIN_PATH)


@pytest.fixture(scope="module")
def test_data():
    return load_jsonl(TEST_PATH)


# ── Schema tests ───────────────────────────────────────────────────────────

class TestSchema:

    def test_train_file_exists(self):
        assert TRAIN_PATH.exists(), f"Training file not found: {TRAIN_PATH}"

    def test_test_file_exists(self):
        assert TEST_PATH.exists(), f"Test file not found: {TEST_PATH}"

    def test_train_required_fields(self, train_data):
        for i, ex in enumerate(train_data):
            missing = REQUIRED_FIELDS - set(ex.keys())
            assert not missing, f"Train example {i} missing fields: {missing}"

    def test_test_required_fields(self, test_data):
        for i, ex in enumerate(test_data):
            missing = REQUIRED_FIELDS - set(ex.keys())
            assert not missing, f"Test example {i} missing fields: {missing}"

    def test_train_no_empty_fields(self, train_data):
        for i, ex in enumerate(train_data):
            for field in REQUIRED_FIELDS:
                assert ex[field].strip(), \
                    f"Train example {i} has empty field: {field}"

    def test_test_no_empty_fields(self, test_data):
        for i, ex in enumerate(test_data):
            for field in REQUIRED_FIELDS:
                assert ex[field].strip(), \
                    f"Test example {i} has empty field: {field}"


# ── Triage level tests ─────────────────────────────────────────────────────

class TestTriageLevels:

    def test_train_valid_levels(self, train_data):
        for i, ex in enumerate(train_data):
            assert ex["triage_level"] in VALID_LEVELS, \
                f"Train example {i} has invalid triage_level: {ex['triage_level']}"

    def test_test_valid_levels(self, test_data):
        for i, ex in enumerate(test_data):
            assert ex["triage_level"] in VALID_LEVELS, \
                f"Test example {i} has invalid triage_level: {ex['triage_level']}"

    def test_output_starts_with_triage_level(self, train_data):
        for i, ex in enumerate(train_data):
            assert ex["output"].startswith("Triage Level:"), \
                f"Train example {i} output does not start with 'Triage Level:'"

    def test_output_contains_correct_level(self, train_data):
        for i, ex in enumerate(train_data):
            level = ex["triage_level"]
            assert level in ex["output"], \
                f"Train example {i} output does not contain triage_level '{level}'"


# ── Balance tests ──────────────────────────────────────────────────────────

class TestBalance:

    def test_train_total_count(self, train_data):
        assert len(train_data) == 1500, \
            f"Expected 1500 training examples, got {len(train_data)}"

    def test_test_total_count(self, test_data):
        assert len(test_data) == 100, \
            f"Expected 100 test examples, got {len(test_data)}"

    def test_train_class_balance(self, train_data):
        dist = Counter(ex["triage_level"] for ex in train_data)
        for level in VALID_LEVELS:
            assert dist[level] == TRAIN_PER_CLASS, \
                f"Train class '{level}' has {dist[level]} examples, expected {TRAIN_PER_CLASS}"

    def test_test_class_balance(self, test_data):
        dist = Counter(ex["triage_level"] for ex in test_data)
        for level in VALID_LEVELS:
            assert dist[level] == TEST_PER_CLASS, \
                f"Test class '{level}' has {dist[level]} examples, expected {TEST_PER_CLASS}"


# ── Data leakage test ──────────────────────────────────────────────────────

class TestNoLeakage:

    def test_no_train_examples_in_test(self, train_data, test_data):
        train_inputs = {ex["input"] for ex in train_data}
        test_inputs  = {ex["input"] for ex in test_data}
        overlap = train_inputs & test_inputs
        assert not overlap, \
            f"Data leakage: {len(overlap)} inputs appear in both train and test"

    def test_no_duplicate_train_inputs(self, train_data):
        inputs = [ex["input"] for ex in train_data]
        duplicates = len(inputs) - len(set(inputs))
        assert duplicates == 0, \
            f"Found {duplicates} duplicate inputs in training set"

    def test_no_duplicate_test_inputs(self, test_data):
        inputs = [ex["input"] for ex in test_data]
        duplicates = len(inputs) - len(set(inputs))
        assert duplicates == 0, \
            f"Found {duplicates} duplicate inputs in test set"


# ── Content quality tests ──────────────────────────────────────────────────

class TestContentQuality:

    def test_train_input_min_length(self, train_data):
        short = [i for i, ex in enumerate(train_data) if len(ex["input"]) < 20]
        assert not short, \
            f"Train examples with input < 20 chars: indices {short[:5]}"

    def test_test_input_min_length(self, test_data):
        short = [i for i, ex in enumerate(test_data) if len(ex["input"]) < 20]
        assert not short, \
            f"Test examples with input < 20 chars: indices {short[:5]}"

    def test_train_output_min_length(self, train_data):
        short = [i for i, ex in enumerate(train_data) if len(ex["output"]) < 50]
        assert not short, \
            f"Train examples with output < 50 chars: indices {short[:5]}"

    def test_all_levels_represented_in_train(self, train_data):
        levels = {ex["triage_level"] for ex in train_data}
        assert levels == VALID_LEVELS, \
            f"Not all triage levels in training set. Found: {levels}"

    def test_all_levels_represented_in_test(self, test_data):
        levels = {ex["triage_level"] for ex in test_data}
        assert levels == VALID_LEVELS, \
            f"Not all triage levels in test set. Found: {levels}"