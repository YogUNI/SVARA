"""Unit tests for the massive 4,000+ Indonesian Smart Home corpus.

Task: Sub-step 2.1, Sub-step 2.2, Sub-step 2.3
Reference: docs/MATRIKS_NLP_DAN_SLANG_INDONESIA_4K.md, docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md
"""

import os
import pandas as pd
import pytest

CORPUS_CSV_PATH = "data/corpus/indonesian_smart_home_corpus_4k.csv"
EXPECTED_INTENT_COUNT = 31
EXPECTED_MIN_PHRASES = 4000


@pytest.fixture(scope="module")
def corpus_df():
    """Loads generated Indonesian smart home corpus."""
    if not os.path.exists(CORPUS_CSV_PATH):
        from scripts.generate_massive_indonesian_corpus import generate_massive_corpus
        return generate_massive_corpus()
    return pd.read_csv(CORPUS_CSV_PATH)


def test_corpus_exists_and_row_count(corpus_df):
    """Verify that corpus CSV exists and exceeds the 4,000 phrase minimum."""
    assert os.path.exists(CORPUS_CSV_PATH)
    assert len(corpus_df) >= EXPECTED_MIN_PHRASES
    assert len(corpus_df) == 4340  # 31 intents * 140 phrases


def test_corpus_required_columns(corpus_df):
    """Verify that all required columns are present with non-empty content."""
    required_cols = [
        "intent_id", "action", "object", "location",
        "english_sample", "indonesian_phrase"
    ]
    for col in required_cols:
        assert col in corpus_df.columns, f"Missing required column: {col}"
        assert corpus_df[col].notna().all(), f"Found NaN values in column: {col}"


def test_corpus_stratified_distribution(corpus_df):
    """Verify perfectly balanced stratified distribution (140 phrases for each of 31 intents)."""
    intent_counts = corpus_df.groupby("intent_id").size()
    assert len(intent_counts) == EXPECTED_INTENT_COUNT
    for intent_id in range(EXPECTED_INTENT_COUNT):
        assert intent_id in intent_counts.index
        assert intent_counts[intent_id] == 140


def test_corpus_linguistic_diversity(corpus_df):
    """Verify the presence of everyday Indonesian slang, colloquial verbs, and particles."""
    all_text = " ".join(corpus_df["indonesian_phrase"].tolist()).lower()
    
    # Check colloquial verbs
    colloquial_verbs = ["nyalain", "matiin", "idupin", "naikin", "turunin", "bawain", "ambilin"]
    for verb in colloquial_verbs:
        assert verb in all_text, f"Colloquial verb '{verb}' not found in corpus text"
        
    # Check particles
    particles = ["dong", "deh", "lah", "sih", "tolong"]
    for p in particles:
        assert p in all_text, f"Particle '{p}' not found in corpus text"

    # Check total unique vocabulary
    vocab = set(all_text.split())
    assert len(vocab) >= 200, f"Vocabulary size {len(vocab)} is smaller than expected 200"


def test_corpus_zero_empty_or_trivial_phrases(corpus_df):
    """Verify that all phrases have at least 2 words and minimum length of 5 characters."""
    for idx, row in corpus_df.iterrows():
        phrase = str(row["indonesian_phrase"]).strip()
        words = phrase.split()
        assert len(words) >= 2, f"Phrase at row {idx} too short: '{phrase}'"
        assert len(phrase) >= 5, f"Phrase at row {idx} too short: '{phrase}'"
