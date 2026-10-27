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
    assert len(corpus_df) == 4495  # 31 intents * 145 phrases


def test_corpus_required_columns(corpus_df):
    """Verify that all required columns are present with non-empty content."""
    required_cols = [
        "intent_id", "action", "object", "location", "domain",
        "english_sample", "indonesian_phrase"
    ]
    for col in required_cols:
        assert col in corpus_df.columns, f"Missing required column: {col}"
        assert corpus_df[col].notna().all(), f"Found NaN values in column: {col}"


def test_corpus_stratified_distribution(corpus_df):
    """Verify perfectly balanced stratified distribution (145 phrases for each of 31 intents)."""
    intent_counts = corpus_df.groupby("intent_id").size()
    assert len(intent_counts) == EXPECTED_INTENT_COUNT
    for intent_id in range(EXPECTED_INTENT_COUNT):
        assert intent_id in intent_counts.index
        assert intent_counts[intent_id] == 145


def test_corpus_domain_distribution(corpus_df):
    """Verify smart home functional domain distribution across all 31 intents."""
    expected_domain_phrases = {
        "Lighting (Pencahayaan)": 1450,
        "Climate (Pengatur Suhu & AC)": 1160,
        "Media & Audio (Hiburan & Volume)": 580,
        "Robot Butler (Pengantaran Barang)": 580,
        "System Language (Pengaturan Bahasa)": 725,
    }
    expected_domain_intents = {
        "Lighting (Pencahayaan)": 10,
        "Climate (Pengatur Suhu & AC)": 8,
        "Media & Audio (Hiburan & Volume)": 4,
        "Robot Butler (Pengantaran Barang)": 4,
        "System Language (Pengaturan Bahasa)": 5,
    }

    domain_counts = corpus_df.groupby("domain")["intent_id"].nunique().to_dict()
    domain_phrases = corpus_df.groupby("domain").size().to_dict()

    assert domain_counts == expected_domain_intents
    assert domain_phrases == expected_domain_phrases


def test_corpus_length_statistics(corpus_df):
    """Verify that phrase word and character lengths fall within realistic spoken command ranges."""
    word_counts = corpus_df["indonesian_phrase"].apply(lambda s: len(s.split()))
    char_counts = corpus_df["indonesian_phrase"].apply(len)

    # Average spoken voice command is typically 4 to 8 words
    assert 5.0 <= word_counts.mean() <= 8.0
    assert word_counts.min() >= 2
    assert word_counts.max() <= 15

    # Character count checks
    assert 30.0 <= char_counts.mean() <= 50.0
    assert char_counts.min() >= 10
    assert char_counts.max() <= 100


def test_corpus_linguistic_diversity(corpus_df):
    """Verify the presence of everyday Indonesian slang, colloquial verbs, regional loan particles, and rich vocabulary."""
    all_text = " ".join(corpus_df["indonesian_phrase"].tolist()).lower()
    
    # Check colloquial verbs
    colloquial_verbs = ["nyalain", "matiin", "idupin", "naikin", "turunin", "bawain", "ambilin"]
    for verb in colloquial_verbs:
        assert verb in all_text, f"Colloquial verb '{verb}' not found in corpus text"
        
    # Check everyday and regional particles
    particles = ["dong", "deh", "lah", "sih", "tolong", "atuh", "bro"]
    for p in particles:
        assert p in all_text, f"Particle '{p}' not found in corpus text"

    # Check total unique vocabulary richness
    vocab = set(all_text.split())
    assert len(vocab) >= 300, f"Vocabulary size {len(vocab)} is smaller than expected 300"


def test_corpus_zero_empty_or_trivial_phrases(corpus_df):
    """Verify that all phrases have at least 2 words and minimum length of 5 characters."""
    for idx, row in corpus_df.iterrows():
        phrase = str(row["indonesian_phrase"]).strip()
        words = phrase.split()
        assert len(words) >= 2, f"Phrase at row {idx} too short: '{phrase}'"
        assert len(phrase) >= 5, f"Phrase at row {idx} too short: '{phrase}'"
