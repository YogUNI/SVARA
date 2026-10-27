"""Comprehensive Indonesian Smart Home NLP Corpus Generator (Target: 4,000 - 5,000 Phrases).

Task: Sub-step 2.1
Reference: docs/MATRIKS_NLP_DAN_SLANG_INDONESIA_4K.md, docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md, AGENTS.md

Generates a rich, colloquial, diverse, and linguistically balanced Indonesian spoken command corpus
across all 31 Fluent Speech Commands (FSC) intents using a 6-pattern syntactic permutation engine.
"""

import os
import json
import re
import random
from typing import Dict, List, Set, Tuple
import pandas as pd

# Set fixed seed for scientific reproducibility (Rule 5 AGENTS.md)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

CONFIG_INTENT_MAP_PATH = "configs/intent_map.json"
OUTPUT_CORPUS_CSV_PATH = "data/corpus/indonesian_smart_home_corpus_4k.csv"
TARGET_PHRASES_PER_INTENT = 140  # 31 intents * 140 ~ 4,340 phrases


# =====================================================================
# 1. COMPREHENSIVE LINGUISTIC LEXICON MATRICES (docs/MATRIKS_NLP_DAN_SLANG_INDONESIA_4K.md)
# =====================================================================

ACTION_LEXICON = {
    "activate": {
        "base": ["nyalakan", "hidupkan", "putar", "setel", "pasang", "aktifkan"],
        "colloquial": ["nyalain", "hidupin", "idupin", "play", "setelin", "puter", "on-in", "start", "buka"],
        "passive": ["dinyalakan", "dihidupkan", "diputar", "disetel", "dinyalain", "diidupin", "di-on-in", "diaktifkan"],
        "short": ["nyala", "on", "hidup", "play", "start", "idup"],
    },
    "deactivate": {
        "base": ["matikan", "padamkan", "hentikan", "nonaktifkan", "tutup"],
        "colloquial": ["matiin", "padamin", "stop", "off-in", "cut", "matiin dong", "jangan nyala", "berhentiin"],
        "passive": ["dimatikan", "dipadamkan", "dihentikan", "dimatiin", "dipadamin", "di-off-in", "dinonaktifkan"],
        "short": ["mati", "off", "padam", "stop", "cut"],
    },
    "increase": {
        "base": ["naikkan", "perbesar", "besarkan", "tambahkan", "tinggikan"],
        "colloquial": ["naikin", "gedein", "tambahin", "kuatin", "besarin", "tinggiin", "angkat", "pol-in", "panasin", "hangatkan", "lebihin"],
        "passive": ["dinaikkan", "dibesarkan", "ditambahkan", "dinaikin", "digedein", "dikuatin", "ditinggikan"],
        "short": ["naik", "up", "tambah", "gedein", "plus", "kencengin"],
    },
    "decrease": {
        "base": ["turunkan", "kecilkan", "kurangkan", "kurangi", "rendahkan"],
        "colloquial": ["turunin", "kecilin", "redupin", "dinginin", "sejukin", "kurangin", "bikin redup", "ademin", "pelanin"],
        "passive": ["diturunkan", "dikecilkan", "dikurangi", "diturunin", "dikecilin", "diredupin", "didinginkan"],
        "short": ["turun", "down", "kurang", "kecilin", "minus", "pelan"],
    },
    "bring": {
        "base": ["bawakan", "ambilkan", "antarkan", "jemputkan"],
        "colloquial": ["bawain", "ambilin", "anterin", "jemput", "bawain ke sini", "tolong bawain", "ambilin dong"],
        "passive": ["dibawakan", "diambilkan", "diantarkan", "dibawain", "diambilin", "dianterin"],
        "short": ["bawa", "ambil", "antar", "bawain"],
    },
    "change language": {
        "base": ["ganti bahasa", "ubah bahasa", "pindah bahasa", "setel bahasa"],
        "colloquial": ["gantiin bahasa", "switch bahasa", "switch ke", "ganti ke", "ubah ke", "pindah ke", "setel ke bahasa"],
        "passive": ["diganti bahasanya ke", "diubah ke bahasa", "diswitch ke", "diganti ke"],
        "short": ["ganti bahasa", "switch", "ubah bahasa"],
    },
}

OBJECT_LEXICON = {
    "lights": [
        "lampu", "lampu-lampu", "lampu ruangan", "penerangan", "lampu utama",
        "bohlam", "cahaya", "lampu tengah", "penerangan ruangan", "lampunya",
        "semua lampu", "lampu gantung", "penerangan utama"
    ],
    "lamp": [
        "lampu meja", "lampu tidur", "lampu belajar", "lampu baca", "lampu tidur sudut",
        "lamp meja", "night lamp", "lampu samping", "lampu kecil", "lampu nakas",
        "lampu hias", "lampu duduk", "lampu stand", "lampu pojok", "lampu samping kasur"
    ],
    "music": [
        "musik", "lagu", "audio", "track", "playlist", "lagunya",
        "musiknya", "suara lagu", "mp3", "pemutar musik", "alunan musik", "daftar lagu"
    ],
    "volume": [
        "volume", "suara", "bunyi", "sound", "volumenya",
        "suaranya", "bunyinya", "kekerasan suara", "audio volume", "level suara"
    ],
    "heat_increase": [
        "suhu", "temperatur", "panas", "penghangat", "hawa", "suhunya",
        "temperaturnya", "hawa ruangan", "pemanas", "heater", "suhu ruangan", "pemanas ruangan"
    ],
    "heat_decrease": [
        "suhu", "temperatur", "ac", "pendingin", "hawa", "suhunya",
        "temperaturnya", "ac-nya", "pendingin ruangan", "air conditioner", "hawa sejuk", "suhu ruangan"
    ],
    "juice": [
        "jus", "jus jeruk", "jus buah", "minuman jus", "jus dingin",
        "sari buah", "jus segar", "segelas jus", "jusnya", "minuman sari buah", "segelas jus jeruk"
    ],
    "newspaper": [
        "koran", "koran pagi", "koran hari ini", "surat kabar", "koran tempo",
        "koran kompas", "harian", "korannya", "bacaan koran", "lembaran koran"
    ],
    "shoes": [
        "sepatu", "sepatu kerja", "sepatu jalan", "sneakers", "alas kaki",
        "sepatunya", "sepatu kets", "sepasang sepatu", "sepatu olahraga"
    ],
    "socks": [
        "kaos kaki", "kaus kaki", "kaos kaki kerja", "kaos kakinya",
        "sepasang kaos kaki", "kaos kaki bersih", "kaos kaki putih", "kaus kakiku"
    ],
    "Chinese": [
        "bahasa mandarin", "mandarin", "bahasa cina", "chinese", "bahasa tiongkok", "mandarinnya"
    ],
    "English": [
        "bahasa inggris", "inggris", "english", "bahasa inggrisnya"
    ],
    "German": [
        "bahasa jerman", "jerman", "german", "bahasa jermannya"
    ],
    "Korean": [
        "bahasa korea", "korea", "korean", "hangul", "bahasa koreanya"
    ],
    "none": [
        "bahasa", "bahasanya", "setelan bahasa", "bahasa lain", "opsi bahasa"
    ],
}

LOCATION_LEXICON = {
    "kitchen": [
        "dapur", "di dapur", "ke dapur", "ruang masak", "tempat masak",
        "area dapur", "kitchen", "di area dapur", "pada dapur", "ruangan dapur"
    ],
    "bedroom": [
        "kamar tidur", "di kamar tidur", "kamar", "di kamar", "ke kamar tidur",
        "ruang tidur", "kamar utama", "bedroom", "ke kamar", "area kamar tidur"
    ],
    "washroom": [
        "kamar mandi", "di kamar mandi", "toilet", "di toilet", "wc",
        "di wc", "kamar kecil", "washroom", "restroom", "ke kamar mandi"
    ],
    "none": [
        "", "di sini", "sekarang", "ruangan ini", "semuanya", "di rumah"
    ],
}

OPENER_PARTICLES = [
    "", "tolong", "coba", "minta tolong", "bisa tolong", "eh", "bro",
    "svara", "punten", "bisa tolongin", "plis", "hei svara", "tolong dong",
    "coba tolong", "eh tolong", "tolong ya"
]

CLOSING_PARTICLES = [
    "", "dong", "deh", "lah", "sih", "ya", "kan", "plis", "buruan",
    "cepetan", "ya bro", "dong tolong", "sekarang", "nih", "ya tolong"
]

INTENSITY_ADVERBS = [
    "", "dikit", "sedikit", "banyak", "agak", "rada", "pol", "banget", "perlahan"
]

SITUATIONAL_CONTEXTS = {
    "dark": ["gelap banget nih", "terlalu gelap", "gelap di sini", "kurang terang", "remang-remang"],
    "bright": ["terlalu terang", "silau banget nih", "silau", "terang banget"],
    "hot": ["gerah banget nih", "panas banget", "gerah di ruangan ini", "kepanasan nih", "hawa panas"],
    "cold": ["dingin banget", "kedinginan nih", "hawa dingin", "menggigil nih", "terlalu dingin"],
    "noisy": ["berisik banget", "kekencengan suaranya", "terlalu kencang suaranya", "bising"],
    "quiet": ["gak kedengeran", "kurang jelas suaranya", "terlalu pelan", "gak kedengaran lagunya"],
    "thirsty": ["haus nih", "pengen minum", "tenggorokan kering nih", "butuh yang segar-segar"],
    "morning": ["pagi-pagi mau baca berita", "mau baca berita", "waktunya baca kabar"],
    "outfit": ["mau berangkat kerja", "mau keluar rumah", "siap-siap pergi", "mau jalan nih"],
}


# =====================================================================
# 2. SYNTACTIC PERMUTATION ENGINE (6 PATTERNS)
# =====================================================================

def clean_sentence(text: str) -> str:
    """Cleans whitespace, double spaces, and normalizes casing."""
    text = re.sub(r"\s+", " ", text).strip()
    # Strip unnecessary trailing commas or double particles
    text = re.sub(r"\s+,", ",", text)
    # Lowercase
    return text.lower()


def get_objects_for_intent(action: str, obj: str) -> List[str]:
    """Selects appropriate object variants depending on action context (e.g. heat increase vs decrease)."""
    if obj == "heat":
        if action == "increase":
            return OBJECT_LEXICON["heat_increase"]
        else:
            return OBJECT_LEXICON["heat_decrease"]
    return OBJECT_LEXICON.get(obj, [obj])


def generate_candidates_for_intent(
    action: str,
    obj: str,
    location: str,
) -> Set[str]:
    """Generates a massive set of candidate Indonesian phrases across 6 syntactic patterns."""
    candidates: Set[str] = set()

    act_dict = ACTION_LEXICON.get(action, {})
    base_verbs = act_dict.get("base", [action])
    colloquial_verbs = act_dict.get("colloquial", [action])
    passive_verbs = act_dict.get("passive", [action])
    short_verbs = act_dict.get("short", [action])

    all_verbs = list(set(base_verbs + colloquial_verbs))
    objects = get_objects_for_intent(action, obj)
    locations = LOCATION_LEXICON.get(location, [""])

    # Determine situational contexts matching this intent
    contexts = []
    if obj in ["lights", "lamp"]:
        if action == "activate":
            contexts = SITUATIONAL_CONTEXTS["dark"]
        else:
            contexts = SITUATIONAL_CONTEXTS["bright"]
    elif obj == "heat":
        if action == "increase":
            contexts = SITUATIONAL_CONTEXTS["cold"]
        else:
            contexts = SITUATIONAL_CONTEXTS["hot"]
    elif obj in ["volume", "music"]:
        if action in ["increase", "activate"]:
            contexts = SITUATIONAL_CONTEXTS["quiet"]
        else:
            contexts = SITUATIONAL_CONTEXTS["noisy"]
    elif obj == "juice":
        contexts = SITUATIONAL_CONTEXTS["thirsty"]
    elif obj == "newspaper":
        contexts = SITUATIONAL_CONTEXTS["morning"]
    elif obj in ["shoes", "socks"]:
        contexts = SITUATIONAL_CONTEXTS["outfit"]

    # -------------------------------------------------------------
    # Pola 1: Standar Formal [Pembuka] + Aksi + Objek + Lokasi + [Partikel]
    # -------------------------------------------------------------
    for opener in OPENER_PARTICLES[:8]:
        for verb in base_verbs:
            for o in objects[:6]:
                for loc in locations[:5]:
                    for part in CLOSING_PARTICLES[:6]:
                        phrase = f"{opener} {verb} {o} {loc} {part}"
                        candidates.add(clean_sentence(phrase))

    # -------------------------------------------------------------
    # Pola 2: Santai dengan Partikel [Pembuka] + Aksi + Partikel + Objek + Lokasi
    # -------------------------------------------------------------
    for opener in OPENER_PARTICLES[:6]:
        for verb in colloquial_verbs:
            for part in ["dong", "deh", "lah", "ya", "kan", "cepetan", ""]:
                for o in objects[:7]:
                    for loc in locations[:5]:
                        phrase = f"{opener} {verb} {part} {o} {loc}"
                        candidates.add(clean_sentence(phrase))

    # -------------------------------------------------------------
    # Pola 3: Objek di Depan (Objek + Lokasi + [Tolong] + Aksi Pasif/Aktif + [Partikel])
    # -------------------------------------------------------------
    for o in objects[:8]:
        for loc in locations[:5]:
            for req in ["tolong", "bisa tolong", "coba", "minta tolong", ""]:
                for p_verb in passive_verbs + colloquial_verbs[:3]:
                    for part in ["ya", "dong", "deh", "lah", ""]:
                        phrase = f"{o} {loc} {req} {p_verb} {part}"
                        candidates.add(clean_sentence(phrase))

    # -------------------------------------------------------------
    # Pola 4: Lokasi di Depan (Lokasi + Objek + [Tolong] + Aksi + [Partikel])
    # -------------------------------------------------------------
    if location != "none":
        for loc in locations[:6]:
            for o in objects[:6]:
                for opener in ["tolong", "coba", "eh", ""]:
                    for verb in all_verbs[:6]:
                        for part in ["dong", "ya", "deh", ""]:
                            phrase = f"{loc} {o} {opener} {verb} {part}"
                            candidates.add(clean_sentence(phrase))

    # -------------------------------------------------------------
    # Pola 5: Implisit / Kontekstual (Kondisi + [Tolong] + Aksi + Objek + Lokasi)
    # -------------------------------------------------------------
    if contexts:
        for ctx in contexts:
            for opener in ["tolong", "coba", "bisa tolong", ""]:
                for verb in all_verbs[:5]:
                    for o in objects[:4]:
                        for loc in locations[:3]:
                            phrase = f"{ctx} {opener} {verb} {o} {loc}"
                            candidates.add(clean_sentence(phrase))

    # -------------------------------------------------------------
    # Pola 6: Singkat / Quick Commands (Objek + Lokasi + Aksi Singkat)
    # -------------------------------------------------------------
    for o in objects[:5]:
        for loc in locations[:4]:
            for s_verb in short_verbs:
                phrase1 = f"{o} {loc} {s_verb}"
                phrase2 = f"{s_verb} {o} {loc}"
                candidates.add(clean_sentence(phrase1))
                candidates.add(clean_sentence(phrase2))

    return candidates


# =====================================================================
# 3. CORPUS EXPANSION GENERATOR RUNNER
# =====================================================================

def generate_massive_corpus(
    intent_map_path: str = CONFIG_INTENT_MAP_PATH,
    output_csv_path: str = OUTPUT_CORPUS_CSV_PATH,
    target_per_intent: int = TARGET_PHRASES_PER_INTENT,
) -> pd.DataFrame:
    """Generates, filters, deduplicates, and exports the massive 4,000+ phrase corpus."""
    if not os.path.exists(intent_map_path):
        raise FileNotFoundError(f"Intent map not found at {intent_map_path}")

    with open(intent_map_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    intent_map = config["intent_map"]
    all_rows: List[Dict[str, str]] = []

    print(f"[*] Starting Indonesian Corpus Generation for {len(intent_map)} intents...")
    print(f"[*] Target: Minimum {target_per_intent} diverse phrases per intent.")

    total_candidates_generated = 0
    intent_stats = {}

    for intent_id_str, meta in sorted(intent_map.items(), key=lambda x: int(x[0])):
        intent_id = int(intent_id_str)
        action = meta["action"]
        obj = meta["object"]
        location = meta["location"]

        candidates = generate_candidates_for_intent(action, obj, location)
        total_candidates_generated += len(candidates)

        # Filter out empty or trivially short phrases (< 2 words)
        valid_phrases = [
            p for p in candidates
            if len(p.split()) >= 2 and len(p) >= 5
        ]

        # Shuffle deterministically
        random.seed(RANDOM_SEED + intent_id)
        random.shuffle(valid_phrases)

        # Select target count (stratified balance)
        selected_phrases = valid_phrases[:target_per_intent]
        if len(selected_phrases) < target_per_intent:
            print(f"[!] Warning: Intent {intent_id} generated {len(selected_phrases)} < {target_per_intent}")

        intent_stats[intent_id] = len(selected_phrases)

        # English reference sample
        eng_loc = f"in the {location}" if location != "none" else ""
        eng_sample = f"{action} the {obj} {eng_loc}".strip()

        for phrase in selected_phrases:
            all_rows.append({
                "intent_id": intent_id,
                "action": action,
                "object": obj,
                "location": location,
                "english_sample": eng_sample,
                "indonesian_phrase": phrase,
            })

    df = pd.DataFrame(all_rows)

    # Global Deduplication check (ensure uniqueness across dataset)
    df = df.drop_duplicates(subset=["indonesian_phrase", "intent_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")

    print("\n" + "=" * 65)
    print("           RINGKASAN GENERASI KORPUS INDONESIA 4K")
    print("=" * 65)
    print(f"Total Intents                   : {len(intent_map)} (IDs 0 to 30)")
    print(f"Total Raw Candidates Generated  : {total_candidates_generated:,} variasi")
    print(f"Total Frasa Terpilih Bersih     : {len(df):,} kalimat")
    print(f"Rata-rata Frasa per Intent      : {len(df) / len(intent_map):.1f} kalimat/intent")
    print(f"Intent Min / Max Kalimat        : {min(intent_stats.values())} / {max(intent_stats.values())}")
    
    # Vocabulary statistics
    words = set()
    for phrase in df["indonesian_phrase"]:
        words.update(phrase.split())
    print(f"Total Kosakata Unik (Lexicon)   : {len(words)} kata unik")
    print(f"Lokasi File Output CSV          : {output_csv_path}")
    print("=" * 65 + "\n")

    return df


if __name__ == "__main__":
    generate_massive_corpus()
