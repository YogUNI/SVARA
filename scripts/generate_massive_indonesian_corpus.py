"""Comprehensive Indonesian Smart Home NLP Corpus Generator (Target: 4,000 - 5,000 Phrases).

Task: Sub-step 2.1, Sub-step 2.2
Reference: docs/MATRIKS_NLP_DAN_SLANG_INDONESIA_4K.md, docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md, AGENTS.md

Generates an exhaustive, super-complete Indonesian spoken command corpus across all 31 Fluent Speech Commands (FSC)
intents, covering everyday conversational slang, regional loan particles (atuh, cuy, lur, bro, min), polite requests,
vocatives, situational complaints, and 6 flexible syntactic permutations.
"""

import os
import json
import random
import argparse
from typing import Dict, List, Set
import pandas as pd

# Fixed seed for scientific reproducibility (Rule 5 AGENTS.md)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

CONFIG_INTENT_MAP_PATH = "configs/intent_map.json"
OUTPUT_CORPUS_CSV_PATH = "data/corpus/indonesian_smart_home_corpus_4k.csv"
TARGET_PHRASES_PER_INTENT = 145  # 31 intents * 145 = 4,495 phrases


# =====================================================================
# 1. EXHAUSTIVE LINGUISTIC LEXICON MATRICES
# =====================================================================

ACTION_LEXICON = {
    "activate": {
        "base": [
            "nyalakan", "hidupkan", "putar", "setel", "pasang", "aktifkan", "buka", "pasangkan"
        ],
        "colloquial": [
            "nyalain", "hidupin", "idupin", "play", "setelin", "puter", "on-in", "start",
            "colokin", "pencet on", "geber", "gas", "idupkanlah", "nyalakanlah", "pasangin"
        ],
        "passive": [
            "dinyalakan", "dihidupkan", "diputar", "disetel", "dinyalain", "diidupin",
            "di-on-in", "diaktifkan", "dipasang", "dibuka"
        ],
        "short": [
            "nyala", "on", "hidup", "idup", "play", "start"
        ],
    },
    "deactivate": {
        "base": [
            "matikan", "padamkan", "hentikan", "nonaktifkan", "tutup"
        ],
        "colloquial": [
            "matiin", "padamin", "stop", "off-in", "cut", "cabut", "berhentiin",
            "pencet off", "jangan nyala", "padamkanlah", "matiinlah", "bikin mati", "shutdown"
        ],
        "passive": [
            "dimatikan", "dipadamkan", "dihentikan", "dimatiin", "dipadamin",
            "di-off-in", "dinonaktifkan", "ditutup", "dicabut"
        ],
        "short": [
            "mati", "off", "padam", "stop", "cut", "tutup"
        ],
    },
    "increase": {
        "base": [
            "naikkan", "perbesar", "besarkan", "tambahkan", "tinggikan", "kuatkan"
        ],
        "colloquial": [
            "naikin", "gedein", "tambahin", "kuatin", "besarin", "tinggiin", "angkat",
            "pol-in", "panasin", "hangatkan", "lebihin", "kencengin", "kencangin", "gaskan"
        ],
        "passive": [
            "dinaikkan", "dibesarkan", "ditambahkan", "dinaikin", "digedein",
            "dikuatin", "ditinggikan", "dikencengin", "dilebihkan"
        ],
        "short": [
            "naik", "up", "tambah", "gedein", "kenceng", "plus"
        ],
    },
    "decrease": {
        "base": [
            "turunkan", "kecilkan", "kurangkan", "kurangi", "rendahkan", "pelankan"
        ],
        "colloquial": [
            "turunin", "kecilin", "redupin", "dinginin", "sejukin", "kurangin",
            "bikin redup", "ademin", "pelanin", "rem", "buat dingin", "turunkan dikit"
        ],
        "passive": [
            "diturunkan", "dikecilkan", "dikurangi", "diturunin", "dikecilin",
            "diredupin", "didinginkan", "dipelanin", "disejukkan"
        ],
        "short": [
            "turun", "down", "kurang", "kecilin", "pelan", "minus", "redup"
        ],
    },
    "bring": {
        "base": [
            "bawakan", "ambilkan", "antarkan", "jemputkan", "berikan"
        ],
        "colloquial": [
            "bawain", "ambilin", "anterin", "jemput", "bawain ke sini", "anterin ke sini",
            "ambilin dong", "comot", "kasihin", "bawa sini", "ambil sini", "tolong bawain"
        ],
        "passive": [
            "dibawakan", "diambilkan", "diantarkan", "dibawain", "diambilin",
            "dianterin", "dikasihkan", "dijemput"
        ],
        "short": [
            "bawa", "ambil", "antar", "bawain", "ambilin"
        ],
    },
    "change language": {
        "base": [
            "ganti bahasa", "ubah bahasa", "pindah bahasa", "setel bahasa"
        ],
        "colloquial": [
            "gantiin bahasa", "switch bahasa", "switch ke", "ganti ke", "ubah ke",
            "pindah ke", "setel ke bahasa", "atur bahasa ke", "ganti bahasanya ke", "alihkan ke bahasa"
        ],
        "passive": [
            "diganti bahasanya ke", "diubah ke bahasa", "diswitch ke", "diganti ke", "dialihkan ke"
        ],
        "short": [
            "ganti bahasa", "switch", "ubah bahasa", "pindah bahasa"
        ],
    },
}

OBJECT_LEXICON = {
    "lights": [
        "lampu", "lampu-lampu", "lampu ruangan", "penerangan", "lampu utama",
        "bohlam", "bohlamnya", "cahaya", "lampu tengah", "penerangan ruangan",
        "lampunya", "semua lampu", "lampu gantung", "penerangan utama", "lampu atas",
        "lampu plafon", "lampu penerangan"
    ],
    "lamp": [
        "lampu meja", "lampu tidur", "lampu belajar", "lampu baca", "lampu tidur sudut",
        "lamp meja", "night lamp", "lampu samping", "lampu kecil", "lampu nakas",
        "lampu hias", "lampu duduk", "lampu stand", "lampu pojok", "lampu samping kasur",
        "lampu tidur samping", "lampu baca sudut"
    ],
    "music": [
        "musik", "lagu", "audio", "track", "playlist", "lagunya",
        "musiknya", "suara lagu", "mp3", "pemutar musik", "alunan musik",
        "daftar lagu", "setelan musik", "alunan lagu", "tembang", "sound"
    ],
    "volume": [
        "volume", "suara", "bunyi", "sound", "volumenya",
        "suaranya", "bunyinya", "kekerasan suara", "audio volume", "level suara",
        "sound volume", "speaker", "volume speaker", "kencengnya suara"
    ],
    "heat_increase": [
        "suhu", "temperatur", "panas", "penghangat", "hawa", "suhunya",
        "temperaturnya", "hawa ruangan", "pemanas", "heater", "suhu ruangan",
        "pemanas ruangan", "temperatur kamar", "derajat panas", "tingkat panas"
    ],
    "heat_decrease": [
        "suhu", "temperatur", "ac", "pendingin", "hawa", "suhunya",
        "temperaturnya", "ac-nya", "pendingin ruangan", "air conditioner",
        "hawa sejuk", "suhu ruangan", "pendingin kamar", "derajat dingin", "sejuknya kamar"
    ],
    "juice": [
        "jus", "jus jeruk", "jus buah", "minuman jus", "jus dingin",
        "sari buah", "jus segar", "segelas jus", "jusnya", "minuman sari buah",
        "segelas jus jeruk", "jus manis", "minuman dingin", "jus buah segar"
    ],
    "newspaper": [
        "koran", "koran pagi", "koran hari ini", "surat kabar", "koran tempo",
        "koran kompas", "harian", "korannya", "bacaan koran", "lembaran koran",
        "koran harian", "koran terbitan hari ini", "berita koran"
    ],
    "shoes": [
        "sepatu", "sepatu kerja", "sepatu jalan", "sneakers", "alas kaki",
        "sepatunya", "sepatu kets", "sepasang sepatu", "sepatu olahraga",
        "sepatu pantofel", "sepatu kerja gue", "sepatu lari", "sneakers gue"
    ],
    "socks": [
        "kaos kaki", "kaus kaki", "kaos kaki kerja", "kaos kakinya",
        "sepasang kaos kaki", "kaos kaki bersih", "kaos kaki putih", "kaus kakiku",
        "kaos kaki abu-abu", "kaos kaki tidur", "sepasang kaus kaki"
    ],
    "Chinese": [
        "bahasa mandarin", "mandarin", "bahasa cina", "chinese", "bahasa tiongkok",
        "mandarinnya", "bahasa cina mandarin"
    ],
    "English": [
        "bahasa inggris", "inggris", "english", "bahasa inggrisnya", "bahasa bule"
    ],
    "German": [
        "bahasa jerman", "jerman", "german", "bahasa jermannya"
    ],
    "Korean": [
        "bahasa korea", "korea", "korean", "hangul", "bahasa koreanya"
    ],
    "none": [
        "bahasa", "bahasanya", "setelan bahasa", "bahasa lain", "opsi bahasa", "bahasa sistem"
    ],
}

LOCATION_LEXICON = {
    "kitchen": [
        "dapur", "di dapur", "ke dapur", "ruang masak", "tempat masak",
        "area dapur", "kitchen", "di area dapur", "pada dapur", "ruangan dapur",
        "di tempat masak", "ke ruang dapur", "dapur belakang", "pojok dapur", "di kitchen"
    ],
    "bedroom": [
        "kamar tidur", "di kamar tidur", "kamar", "di kamar", "ke kamar tidur",
        "ruang tidur", "kamar utama", "bedroom", "ke kamar", "area kamar tidur",
        "bilik", "bilik tidur", "kamar gue", "kamar saya", "ruangan tidur", "tempat tidur"
    ],
    "washroom": [
        "kamar mandi", "di kamar mandi", "toilet", "di toilet", "wc",
        "di wc", "kamar kecil", "washroom", "restroom", "ke kamar mandi",
        "ke toilet", "ke wc", "peturasan", "kamar bilas", "di kamar kecil", "di washroom"
    ],
    "none": [
        "", "di sini", "sekarang", "ruangan ini", "semuanya", "di rumah",
        "seisi rumah", "seluruhnya", "area sini", "sekitar sini", "sekarang juga"
    ],
}

OPENER_PARTICLES = [
    "", "tolong", "coba", "minta tolong", "bisa tolong", "eh", "bro", "cuy",
    "gan", "min", "bang", "mas", "mbak", "kak", "svara", "hey svara", "halo svara",
    "punten", "bisa tolongin", "plis", "please", "mohon", "bantu dong", "tolong dong",
    "coba dong", "eh tolong", "tolong ya", "mangga", "woy", "lur"
]

CLOSING_PARTICLES = [
    "", "dong", "deh", "lah", "sih", "ya", "kan", "plis", "buruan", "cepetan",
    "ya bro", "ya cuy", "ya gan", "ya bang", "ya kak", "ya lur", "dong tolong", "sekarang",
    "nih", "tuh", "atuh", "dong atuh", "deh ya", "ya tolong", "ya svara", "sekarang ya", "lur"
]

INTENSITY_ADVERBS = [
    "", "dikit", "sedikit", "banyak", "agak", "rada", "pol", "banget", "amat",
    "full", "maksimal", "perlahan", "pelan-pelan", "makin", "lebih", "kenceng"
]

SITUATIONAL_CONTEXTS = {
    "dark": [
        "gelap banget nih", "gelap gulita", "terlalu gelap", "kurang terang",
        "remang-remang", "gak keliatan apa-apa", "gelap di ruangan ini"
    ],
    "bright": [
        "terlalu terang", "silau banget nih", "silau", "terang banget",
        "sakit mata nih", "terlalu silau"
    ],
    "hot": [
        "gerah banget nih", "panas banget", "gerah di ruangan ini", "kepanasan nih",
        "hawa panas banget", "sumuk nih", "mandi keringat nih"
    ],
    "cold": [
        "dingin banget nih", "kedinginan nih", "hawa dingin menusuk", "menggigil nih",
        "terlalu dingin", "beku nih"
    ],
    "noisy": [
        "berisik banget", "kekencengan suaranya", "terlalu kencang suaranya",
        "bising amat", "budek nih", "terlalu berisik"
    ],
    "quiet": [
        "gak kedengeran", "kurang jelas suaranya", "terlalu pelan",
        "gak kedengaran lagunya", "kecil banget suaranya", "kurang kenceng nih"
    ],
    "thirsty": [
        "haus nih", "pengen minum", "tenggorokan kering nih",
        "butuh yang segar-segar", "haus banget pengen jus"
    ],
    "morning": [
        "pagi-pagi mau baca berita", "mau baca berita", "waktunya baca kabar",
        "pagi-pagi mau baca koran", "mau baca harian pagi"
    ],
    "outfit": [
        "mau berangkat kerja", "mau keluar rumah", "siap-siap pergi",
        "mau jalan nih", "mau berangkat ngantor", "mau pergi keluar"
    ],
}


# =====================================================================
# 2. FAST SYNTACTIC PERMUTATION ENGINE (6 PATTERNS)
# =====================================================================

def clean_sentence(text: str) -> str:
    """Fast whitespace cleanup and normalization without heavy regex overhead."""
    return " ".join(text.lower().split())


def get_objects_for_intent(action: str, obj: str) -> List[str]:
    """Selects appropriate object variants depending on action context."""
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

    all_verbs = list(dict.fromkeys(base_verbs + colloquial_verbs))
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
    for opener in OPENER_PARTICLES:
        for verb in base_verbs:
            for o in objects[:6]:
                for loc in locations[:5]:
                    for part in CLOSING_PARTICLES[:10]:
                        phrase = f"{opener} {verb} {o} {loc} {part}"
                        candidates.add(clean_sentence(phrase))

    # -------------------------------------------------------------
    # Pola 2: Santai dengan Partikel [Pembuka] + Aksi + Partikel + Objek + Lokasi
    # -------------------------------------------------------------
    for opener in OPENER_PARTICLES:
        for verb in colloquial_verbs:
            for part in ["dong", "deh", "lah", "ya", "kan", "cepetan", "atuh", "nih", "ya bro", "lur", "ya lur", ""]:
                for o in objects[:6]:
                    for loc in locations[:4]:
                        phrase = f"{opener} {verb} {part} {o} {loc}"
                        candidates.add(clean_sentence(phrase))

    # -------------------------------------------------------------
    # Pola 3: Objek di Depan (Objek + Lokasi + [Tolong] + Aksi Pasif/Aktif + [Partikel])
    # -------------------------------------------------------------
    for o in objects[:6]:
        for loc in locations[:4]:
            for req in ["tolong", "bisa tolong", "coba", "plis", "minta tolong", "lur", "bro", ""]:
                for p_verb in passive_verbs + colloquial_verbs[:4]:
                    for part in ["ya", "dong", "deh", "lah", "atuh", "ya bro", "lur", "ya lur", ""]:
                        phrase = f"{o} {loc} {req} {p_verb} {part}"
                        candidates.add(clean_sentence(phrase))

    # -------------------------------------------------------------
    # Pola 4: Lokasi di Depan (Lokasi + Objek + [Tolong] + Aksi + [Partikel])
    # -------------------------------------------------------------
    if location != "none":
        for loc in locations[:4]:
            for o in objects[:6]:
                for opener in ["tolong", "coba", "eh", "bisa tolong", "lur", "bro", ""]:
                    for verb in all_verbs[:6]:
                        for part in ["dong", "ya", "deh", "atuh", "ya bro", "lur", ""]:
                            phrase = f"{loc} {o} {opener} {verb} {part}"
                            candidates.add(clean_sentence(phrase))

    # -------------------------------------------------------------
    # Pola 5: Implisit / Kontekstual (Kondisi + [Tolong] + Aksi + Objek + Lokasi)
    # -------------------------------------------------------------
    if contexts:
        for ctx in contexts:
            for opener in ["tolong", "coba", "bisa tolong", "plis", ""]:
                for verb in all_verbs[:6]:
                    for o in objects[:6]:
                        for loc in locations[:4]:
                            phrase = f"{ctx} {opener} {verb} {o} {loc}"
                            candidates.add(clean_sentence(phrase))

    # -------------------------------------------------------------
    # Pola 6: Singkat / Quick Commands (Objek + Lokasi + Aksi Singkat)
    # -------------------------------------------------------------
    for o in objects[:6]:
        for loc in locations[:5]:
            for s_verb in short_verbs:
                phrase1 = f"{o} {loc} {s_verb}"
                phrase2 = f"{s_verb} {o} {loc}"
                phrase3 = f"{o} {s_verb} {loc}"
                candidates.add(clean_sentence(phrase1))
                candidates.add(clean_sentence(phrase2))
                candidates.add(clean_sentence(phrase3))

    return candidates


# =====================================================================
# 3. CORPUS EXPANSION GENERATOR RUNNER & DOMAIN ANALYTICS
# =====================================================================

def categorize_domain(action: str, obj: str) -> str:
    """Classifies an intent into one of five smart home functional domains."""
    if obj in ["lights", "lamp"]:
        return "Lighting (Pencahayaan)"
    elif obj == "heat":
        return "Climate (Pengatur Suhu & AC)"
    elif obj in ["music", "volume"]:
        return "Media & Audio (Hiburan & Volume)"
    elif action == "bring":
        return "Robot Butler (Pengantaran Barang)"
    elif action == "change language":
        return "System Language (Pengaturan Bahasa)"
    return "Other / Miscellaneous"


def generate_massive_corpus(
    intent_map_path: str = CONFIG_INTENT_MAP_PATH,
    output_csv_path: str = OUTPUT_CORPUS_CSV_PATH,
    target_per_intent: int = TARGET_PHRASES_PER_INTENT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generates, filters, deduplicates, and exports the massive 4,000+ phrase corpus."""
    if not os.path.exists(intent_map_path):
        raise FileNotFoundError(f"Intent map not found at {intent_map_path}")

    with open(intent_map_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    intent_map = config["intent_map"]
    all_rows: List[Dict[str, str]] = []

    print(f"[*] Starting Indonesian Corpus Generation for {len(intent_map)} intents...")
    print(f"[*] Target: {target_per_intent} diverse phrases per intent (Total: {len(intent_map) * target_per_intent:,}).")
    print(f"[*] Random Seed: {seed}")

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
        random.seed(seed + intent_id)
        random.shuffle(valid_phrases)

        # Select target count (stratified balance)
        selected_phrases = valid_phrases[:target_per_intent]
        if len(selected_phrases) < target_per_intent:
            print(f"[!] Warning: Intent {intent_id} generated {len(selected_phrases)} < {target_per_intent}")

        intent_stats[intent_id] = len(selected_phrases)

        # English reference sample
        eng_loc = f"in the {location}" if location != "none" else ""
        eng_sample = f"{action} the {obj} {eng_loc}".strip()

        domain = categorize_domain(action, obj)

        for phrase in selected_phrases:
            all_rows.append({
                "intent_id": intent_id,
                "action": action,
                "object": obj,
                "location": location,
                "domain": domain,
                "english_sample": eng_sample,
                "indonesian_phrase": phrase,
            })

    df = pd.DataFrame(all_rows)

    # Global Deduplication check (ensure uniqueness across dataset)
    df = df.drop_duplicates(subset=["indonesian_phrase", "intent_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")

    # Domain breakdown statistics
    domain_counts = df.groupby("domain")["intent_id"].nunique()
    domain_phrases = df.groupby("domain").size()

    # Sentence length statistics
    word_counts = df["indonesian_phrase"].apply(lambda s: len(s.split()))
    char_counts = df["indonesian_phrase"].apply(len)

    words = set()
    for phrase in df["indonesian_phrase"]:
        words.update(phrase.split())

    print("\n" + "=" * 70)
    print("      RINGKASAN GENERASI KORPUS INDONESIA SUPER LENGKAP 4K")
    print("=" * 70)
    print(f"Total Intents                   : {len(intent_map)} (IDs 0 to 30)")
    print(f"Total Raw Candidates Evaluated  : {total_candidates_generated:,} variasi kombinasi")
    print(f"Total Frasa Terpilih Bersih     : {len(df):,} kalimat")
    print(f"Rata-rata Frasa per Intent      : {len(df) / len(intent_map):.1f} kalimat/intent")
    print(f"Intent Min / Max Kalimat        : {min(intent_stats.values())} / {max(intent_stats.values())}")
    print(f"Total Kosakata Unik (Lexicon)   : {len(words)} kata unik")
    print(f"Panjang Kalimat (Kata)          : Rata-rata {word_counts.mean():.2f} (Min: {word_counts.min()}, Max: {word_counts.max()}, Std: {word_counts.std():.2f})")
    print(f"Panjang Karakter                : Rata-rata {char_counts.mean():.2f} (Min: {char_counts.min()}, Max: {char_counts.max()})")
    print("-" * 70)
    print("SEBARAN DOMAIN FUNGSIONAL SMART HOME:")
    for dom in sorted(domain_counts.index):
        print(f" - {dom:<36} : {domain_counts[dom]:2d} intents | {domain_phrases[dom]:5,d} frasa ({domain_phrases[dom]/len(df)*100:5.1f}%)")
    print("-" * 70)
    print(f"Lokasi File Output CSV          : {output_csv_path}")
    print("=" * 70 + "\n")

    return df


def main():
    parser = argparse.ArgumentParser(description="Indonesian Smart Home Spoken Corpus Generator")
    parser.add_argument("--intent-map", default=CONFIG_INTENT_MAP_PATH, help="Path to intent_map.json")
    parser.add_argument("--output", default=OUTPUT_CORPUS_CSV_PATH, help="Output CSV path")
    parser.add_argument("--target-per-intent", type=int, default=TARGET_PHRASES_PER_INTENT, help="Phrases per intent (130-150)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_massive_corpus(
        intent_map_path=args.intent_map,
        output_csv_path=args.output,
        target_per_intent=args.target_per_intent,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
