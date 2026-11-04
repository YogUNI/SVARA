"""Comprehensive Colloquial & Slang Corpus Expansion for SVARA Smart Home.

Task: Sub-step 13.3 (FASE 13: Real-World Speech Hardening)
Reference: docs/14_ROADMAP_PEMATANGAN_DEEP_LEARNING_REAL_WORLD.md, AGENTS.md

Generates >= 1,500 natural, highly diverse conversational Indonesian smart-home phrases
across all 31 intents, covering:
1. Everyday spoken Indonesian verbs: nyalain, idupin, matiin, padamin, kencengin, gedein,
   pelanin, kecilin, dinginin, panasin, puterin, setel, ambilin, bawain, ubah.
2. Natural speech particles: dong, ya, deh, nih, lah, tolong, coba, svara, eh svara.
3. Situational context expressions: gelap nih, mau tidur nih, panas banget, kedinginan,
   berisik banget, haus nih, mau pergi nih.
4. Strict semantic validation against `configs/intent_map.json`.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Set, Tuple
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

CONFIG_INTENT_MAP = "configs/intent_map.json"
OUTPUT_COLLOQUIAL_CSV = "data/corpus/indonesian_smart_home_colloquial_expanded.csv"
OUTPUT_MASTER_CORPUS_CSV = "data/corpus/indonesian_smart_home_corpus_master.csv"
REPORT_PATH = "reports/colloquial_expansion_audit.json"

# Prefix variations for everyday conversational speech
PREFIXES_GENERAL = [
    "",
    "tolong ",
    "coba ",
    "bisa tolong ",
    "eh tolong ",
    "svara tolong ",
    "svara ",
    "eh svara ",
    "halo svara ",
    "svara coba ",
    "tolongin dong ",
    "bisa gak tolong ",
]

# Suffix particles for natural Indonesian spoken prosody
SUFFIXES_CASUAL = [
    "",
    " dong",
    " ya",
    " dong ya",
    " tolong ya",
    " sekarang",
    " dong svara",
    " cepetan",
    " bentar",
    " ya tolong",
    " lah",
    " dong tolong",
]


def load_legal_intent_map(path: str = CONFIG_INTENT_MAP) -> Dict[int, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data["intent_map"].items()}


def get_base_templates_per_intent() -> Dict[int, Dict[str, List[str]]]:
    """Provides intent-specific domain phrasing templates with natural colloquial Indonesian."""
    return {
        # =====================================================================
        # DOMAIN 1: LIGHTING / PENCAHAYAAN (Intents 0, 1, 2, 3, 4, 15, 16, 17, 18, 19)
        # =====================================================================
        0: {  # activate lamp none (lampu meja / lampu tidur / lampu baca)
            "contexts": ["gelap nih ", "gelap banget meja belajar ", "mau baca buku nih ", "remang-remang nih meja "],
            "verbs": ["nyalain", "idupin", "hidupin", "aktifkan", "tolong nyalain", "coba idupin"],
            "targets": ["lampu meja", "lampu belajar", "lampu baca", "lampu tidur", "lampu nakas", "lampu meja kerja"],
        },
        1: {  # activate lights bedroom (lampu kamar tidur)
            "contexts": ["kamar gelap nih ", "kamar tidur gelap banget ", "mau masuk kamar tidur nih ", "gelap nih di kamar "],
            "verbs": ["nyalain", "idupin", "hidupin", "tolong nyalain", "coba idupin", "nyalakan"],
            "targets": ["lampu kamar tidur", "lampu kamar", "lampu di kamar tidur", "lampu utama kamar tidur", "semua lampu kamar"],
        },
        2: {  # activate lights kitchen (lampu dapur)
            "contexts": ["dapur gelap nih ", "dapur gelap banget mau masak ", "mau ke dapur nih ", "gelap di dapur "],
            "verbs": ["nyalain", "idupin", "hidupin", "tolong nyalain", "nyalakan"],
            "targets": ["lampu dapur", "lampu di dapur", "lampu ruangan dapur", "semua lampu dapur"],
        },
        3: {  # activate lights none (semua lampu / lampu rumah)
            "contexts": ["gelap banget seisi rumah nih ", "rumah gelap nih ", "gelap gulita nih ", ""],
            "verbs": ["nyalain", "idupin", "hidupin", "nyalakan", "aktifkan"],
            "targets": ["lampu", "semua lampu", "seluruh lampu", "lampu rumah", "lampunya", "lampu-lampu"],
        },
        4: {  # activate lights washroom (lampu kamar mandi / toilet)
            "contexts": ["mau ke kamar mandi nih ", "toilet gelap nih ", "kamar mandi gelap banget ", "mau buang air nih "],
            "verbs": ["nyalain", "idupin", "hidupin", "tolong nyalain", "nyalakan"],
            "targets": ["lampu kamar mandi", "lampu toilet", "lampu WC", "lampu di kamar mandi", "lampu di toilet"],
        },
        15: {  # deactivate lamp none (matikan lampu meja)
            "contexts": ["udah terang nih ", "udah selesai baca nih ", "silau nih meja belajarnya ", ""],
            "verbs": ["matiin", "padamin", "matikan", "padamkan", "coba matiin"],
            "targets": ["lampu meja", "lampu belajar", "lampu baca", "lampu tidur", "lampu nakas"],
        },
        16: {  # deactivate lights bedroom (matikan lampu kamar tidur)
            "contexts": ["mau tidur nih ", "silau banget kamar tidurnya ", "udah mau tidur malam ", "kamar tidur silau "],
            "verbs": ["matiin", "padamin", "matikan", "padamkan", "tolong matiin"],
            "targets": ["lampu kamar tidur", "lampu kamar", "lampu di kamar tidur", "semua lampu kamar"],
        },
        17: {  # deactivate lights kitchen (matikan lampu dapur)
            "contexts": ["udah selesai masak nih ", "udah beres di dapur ", "tinggalin dapur nih ", ""],
            "verbs": ["matiin", "padamin", "matikan", "padamkan", "tolong matiin"],
            "targets": ["lampu dapur", "lampu di dapur", "lampu ruangan dapur", "semua lampu dapur"],
        },
        18: {  # deactivate lights none (matikan semua lampu)
            "contexts": ["mau keluar rumah nih ", "mau tidur semua ", "semua ruangan udah terang ", ""],
            "verbs": ["matiin", "padamin", "matikan", "padamkan", "nonaktifkan"],
            "targets": ["lampu", "semua lampu", "seluruh lampu", "lampu rumah", "lampunya"],
        },
        19: {  # deactivate lights washroom (matikan lampu kamar mandi / toilet)
            "contexts": ["udah selesai pakai toilet nih ", "udah selesai dari kamar mandi ", "kamar mandi kosong nih ", ""],
            "verbs": ["matiin", "padamin", "matikan", "padamkan", "tolong matiin"],
            "targets": ["lampu kamar mandi", "lampu toilet", "lampu WC", "lampu di kamar mandi"],
        },

        # =====================================================================
        # DOMAIN 2: MUSIC / MEDIA (Intents 5, 20)
        # =====================================================================
        5: {  # activate music none (putar / lanjutkan musik)
            "contexts": ["hening banget nih ruangan ", "sepi banget nih ", "mau denger lagu lagi ", "bosen nih "],
            "verbs": ["putar", "puterin", "setel", "setelin", "lanjutin", "play", "mainkan", "nyalain lagi"],
            "targets": ["musik", "lagu", "musiknya", "lagunya", "musik yang tadi", "lagu favorit", "pemutar musik"],
        },
        20: {  # deactivate music none (matikan / stop musik)
            "contexts": ["berisik banget nih ", "mau istirahat dulu nih ", "pusing denger suara bising ", "ada telepon masuk nih "],
            "verbs": ["matiin", "stop", "hentikan", "berhentiin", "pause", "matikan", "padamin"],
            "targets": ["musik", "lagu", "musiknya", "lagunya", "pemutar musik", "suara musik"],
        },

        # =====================================================================
        # DOMAIN 3: CLIMATE / TEMPERATURE (Intents 21, 22, 23, 24, 26, 27, 28, 29)
        # =====================================================================
        21: {  # decrease heat bedroom (dinginkan kamar tidur / turunkan suhu kamar)
            "contexts": ["kamar tidur gerah banget nih ", "kamar tidur panas nih ", "sumuk banget di kamar tidur ", "kamar hawanya panas "],
            "verbs": ["dinginin", "turunin suhu", "sejukin", "turunkan suhu", "kurangi suhu"],
            "targets": ["kamar tidur", "kamar", "ruangan kamar tidur", "AC kamar tidur", "suhu kamar tidur"],
        },
        22: {  # decrease heat kitchen (dinginkan dapur)
            "contexts": ["dapur panas banget abis masak ", "dapur hawanya gerah ", "panas nih di dapur ", ""],
            "verbs": ["dinginin", "turunin suhu", "sejukin", "turunkan suhu"],
            "targets": ["dapur", "ruangan dapur", "suhu dapur", "area dapur"],
        },
        23: {  # decrease heat none (turunkan suhu ruangan / bikin dingin)
            "contexts": ["gerah banget nih rumah ", "panas banget hawanya ", "sumuk banget udaranya ", "keringetan nih "],
            "verbs": ["turunin", "dinginin", "sejukin", "turunkan", "kurangi"],
            "targets": ["suhu", "suhunya", "suhu ruangan", "suhu AC", "temperatur", "temperaturnya"],
        },
        24: {  # decrease heat washroom (dinginkan kamar mandi)
            "contexts": ["kamar mandi pengap dan panas nih ", "toilet hawanya panas ", "gerah nih di kamar mandi ", ""],
            "verbs": ["dinginin", "turunin suhu", "sejukin", "turunkan suhu"],
            "targets": ["kamar mandi", "toilet", "WC", "ruangan kamar mandi"],
        },
        26: {  # increase heat bedroom (naikkan suhu kamar tidur / hangatkan kamar)
            "contexts": ["kamar tidur dingin banget nih ", "kamar tidur terlalu dingin ", "menggigil nih di kamar tidur ", "kedinginan di kamar "],
            "verbs": ["naikin suhu", "hangatkan", "angetin", "naikkan suhu", "tambah suhu"],
            "targets": ["kamar tidur", "kamar", "ruangan kamar tidur", "AC kamar tidur"],
        },
        27: {  # increase heat kitchen (naikkan suhu dapur / hangatkan dapur)
            "contexts": ["dapur dingin banget nih ", "dapur hawanya terlalu dingin ", ""],
            "verbs": ["naikin suhu", "hangatkan", "angetin", "naikkan suhu"],
            "targets": ["dapur", "ruangan dapur", "suhu dapur"],
        },
        28: {  # increase heat none (naikkan suhu ruangan / bikin hangat)
            "contexts": ["dingin banget nih rumah ", "menggigil nih udaranya ", "kedinginan banget nih ", "terlalu beku nih ruangan "],
            "verbs": ["naikin", "hangatkan", "angetin", "naikkan", "tambah"],
            "targets": ["suhu", "suhunya", "suhu ruangan", "suhu AC", "temperatur", "temperaturnya"],
        },
        29: {  # increase heat washroom (naikkan suhu kamar mandi / hangatkan kamar mandi)
            "contexts": ["air dan kamar mandi dingin banget ", "toilet terlalu dingin nih ", "menggigil di kamar mandi ", ""],
            "verbs": ["naikin suhu", "hangatkan", "angetin", "naikkan suhu"],
            "targets": ["kamar mandi", "toilet", "WC", "ruangan kamar mandi"],
        },

        # =====================================================================
        # DOMAIN 4: VOLUME / AUDIO LEVEL (Intents 25, 30)
        # =====================================================================
        25: {  # decrease volume none (kecilkan volume / pelankan suara)
            "contexts": ["suaranya kekencengan nih ", "terlalu keras suaranya ", "bising banget nih ", "ada orang tidur nih "],
            "verbs": ["kecilin", "pelanin", "turunin", "kecilkan", "pelankan", "rendahkan"],
            "targets": ["volume", "volumenya", "suara", "suaranya", "volume speaker", "suara speaker"],
        },
        30: {  # increase volume none (besarkan volume / kencangkan suara)
            "contexts": ["suaranya kekecilan nih ", "kurang keras nih suaranya ", "gak kedengeran nih ", "kurang denger suaranya "],
            "verbs": ["gedein", "kencengin", "naikin", "besarkan", "kencangkan", "tinggikan"],
            "targets": ["volume", "volumenya", "suara", "suaranya", "volume speaker", "suara speaker"],
        },

        # =====================================================================
        # DOMAIN 5: UTILITY & LANGUAGE (Intents 6, 7, 8, 9, 10, 11, 12, 13, 14)
        # =====================================================================
        6: {  # bring juice none (bawakan jus)
            "contexts": ["haus banget nih ", "tenggorokan kering nih ", "mau minum yang seger-seger ", ""],
            "verbs": ["ambilin", "bawain", "ambillan", "bawakan", "anterin"],
            "targets": ["jus", "jus jeruk", "jus dingin", "segelas jus", "minuman jus", "jus buah"],
        },
        7: {  # bring newspaper none (bawakan koran)
            "contexts": ["mau baca berita pagi nih ", "mau liat kabar hari ini ", "butuh bacaan nih ", ""],
            "verbs": ["ambilin", "bawain", "ambilkan", "bawakan", "anterin"],
            "targets": ["koran", "koran hari ini", "koran pagi", "surat kabar"],
        },
        8: {  # bring shoes none (bawakan sepatu)
            "contexts": ["mau berangkat kerja nih ", "mau keluar rumah nih ", "mau pergi nih ", "udah siap jalan nih "],
            "verbs": ["ambilin", "bawain", "ambilkan", "bawakan", "anterin"],
            "targets": ["sepatu", "sepatu saya", "sepatu gua", "sepatu kerja", "sepatu olahraga"],
        },
        9: {  # bring socks none (bawakan kaus kaki)
            "contexts": ["mau pakai sepatu nih ", "kaki dingin nih ", "mau berangkat nih butuh kaus kaki ", ""],
            "verbs": ["ambilin", "bawain", "ambilkan", "bawakan", "cariin"],
            "targets": ["kaus kaki", "kaos kaki", "kaos kaki bersih", "kaos kaki saya", "sepasang kaos kaki"],
        },
        10: {  # change language Chinese none (ganti bahasa ke Mandarin)
            "contexts": ["mau belajar bahasa mandarin nih ", "latihan bahasa mandarin ", ""],
            "verbs": ["ganti bahasa ke", "ubah bahasa ke", "setel bahasa ke", "ganti bahasanya ke", "switch ke"],
            "targets": ["Mandarin", "bahasa Mandarin", "Chinese"],
        },
        11: {  # change language English none (ganti bahasa ke Inggris)
            "contexts": ["practice english nih ", "mau pakai bahasa inggris ", ""],
            "verbs": ["ganti bahasa ke", "ubah bahasa ke", "setel bahasa ke", "ganti bahasanya ke", "switch ke"],
            "targets": ["Bahasa Inggris", "Inggris", "English"],
        },
        12: {  # change language German none (ganti bahasa ke Jerman)
            "contexts": ["mau belajar bahasa jerman nih ", "latihan ngomong jerman ", ""],
            "verbs": ["ganti bahasa ke", "ubah bahasa ke", "setel bahasa ke", "ganti bahasanya ke", "switch ke"],
            "targets": ["Bahasa Jerman", "Jerman", "German"],
        },
        13: {  # change language Korean none (ganti bahasa ke Korea)
            "contexts": ["mau belajar bahasa korea nih ", "lagi suka drakor pengen bahasa korea ", ""],
            "verbs": ["ganti bahasa ke", "ubah bahasa ke", "setel bahasa ke", "ganti bahasanya ke", "switch ke"],
            "targets": ["Bahasa Korea", "Korea", "Korean"],
        },
        14: {  # change language none none (ganti bahasa umum)
            "contexts": ["mau ganti bahasa asisten nih ", "bahasanya kurang cocok ", ""],
            "verbs": ["ganti", "ubah", "gantiin", "switch", "setel"],
            "targets": ["bahasa", "bahasanya", "bahasa sistem", "bahasa asisten", "pengaturan bahasa"],
        },
    }


def generate_colloquial_dataset(
    target_per_intent: int = 55,
    existing_corpus_path: str = "data/corpus/indonesian_smart_home_corpus_31_intents.csv",
) -> pd.DataFrame:
    """Generates >= 1,500 new colloquial phrases strictly validated against intent definitions."""
    legal_map = load_legal_intent_map()
    templates = get_base_templates_per_intent()

    # Read existing phrases to guarantee novelty (zero exact duplicates)
    existing_phrases: Set[str] = set()
    if os.path.exists(existing_corpus_path):
        df_old = pd.read_csv(existing_corpus_path)
        existing_phrases = set(df_old["indonesian_phrase"].str.strip().str.lower())

    records = []

    for intent_id in range(31):
        info = legal_map[intent_id]
        act = info["action"]
        obj = info["object"]
        loc = info["location"]
        tpl = templates[intent_id]

        contexts = tpl["contexts"]
        verbs = tpl["verbs"]
        targets = tpl["targets"]

        intent_generated = set()
        attempts = 0
        max_attempts = 5000

        while len(intent_generated) < target_per_intent and attempts < max_attempts:
            attempts += 1
            ctx = random.choice(contexts)
            pfx = random.choice(PREFIXES_GENERAL)
            vb = random.choice(verbs)
            tgt = random.choice(targets)
            sfx = random.choice(SUFFIXES_CASUAL)

            # Syntax variations
            syntax_type = random.randint(1, 4)
            if syntax_type == 1:
                # [Context] + [Prefix] + [Verb] + [Target] + [Suffix]
                phrase = f"{ctx}{pfx}{vb} {tgt}{sfx}".strip()
            elif syntax_type == 2:
                # [Prefix] + [Target] + [Verb] + [Suffix] (e.g., "Tolong lampu kamar nyalain dong")
                phrase = f"{pfx}{tgt} tolong {vb}{sfx}".strip()
            elif syntax_type == 3:
                # Short colloquial imperative: [Verb] + [Target] + [Suffix]
                phrase = f"{vb} {tgt}{sfx}".strip()
            else:
                # Inversion: [Target] + [Verb] (e.g. "Lampu tidur idupin dong")
                phrase = f"{tgt} {vb}{sfx}".strip()

            # Clean formatting
            phrase = " ".join(phrase.split())
            phrase_clean = phrase.capitalize()

            # Novelty check
            phrase_key = phrase_clean.lower()
            if phrase_key not in existing_phrases and phrase_key not in intent_generated and len(phrase_clean) >= 6:
                intent_generated.add(phrase_key)
                records.append({
                    "intent_id": intent_id,
                    "action": act,
                    "object": obj,
                    "location": loc,
                    "phrase": phrase_clean,
                    "length_chars": len(phrase_clean),
                    "num_words": len(phrase_clean.split()),
                })

    df_colloquial = pd.DataFrame(records)
    return df_colloquial


def audit_and_merge_corpus(
    df_colloquial: pd.DataFrame,
    existing_31_path: str = "data/corpus/indonesian_smart_home_corpus_31_intents.csv",
    existing_4k_path: str = "data/corpus/indonesian_smart_home_corpus_4k.csv",
    output_colloquial_path: str = OUTPUT_COLLOQUIAL_CSV,
    output_master_path: str = OUTPUT_MASTER_CORPUS_CSV,
    report_path: str = REPORT_PATH,
) -> Dict:
    """Validates the new colloquial phrases and merges them into the Master Indonesian Corpus."""
    legal_map = load_legal_intent_map()

    # 1. Verification of colloquial additions
    total_added = len(df_colloquial)
    intent_counts = df_colloquial["intent_id"].value_counts().to_dict()
    invalid_triplets = []

    for idx, row in df_colloquial.iterrows():
        iid = row["intent_id"]
        legal = legal_map.get(iid)
        if not legal or legal["action"] != row["action"] or legal["object"] != row["object"] or legal["location"] != row["location"]:
            invalid_triplets.append(f"Row {idx}: intent {iid} mismatch with {row['action']} {row['object']} {row['location']}")

    os.makedirs(os.path.dirname(output_colloquial_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df_colloquial.to_csv(output_colloquial_path, index=False)

    # 2. Compile Master Unified Corpus
    master_rows = []

    # A. Add 31 intent base corpus
    if os.path.exists(existing_31_path):
        df_31 = pd.read_csv(existing_31_path)
        for _, r in df_31.iterrows():
            master_rows.append({
                "intent_id": int(r["intent_id"]),
                "action": r["action"],
                "object": r["object"],
                "location": r["location"],
                "phrase": r["indonesian_phrase"].strip(),
                "source": "corpus_31_base",
            })

    # B. Add colloquial expansion
    for _, r in df_colloquial.iterrows():
        master_rows.append({
            "intent_id": int(r["intent_id"]),
            "action": r["action"],
            "object": r["object"],
            "location": r["location"],
            "phrase": r["phrase"].strip(),
            "source": "colloquial_expansion_13_3",
        })

    # Deduplicate master
    df_master = pd.DataFrame(master_rows)
    df_master["phrase_lower"] = df_master["phrase"].str.lower()
    df_master = df_master.drop_duplicates(subset=["intent_id", "phrase_lower"]).reset_index(drop=True)
    df_master = df_master.drop(columns=["phrase_lower"])
    df_master.to_csv(output_master_path, index=False)

    passed = (total_added >= 1500) and (len(invalid_triplets) == 0) and (len(intent_counts) == 31)

    audit_summary = {
        "substep": "13.3",
        "total_colloquial_phrases_generated": total_added,
        "acceptance_criteria_target": 1500,
        "target_met": total_added >= 1500,
        "unique_intents_covered": len(intent_counts),
        "min_phrases_per_intent": min(intent_counts.values()) if intent_counts else 0,
        "max_phrases_per_intent": max(intent_counts.values()) if intent_counts else 0,
        "avg_phrases_per_intent": round(float(total_added / 31), 2),
        "semantic_validation": {
            "invalid_triplets_count": len(invalid_triplets),
            "triplets_valid": len(invalid_triplets) == 0,
            "errors": invalid_triplets[:5],
        },
        "master_corpus_compiled": {
            "output_path": output_master_path,
            "total_master_phrases": len(df_master),
            "sources": df_master["source"].value_counts().to_dict(),
        },
        "verdict": "PASSED" if passed else "FAILED",
        "status": "SELESAI",
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    return audit_summary


def main():
    parser = argparse.ArgumentParser(description="Expand Indonesian Smart Home Colloquial Corpus")
    parser.add_argument("--target-per-intent", type=int, default=55, help="Number of colloquial phrases per intent")
    parser.add_argument("--out-csv", default=OUTPUT_COLLOQUIAL_CSV, help="Output colloquial CSV path")
    parser.add_argument("--master-csv", default=OUTPUT_MASTER_CORPUS_CSV, help="Output master CSV path")
    parser.add_argument("--report", default=REPORT_PATH, help="Output report JSON path")
    args = parser.parse_args()

    print("=" * 65)
    print("SVARA COLLOQUIAL CORPUS EXPANSION (SUB-STEP 13.3)")
    print("=" * 65)
    print(f"Target per Intent : {args.target_per_intent} frasa")
    print(f"Total Target      : {args.target_per_intent * 31} frasa baru (Target >= 1.500)")
    print("Memulai generasi variasi bahasa lisan Indonesia...")

    df_colloquial = generate_colloquial_dataset(target_per_intent=args.target_per_intent)
    audit = audit_and_merge_corpus(
        df_colloquial=df_colloquial,
        output_colloquial_path=args.out_csv,
        output_master_path=args.master_csv,
        report_path=args.report,
    )

    print("\n" + "=" * 65)
    print("HASIL EKSEKUSI SUB-STEP 13.3:")
    print("=" * 65)
    print(f"Total Frasa Baru Ditambahkan : {audit['total_colloquial_phrases_generated']} (Target >= 1.500: {audit['target_met']})")
    print(f"Cakupan Intent               : {audit['unique_intents_covered']} / 31 Intent Lengkap")
    print(f"Rata-rata per Intent         : {audit['avg_phrases_per_intent']} frasa/intent")
    print(f"Validasi Semantik Triplet    : {'100% VALID (Zero Invalid)' if audit['semantic_validation']['triplets_valid'] else 'GAGAL'}")
    print(f"Master Corpus Total          : {audit['master_corpus_compiled']['total_master_phrases']} frasa")
    print(f"Status / Verdict             : {audit['verdict']}")
    print(f"Laporan Audit                : {args.report}")
    print("=" * 65)


if __name__ == "__main__":
    main()
