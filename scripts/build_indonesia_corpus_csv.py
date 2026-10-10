import csv
import re
import os

def parse_markdown_to_csv():
    md_path = os.path.join("docs", "corpus_indonesia_31_intents.md")
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    table_pattern = re.compile(
        r'\|\s*\*\*(\d+)\*\*\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*"([^"]+)"\s*\|\s*(.*?)\s*\|',
        re.DOTALL
    )
    matches = table_pattern.findall(content)

    output_rows = []
    for m in matches:
        intent_id, action, obj, loc, orig_text, indonesian_block = m
        phrases = [re.sub(r'^\d+\.\s*', '', p.strip()) for p in indonesian_block.split('<br>') if p.strip()]
        for p in phrases:
            output_rows.append({
                'intent_id': int(intent_id),
                'action': action,
                'object': obj,
                'location': loc,
                'english_sample': orig_text,
                'indonesian_phrase': p
            })

    output_dir = os.path.join("data", "corpus")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "indonesian_smart_home_corpus_31_intents.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ['intent_id', 'action', 'object', 'location', 'english_sample', 'indonesian_phrase']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Sukses! Tersimpan di {csv_path} dengan total {len(output_rows)} variasi kalimat (31 intent).")

if __name__ == "__main__":
    parse_markdown_to_csv()
