"""Simulated Autonomous Voice Tester for SVARA API."""
import os
import json
import urllib.request

test_samples = [
    # Kitchen (kasus yang sebelumnya salah)
    ("data/synthetic_indonesian/wavs_16k/id_intent06_210_tts_ardi.wav", "Nyalakan lampu dapur (Ardi)"),
    ("data/synthetic_indonesian/wavs_16k/id_intent07_245_tts_gadis.wav", "Matikan lampu dapur (Gadis)"),
    # Washroom
    ("data/synthetic_indonesian/wavs_16k/id_intent09_315_tts_ardi.wav", "Nyalakan lampu kamar mandi (Ardi)"),
    ("data/synthetic_indonesian/wavs_16k/id_intent10_350_tts_gadis.wav", "Matikan lampu kamar mandi (Gadis)"),
    # Bedroom
    ("data/synthetic_indonesian/wavs_16k/id_intent12_420_tts_ardi.wav", "Nyalakan lampu kamar tidur (Ardi)"),
    ("data/synthetic_indonesian/wavs_16k/id_intent13_455_tts_gadis.wav", "Matikan lampu kamar tidur (Gadis)"),
    # Music Hallway
    ("data/synthetic_indonesian/wavs_16k/id_intent24_840_tts_ardi.wav", "Nyalakan musik di lorong (Ardi)"),
    ("data/synthetic_indonesian/wavs_16k/id_intent25_875_tts_gadis.wav", "Matikan musik di lorong (Gadis)"),
]

boundary = "----TestBoundary12345"

print("="*65)
print("TESTING OTOMATIS ENGINE SUARA DENGAN BERBAGAI KONDISI AUDIO NYATA")
print("="*65)

for path, desc in test_samples:
    if not os.path.exists(path):
        # find matching file
        intent_str = os.path.basename(path).split("_")[1]
        cands = [p for p in os.listdir("data/synthetic_indonesian/wavs_16k") if intent_str in p]
        if cands:
            path = os.path.join("data/synthetic_indonesian/wavs_16k", cands[0])
        else:
            continue

    with open(path, "rb") as f:
        audio_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio"; filename="test.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8") + audio_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )

    try:
        res = urllib.request.urlopen(req)
        data = json.loads(res.read().decode())
        intent = data["intent"]
        conf = data["confidence"]["overall"] * 100
        lat = data["timing_ms"]["total_ms"]
        inf_lat = data["timing_ms"]["inference_ms"]
        status = "PASSED" if data["accepted"] else "REJECTED"
        print(f"[{status}] {desc:<35} -> ({intent['action']:<8} {intent['object']:<7} {intent['location']:<8}) | Conf: {conf:4.1f}% | Latency: {lat:5.1f}ms (Inference: {inf_lat:5.1f}ms)")
    except Exception as e:
        print(f"[FAIL] {desc}: {e}")

print("="*65)
