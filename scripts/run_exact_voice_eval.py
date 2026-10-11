"""Find audio paths matching actual intents and test against API."""
import os
import json
import urllib.request
import pandas as pd

df = pd.read_csv("data/synthetic_indonesian/metadata.csv")

targets = [
    (2, "activate", "lights", "kitchen"),
    (17, "deactivate", "lights", "kitchen"),
    (4, "activate", "lights", "washroom"),
    (19, "deactivate", "lights", "washroom"),
    (1, "activate", "lights", "bedroom"),
    (16, "deactivate", "lights", "bedroom"),
    (5, "activate", "music", "none"),
    (20, "deactivate", "music", "none"),
    (25, "decrease", "volume", "none"),
    (30, "increase", "volume", "none"),
]

boundary = "----TestBoundaryExact123"

print("="*75)
print("HASIL PENGUJIAN OTOMATIS SUARA INDONESIA PADA ENGINE SVARA TERBARU")
print("="*75)

for intent_id, act_target, obj_target, loc_target in targets:
    sub = df[df["intent_id"] == intent_id]
    if len(sub) == 0:
        continue
    # Pick sample
    row = sub.iloc[0]
    wav_path = os.path.join("data/synthetic_indonesian", row["path"])
    phrase = row["transcript"]

    with open(wav_path, "rb") as f:
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

    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode())
    pred = data["intent"]
    conf = data["confidence"]["overall"] * 100
    lat = data["timing_ms"]["total_ms"]

    is_match = (pred["action"] == act_target and pred["object"] == obj_target and pred["location"] == loc_target)
    status = "SUCCESS" if is_match else "MISMATCH"

    print(f"[{status:^8}] \"{phrase:<32}\" -> Prediksi: ({pred['action']:<10} {pred['object']:<7} {pred['location']:<8}) | Conf: {conf:5.1f}% | Latency: {lat:4.0f}ms")

print("="*75)
