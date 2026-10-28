import urllib.request
import json
import time

boundary = "----BenchBoundary123"

for path in [
    "data/synthetic_indonesian/wavs_16k/id_intent02_030_tts_ardi.wav",
    "data/synthetic_indonesian/wavs_16k/id_intent25_375_tts_ardi.wav"
]:
    with open(path, "rb") as f:
        audio_data = f.read()

    hdr = f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"test.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode("utf-8")
    ftr = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = hdr + audio_data + ftr

    t0 = time.perf_counter()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode())
    dur = (time.perf_counter() - t0) * 1000
    print(f"File: {path}")
    print(f"Roundtrip: {dur:.2f} ms")
    print(f"Server timing: {data['timing_ms']}")
    print("-" * 50)
