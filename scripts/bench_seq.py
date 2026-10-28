import urllib.request
import time
import json

boundary = "----BenchBoundary123"
path = "data/synthetic_indonesian/wavs_16k/id_intent02_030_tts_ardi.wav"
with open(path, "rb") as f:
    audio_data = f.read()

hdr = f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"test.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode("utf-8")
ftr = f"\r\n--{boundary}--\r\n".encode("utf-8")
body = hdr + audio_data + ftr

for i in range(3):
    t0 = time.perf_counter()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode())
    dur = (time.perf_counter() - t0) * 1000
    inf = data["timing_ms"]["inference_ms"]
    print(f"Req {i}: Total latency = {dur:.1f}ms | Inference: {inf:.1f}ms")
