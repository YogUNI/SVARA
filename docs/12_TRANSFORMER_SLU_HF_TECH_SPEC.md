# 12 — Technical Spec: "Transformer SLU, wav2vec2" with Python + HuggingFace

This file closes the gap between the lecturer's requirement and our implementation. It complements docs/02 (experiments) and docs/03
(stack). Where they differ, **stop and report**; do not silently pick one. Everything here is written from knowledge and has **not been
executed yet**: items marked `VERIFY` must be confirmed in the first sprint (docs/11 §10).

## 1. Compliance map: lecturer spec -> what we do
| Lecturer's field | Our implementation | Where |
|---|---|---|
| Domain: Speech Recognition | Speech understanding from raw audio (end-to-end SLU, a branch of speech processing) | docs/00 |
| Task: Smart Home Voice Control | speech -> (action, object, location) -> simulated home action | docs/05, 06 |
| Method: **Transformer SLU** | pretrained **Transformer encoder (wav2vec 2.0)** + pooling + classification heads, fine-tuned end-to-end | §2–4 |
| Method: **wav2vec2** | `facebook/wav2vec2-base` (self-supervised) as default; `-960h` and layer-truncated variants as ablations | §3, docs/02 |
| Tools: **Python** | Python 3.10/3.11, PyTorch, torchaudio | docs/03 |
| Tools: **HuggingFace** | `transformers` models + feature-extractor conventions + Hub checkpoints (+ optional Hub model card/upload and Spaces hosting), plus an HF-native baseline | §5–6 |
| Dataset: FSC | lecturer's GitHub mirror, verified against README and paper counts | docs/01 |
| Main paper: Lugosch et al. 2019 | introduces FSC and an end-to-end SLU baseline with pretraining; wav2vec2 is a **later** method (do not claim the paper used it) | docs/07 |

What "Transformer SLU" means in our report: spoken language understanding done **end-to-end** (no intermediate transcript) with a
self-attention (Transformer) encoder reading the speech representation and a small head predicting the intent slots.

## 2. Architecture details (wav2vec2-base facts; confirm with the model config)
- Input: 16 kHz mono waveform, per-utterance zero-mean/unit-variance normalization.
- **Feature encoder**: 7 temporal conv layers (512 channels, kernels 10,3,3,3,3,2,2; strides 5,2,2,2,2,2,2), total stride 320 samples -> one frame
  per 20 ms (about 49 frames per second), receptive field about 25 ms. First conv layer uses group normalization (relevant for padding, §4).
- **Transformer encoder**: 12 layers, hidden size 768, 12 attention heads, feed-forward size 3072, about 95 M parameters in the full base model;
  convolutional relative positional embedding before the layers; GELU; post-layer-norm style for base.
- Pretraining: self-supervised contrastive objective on unlabeled speech (LibriSpeech, about 960 h for base). The `-960h` checkpoint additionally
  has CTC fine-tuning for English ASR.
- A 3-second clip is about 150 frames: attention cost is tiny; memory is dominated by weights, optimizer states and activations of 12 layers.
- SVARA head: length-aware mean pooling over frames -> dropout -> three `Linear(768, C)` heads (action 6, object 14, location 4) (+ optional joint head 31).
- Rough memory (estimate, VERIFY on Colab): fp32 weights about 0.4 GB; AdamW adds about 2× more; activations depend on batch and clip length.
  Batch 16 of 2–4 s clips on a 16 GB GPU should fit with fp16 (VERIFY).

## 3. Reference sketch (indicative, not tested; keep close to this design)
```python
import torch, torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2VecSLU(nn.Module):
    def __init__(self, ckpt="facebook/wav2vec2-base", n_action=6, n_object=14, n_location=4, n_joint=None,
                 keep_layers=None, freeze_encoder=False, dropout=0.1, mask_time_prob=0.05, layerdrop=0.0):
        super().__init__()
        # config overrides are passed to from_pretrained (VERIFY the exact kwargs for the installed transformers version)
        self.enc = Wav2Vec2Model.from_pretrained(ckpt, mask_time_prob=mask_time_prob, layerdrop=layerdrop)
        self.enc.freeze_feature_encoder()                       # CNN frozen, always
        if keep_layers is not None and keep_layers < len(self.enc.encoder.layers):
            self.enc.encoder.layers = nn.ModuleList(self.enc.encoder.layers[:keep_layers])  # truncation (E5)
            self.enc.config.num_hidden_layers = keep_layers
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.enc.parameters(): p.requires_grad = False
        d = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({"action": nn.Linear(d, n_action), "object": nn.Linear(d, n_object),
                                    "location": nn.Linear(d, n_location)})
        self.joint = nn.Linear(d, n_joint) if n_joint else None

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder: self.enc.eval()                 # frozen encoder must not apply dropout/SpecAugment
        return self

    def forward(self, wav, lengths):                            # wav [B, T] zero-padded; lengths [B] in samples
        h = self.enc(wav).last_hidden_state                     # NO attention_mask for base models (see §4)
        f_len = self.enc._get_feat_extract_output_lengths(lengths)   # VERIFY helper name/behavior for the installed version
        mask = (torch.arange(h.size(1), device=h.device)[None, :] < f_len[:, None]).unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = self.drop(pooled)
        out = {k: head(pooled) for k, head in self.heads.items()}
        if self.joint is not None: out["joint"] = self.joint(pooled)
        return out
```
Loss: sum of three cross-entropies (+ optional label smoothing). Optimizer: AdamW with two parameter groups (encoder, heads), no weight decay on
bias and LayerNorm weights; warmup + linear decay; gradient clipping max-norm 1.0; fp16 autocast with GradScaler (or bf16 if the GPU supports it).

## 4. wav2vec2 gotchas (with the reason, so the agent can judge edge cases)
1. **No attention mask for base**: base-style models use group norm in the first conv layer; masks are not meant to be passed. Pad with zeros, bucket by length.
2. **Padding is not perfectly invariant** for these models (group norm statistics depend on the padded length). Expect small differences in V-18; the test
   should check differences are small and **serving uses batch size 1 without padding**. If evaluation uses batches, use length bucketing; the evaluation script
   may also run with batch size 1 for the final reported numbers (decide and log).
3. **Frozen modules must be in eval mode** (dropout and time-masking are active in train mode even if parameters are frozen).
4. **Time masking (SpecAugment)** is a config feature: `apply_spec_augment`, `mask_time_prob`, `mask_time_length`; active only in train mode. Also check the default
   `layerdrop` (often non-zero in the config): set it explicitly (0.0 or small) and log it.
5. **Truncation** of layers: after truncating, the last hidden state is the output of the last kept layer for base models. For models with "stable layer norm"
   (large/XLS-R style) a final layer norm follows the stack; re-check before reusing truncation code there.
6. **Normalization parity**: the HuggingFace feature extractor normalizes each utterance to zero mean and unit variance with a small epsilon. Reimplement the
   exact same math in `audio.py` and in the server (numpy), and test parity against `Wav2Vec2FeatureExtractor` on sample clips.
7. **Sampling rate**: always 16 kHz; reject or resample anything else explicitly.
8. **Max length**: crop/pad deterministically in eval; random crop only in training; length from EDA (docs/01).
9. **fp16**: if NaN/inf appears, lower LR, check the clipping, or run that experiment in fp32; log the decision.

## 5. HuggingFace usage plan (what is HF-based and what is not)
- **Used**: `transformers` model classes and config (`Wav2Vec2Model`, optionally `Wav2Vec2ForSequenceClassification`), pretrained weights from the HF Hub,
  `Wav2Vec2FeatureExtractor` as the reference for normalization, `huggingface_hub` caching, optional `safetensors` checkpoint format.
- **Not used (on purpose)**: HF `datasets` (local CSV + our loader is simpler for grouped splits), HF `Trainer` for the main model (we need multi-head loss,
  two LR groups, custom metrics and resumable training on Colab; a thin custom PyTorch loop around the HF model is easier to control and test).
  We say this explicitly in the report so the lecturer sees HF is used for the model/weights/ecosystem.
- **Cache**: set `HF_HOME` to a Drive path in Colab so checkpoints are not re-downloaded every session. Public checkpoints need no token; never put a token in the repo.
- **HF-native baseline (cheap, valuable)**: `Wav2Vec2ForSequenceClassification` with a 31-way joint head (and the HF `Trainer` or a short loop) as an independent
  implementation. Purpose: (a) sanity-check our custom model (similar accuracy expected within noise), (b) show clean use of the HF ecosystem. Task P3-17.
- **Optional publication**: upload the final fine-tuned model and a model card (intended use, data, metrics from `summary.json`, limitations, license note) to the
  HF Hub; host the demo on HF Spaces (Docker) if the free tier suffices. Check license terms of FSC before uploading anything derived from it. Task P7-09 (optional).
- **Serving does not need HF**: the Docker image runs onnxruntime + numpy (docs/05). `requirements-serve.txt` must not include `transformers` or `torch`.

## 6. Innovative-but-contained extensions (all optional, each answers a clear question)
| ID | Idea | Question it answers | Cost |
|---|---|---|---|
| E11 | **Layer-weighted sum** of all hidden layers (learnable softmax weights, SUPERB-style) instead of last layer only | which layers carry intent information? Strong for the frozen case; plot learned weights | M |
| E12 | **Attentive pooling** (learned attention over frames) vs mean pooling | does focusing on informative frames help or hurt? | S |
| E13 | **Layer-wise freezing schedule** (train heads first, then unfreeze top-k layers) | stability vs accuracy, compute | M |
| E14 | **Probing**: per-layer linear probes on frozen features for action / object / location | where does each slot become decodable? nice figure for the report | M |
| E15 | **Few-shot adaptation** on a few own-set clips (own speakers split) | how fast does accent shift close? | M |
Do not start these before the MVP rows in docs/10 §19 are verified.

## 7. Export notes (details in docs/05)
- Export a wrapper with pooling **inside** the graph; input `float32 [batch, samples]` (already normalized); output logits per head.
- Use dynamic axes for batch and samples; test lengths 1 s, 2 s, 4 s; compare against PyTorch (V-40).
- Time masking is off in eval mode; make sure `model.eval()` and `torch.no_grad()` are used during export. VERIFY that the export succeeds with the
  installed `transformers`/`torch` versions; if control-flow in the model blocks export, fall back to fixed example lengths with a short list of
  supported shapes, or export the encoder and do pooling in numpy (document the choice).
- int8 dynamic quantization must be evaluated, not assumed (V-41).

## 8. What to write in the report about the method (Bab 2 and Bab 4 material)
1. Why end-to-end SLU instead of ASR + NLU (fewer error-propagation points, lower latency, no transcript needed).
2. How wav2vec 2.0 learns speech representations from unlabeled audio and why that helps with a small labeled set.
3. Our fine-tuning recipe (freezing, LR groups, SpecAugment, truncation) and why each choice.
4. Honest positioning: Lugosch et al. propose pretraining for end-to-end SLU on FSC; we use a newer self-supervised Transformer encoder and evaluate on
   harder protocols. Do not attribute results to the paper that it does not report.

## 9. Acceptance criteria for "Transformer SLU done" (checks the agent must satisfy)
- [ ] `Wav2VecSLU` implemented per §3 with config-driven freezing/truncation; unit tests for shapes, freezing (V-15), eval-mode determinism (V-17).
- [ ] Preprocessing parity with `Wav2Vec2FeatureExtractor` verified by test (§4.6).
- [ ] M0 trained and evaluated per docs/02/11; HF-native baseline (P3-17) reported next to it.
- [ ] Pooling/padding behavior documented with measured differences (V-18) and the evaluation batch policy logged.
- [ ] Report text (Bab 2/4) states precisely what is HuggingFace-based, what is custom, and why.
