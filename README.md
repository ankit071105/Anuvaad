# Anuvaad — English → Hindi, transformer built from scratch

A complete, **fully local** English-to-Hindi translator. The transformer
(encoder–decoder, multi-head self-attention, positional encoding) is
hand-written in PyTorch — no `nn.Transformer`, no translation API, nothing
leaves your machine. A small Flask app lets you use it from the browser.

This is a *learning* build. The point is understanding the architecture, not
beating Google Translate.

## What's where (map to the concepts)

| File | What it teaches |
|------|-----------------|
| `model.py` | The whole transformer from scratch: scaled dot-product attention → multi-head attention → encoder/decoder layers → masks. **Read this first.** |
| `tokenizer_train.py` | Subword (BPE) tokenization with SentencePiece — why word-level fails for Devanagari. |
| `dataset.py` | Teacher forcing, `<bos>/<eos>` wrapping, padding, the shifted-target trick. |
| `train.py` | Noam LR warmup schedule, label smoothing, gradient clipping, device pick (CUDA/MPS/CPU). |
| `translate.py` | Autoregressive inference: greedy vs. beam search decoding. |
| `app.py` + `templates/` + `static/` | Local Flask UI. |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run it (3 steps, in order)

```bash
# 1. Download the IIT Bombay corpus + train the BPE tokenizers (one time)
python tokenizer_train.py

# 2. Train the transformer (writes checkpoints/transformer.pt)
python train.py

# 3. Launch the local website
python app.py
# open http://127.0.0.1:5000
```

Quick command-line test without the browser:

```bash
python translate.py "How are you today?"
```

## Apple Silicon (M1/M2) notes

`train.py` auto-selects the MPS backend. It works, though some ops fall back to
CPU. With the default small config (50k pairs, d_model=256, 3 layers) expect
roughly a few minutes per epoch on an M1. To go faster or get better quality,
train on a free Colab/Kaggle GPU and copy `checkpoints/transformer.pt` +
`data/spm_en.model` + `data/spm_hi.model` back here — inference still runs
100% locally.

## Tuning quality (in `config.py`)

- `max_pairs` — start at 50k to verify the pipeline, then raise toward the full
  ~1.6M for real quality. This is the biggest lever.
- `d_model`, `num_layers`, `d_ff` — bigger = better but slower. Try 512 / 6 /
  2048 (the paper's "base") if you have a GPU.
- `epochs` — 10 is a starting point; watch the perplexity drop.

## Honest expectations

Trained on a laptop with a small data subset, output will be roughly grammatical
for short, common sentences and shaky on long or rare ones. That's expected for
a from-scratch model at this scale — the value here is that **every line of the
architecture is yours to read and explain.** Scale the data and model up (GPU)
and quality improves substantially.

## Why these design choices (good interview talking points)

- **Separate src/tgt BPE vocabularies** — English (Latin) and Hindi (Devanagari)
  share almost no subwords, so separate tokenizers are cleaner than a joint one.
- **Sinusoidal positional encoding** (fixed, not learned) — no extra params,
  generalizes to lengths unseen in training.
- **Noam schedule + label smoothing** — the two tricks from the original paper
  that make training actually converge well.
- **Beam search at inference** — explores multiple hypotheses; usually more
  fluent than greedy for the same model.
```
