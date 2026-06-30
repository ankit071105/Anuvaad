"""
config.py
=========
All knobs in one place. The defaults are intentionally SMALL so the model
trains in a reasonable time on a laptop (including Apple-Silicon MPS). Bump the
sizes up if you have a GPU and want better quality.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
DATA_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)


@dataclass
class Config:
    # --- data ---------------------------------------------------------------
    # How many sentence pairs to use. The full IITB corpus is ~1.6M pairs;
    # training on all of it from scratch on a laptop is impractical. Start
    # small, increase once the pipeline works end to end.
    max_pairs: int = 150_000     # 👈 SET FOR YOU. More data = better. Lower to 50_000 for a fast test.
    max_len: int = 40            # truncate/pad sentences to this many tokens
    min_freq: int = 2

    # --- tokenizer ----------------------------------------------------------
    src_vocab_size: int = 8000
    tgt_vocab_size: int = 8000
    src_spm: str = str(DATA_DIR / "spm_en.model")
    tgt_spm: str = str(DATA_DIR / "spm_hi.model")

    # special token ids (must match tokenizer_train.py)
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    # --- model --------------------------------------------------------------
    # 👈 CHANGE #2 (model size). Defaults below are laptop-friendly.
    #    For a GPU, use the paper's "base": d_model=512, num_heads=8,
    #    d_ff=2048, num_layers=6  -> much better, much slower.
    d_model: int = 256
    num_heads: int = 8           # must divide d_model evenly
    d_ff: int = 512
    num_layers: int = 3
    dropout: float = 0.1

    # --- training -----------------------------------------------------------
    batch_size: int = 32         # 👈 CHANGE #3: lower to 16/8 if you hit out-of-memory
    epochs: int = 12             # 👈 SET FOR YOU. Raise if perplexity is still falling at the end.
    lr: float = 5e-4
    warmup_steps: int = 2000
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42

    ckpt_path: str = str(CKPT_DIR / "transformer.pt")


cfg = Config()
