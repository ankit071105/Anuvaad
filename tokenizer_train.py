"""
tokenizer_train.py
==================
Step 1 of the pipeline.

We keep the *transformer* fully from scratch, but for tokenization we use
SentencePiece (a standard, tiny dependency) because subword tokenization is
essential for Hindi/Devanagari — a plain word vocabulary explodes and can't
handle unseen word forms. We train two separate BPE models (English & Hindi).

This script:
  1. loads the IIT Bombay English-Hindi parallel corpus (via HuggingFace
     `datasets`, downloaded once and cached locally),
  2. writes raw text files,
  3. trains two SentencePiece models with reserved ids:
        0=<pad> 1=<unk> 2=<bos> 3=<eos>

Run:  python tokenizer_train.py
"""
import sentencepiece as spm
from datasets import load_dataset

from config import cfg, DATA_DIR


def dump_corpus():
    print("Loading IIT Bombay en-hi corpus (first run downloads & caches)...")
    ds = load_dataset("cfilt/iitb-english-hindi", split="train")
    # Shuffle first: the corpus is ordered and front-loaded with software/UI
    # localization text. Same seed as the dataset so both see the same sample.
    ds = ds.shuffle(seed=cfg.seed)
    n = min(cfg.max_pairs, len(ds))
    print(f"Using {n:,} of {len(ds):,} pairs for tokenizer training.")

    en_path = DATA_DIR / "train.en"
    hi_path = DATA_DIR / "train.hi"
    with open(en_path, "w", encoding="utf-8") as fe, \
         open(hi_path, "w", encoding="utf-8") as fh:
        for i in range(n):
            pair = ds[i]["translation"]
            en, hi = pair["en"].strip(), pair["hi"].strip()
            if en and hi:
                fe.write(en + "\n")
                fh.write(hi + "\n")
    return en_path, hi_path


def train_spm(input_file, model_prefix, vocab_size):
    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=cfg.pad_id, unk_id=cfg.unk_id,
        bos_id=cfg.bos_id, eos_id=cfg.eos_id,
        pad_piece="<pad>", unk_piece="<unk>",
        bos_piece="<bos>", eos_piece="<eos>",
    )
    print(f"  wrote {model_prefix}.model")


if __name__ == "__main__":
    en_path, hi_path = dump_corpus()
    print("Training English SentencePiece...")
    train_spm(en_path, DATA_DIR / "spm_en", cfg.src_vocab_size)
    print("Training Hindi SentencePiece...")
    train_spm(hi_path, DATA_DIR / "spm_hi", cfg.tgt_vocab_size)
    print("Done. Tokenizers are in ./data/")
