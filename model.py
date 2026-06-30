"""
model.py
========
The Transformer ("Attention Is All You Need", Vaswani et al. 2017) implemented
*from scratch* in PyTorch. We deliberately do NOT use nn.Transformer,
nn.MultiheadAttention, or nn.TransformerEncoder. Every block is hand-written so
you can read exactly how self-attention, cross-attention, masking and
positional encoding work.

Read top-to-bottom; the small pieces are defined first and composed upward:
    ScaledDotProductAttention -> MultiHeadAttention -> PositionwiseFeedForward
    -> EncoderLayer / DecoderLayer -> Encoder / Decoder -> Transformer
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Token embedding (scaled, as in the paper: multiply by sqrt(d_model))
# ---------------------------------------------------------------------------
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):                       # x: (B, T) integer ids
        return self.embed(x) * math.sqrt(self.d_model)   # (B, T, d_model)


# ---------------------------------------------------------------------------
# 2. Positional encoding (fixed sinusoids). Transformers have no recurrence,
#    so we inject position information by adding these sine/cosine waves.
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()      # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # even dims
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dims
        pe = pe.unsqueeze(0)                           # (1, max_len, d_model)
        self.register_buffer("pe", pe)                 # not a parameter

    def forward(self, x):                              # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 3. Scaled dot-product attention  ->  softmax(QK^T / sqrt(d_k)) V
# ---------------------------------------------------------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q,k,v: (B, heads, T, d_k)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B,h,Tq,Tk)
        if mask is not None:
            # mask is broadcastable; positions with 0 are blocked.
            scores = scores.masked_fill(mask == 0, float("-1e9"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)                                    # (B,h,Tq,d_k)
        return out, attn


# ---------------------------------------------------------------------------
# 4. Multi-head attention: project into h subspaces, attend, concat, project.
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.h = num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)

    def _split_heads(self, x):
        # (B, T, d_model) -> (B, h, T, d_k)
        B, T, _ = x.size()
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def _combine_heads(self, x):
        # (B, h, T, d_k) -> (B, T, d_model)
        B, h, T, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, h * d_k)

    def forward(self, query, key, value, mask=None):
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        # Masks arrive already shaped to broadcast over heads & query positions:
        #   src padding mask -> (B, 1, 1, Tk)
        #   target causal mask -> (B, 1, T, T)
        out, _ = self.attention(q, k, v, mask)
        out = self._combine_heads(out)
        return self.w_o(out)


# ---------------------------------------------------------------------------
# 5. Position-wise feed-forward network (applied identically at each position)
# ---------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


# ---------------------------------------------------------------------------
# 6. Encoder layer = self-attention + FFN, each wrapped in residual + LayerNorm
#    (we use the common pre-norm-friendly post-norm residual style)
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attn = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn))
        ff = self.ffn(x)
        x = self.norm2(x + self.dropout(ff))
        return x


# ---------------------------------------------------------------------------
# 7. Decoder layer = masked self-attention + cross-attention to encoder + FFN
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        # 1) masked self-attention over target so far
        s = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(s))
        # 2) cross-attention: queries from decoder, keys/values from encoder
        c = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(c))
        # 3) feed forward
        f = self.ffn(x)
        x = self.norm3(x + self.dropout(f))
        return x


# ---------------------------------------------------------------------------
# 8. Full encoder / decoder stacks
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 dropout, max_len):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, src, src_mask):
        x = self.pos(self.embed(src))
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 dropout, max_len):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.pos(self.embed(tgt))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x


# ---------------------------------------------------------------------------
# 9. The complete Transformer + mask helpers
# ---------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, num_heads=8,
                 d_ff=512, num_layers=3, dropout=0.1, max_len=128, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = Encoder(src_vocab, d_model, num_heads, d_ff,
                               num_layers, dropout, max_len)
        self.decoder = Decoder(tgt_vocab, d_model, num_heads, d_ff,
                               num_layers, dropout, max_len)
        self.generator = nn.Linear(d_model, tgt_vocab)   # project to vocab logits
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ---- masks -----------------------------------------------------------
    def make_src_mask(self, src):
        # (B, 1, 1, Tk): block padding positions in the keys
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        B, T = tgt.size()
        pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)      # (B,1,1,T)
        # lower-triangular: position i may only see <= i (no peeking ahead)
        subseq = torch.tril(torch.ones(T, T, device=tgt.device)).bool()
        subseq = subseq.unsqueeze(0).unsqueeze(0)                      # (1,1,T,T)
        return pad_mask & subseq                                      # (B,1,T,T)

    # ---- forward ---------------------------------------------------------
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc = self.encoder(src, src_mask)
        dec = self.decoder(tgt, enc, src_mask, tgt_mask)
        return self.generator(dec)            # (B, T, tgt_vocab) logits

    # ---- helpers for inference ------------------------------------------
    def encode(self, src):
        return self.encoder(src, self.make_src_mask(src))

    def decode_step(self, tgt, enc_out, src):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        dec = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.generator(dec)


if __name__ == "__main__":
    # Smoke test: random ids through the model, check output shape.
    torch.manual_seed(0)
    B, S, T = 2, 7, 9
    src_vocab, tgt_vocab = 100, 120
    model = Transformer(src_vocab, tgt_vocab, d_model=64, num_heads=4,
                        d_ff=128, num_layers=2, max_len=32, pad_id=0)
    src = torch.randint(1, src_vocab, (B, S))
    tgt = torch.randint(1, tgt_vocab, (B, T))
    src[0, -2:] = 0          # add some padding to exercise the mask
    out = model(src, tgt)
    print("output logits:", tuple(out.shape), "expected:", (B, T, tgt_vocab))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")
