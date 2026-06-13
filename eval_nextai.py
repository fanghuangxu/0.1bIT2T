"""Evaluate NextAI model with various prompts."""
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS, UNK = 0, 1, 2, 3


class ByteTokenizer:
    """Maps each byte to 4..259; then merges frequent pairs up to VOCAB-1."""

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.b2i = {bytes([i]): i + 4 for i in range(256)}
        self.i2b: dict[int, bytes] = {v: k for k, v in self.b2i.items()}
        self.merges: list[tuple[bytes, bytes]] = []

    @staticmethod
    def _get_counts(tokens: list[bytes]) -> dict[tuple[bytes, bytes], int]:
        counts: dict[tuple[bytes, bytes], int] = {}
        for i in range(len(tokens) - 1):
            counts[(tokens[i], tokens[i + 1])] = counts.get((tokens[i], tokens[i + 1]), 0) + 1
        return counts

    def learn(self, texts: list, max_merges: int | None = None) -> None:
        if max_merges is None:
            max_merges = self.vocab_size - 260
        sequences: list[list[bytes]] = []
        for t in texts:
            data = t.encode("utf-8", errors="replace")
            sequences.append([bytes([b]) for b in data])

        for step in range(max_merges):
            pair_counts: dict[tuple[bytes, bytes], int] = {}
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] = pair_counts.get((seq[i], seq[i + 1]), 0) + 1
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)
            if pair_counts[best] < 2:
                break
            new_tok = best[0] + best[1]
            new_id = max(self.b2i.values()) + 1
            self.b2i[new_tok] = new_id
            self.i2b[new_id] = new_tok
            self.merges.append(best)
            # apply merge
            new_seqs: list[list[bytes]] = []
            for seq in sequences:
                out: list[bytes] = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best:
                        out.append(new_tok)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                new_seqs.append(out)
            sequences = new_seqs

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        data = text.encode("utf-8", errors="replace")
        tokens: list[bytes] = [bytes([b]) for b in data]
        # apply merges greedily (repeat pass)
        for _ in range(4):
            changed = False
            for pair in self.merges:
                new_seq: list[bytes] = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                        new_seq.append(pair[0] + pair[1])
                        i += 2
                        changed = True
                    else:
                        new_seq.append(tokens[i])
                        i += 1
                tokens = new_seq
            if not changed:
                break
        max_id = self.vocab_size - 1
        ids: list[int] = []
        for tok in tokens:
            i = self.b2i.get(tok, UNK)
            if i > max_id:
                i = UNK
            ids.append(i)
        if max_len is not None and len(ids) > max_len - 2:
            ids = ids[: max_len - 2]
        return [BOS] + ids + [EOS]

    def decode(self, ids: list[int]) -> str:
        buf: bytearray = bytearray()
        for i in ids:
            if i == BOS or i == EOS or i == PAD:
                continue
            tok = self.i2b.get(i, b"\xef\xbf\xbd")
            buf.extend(tok)
        return buf.decode("utf-8", errors="replace")


from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    d_model: int = 160
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    max_len: int = 160
    dropout: float = 0.1


class MultiHeadAttention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)

    def forward(self, xq, xk, xv, mask=None):
        B, T, D = xq.shape
        q = self.q(xq).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k(xk).view(B, xk.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        v = self.v(xv).view(B, xv.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        return self.o(out)


class EncoderLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(d, n_heads)
        self.ff1 = nn.Linear(d, d_ff)
        self.ff2 = nn.Linear(d_ff, d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.drop(self.attn(x, x, x, mask)))
        return self.ln2(x + self.drop(self.ff2(F.relu(self.ff1(x)))))


class DecoderLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d, n_heads)
        self.cross_attn = MultiHeadAttention(d, n_heads)
        self.ff1 = nn.Linear(d, d_ff)
        self.ff2 = nn.Linear(d_ff, d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc, self_mask, cross_mask=None):
        x = self.ln1(x + self.drop(self.self_attn(x, x, x, self_mask)))
        x = self.ln2(x + self.drop(self.cross_attn(x, enc, enc, cross_mask)))
        return self.ln3(x + self.drop(self.ff2(F.relu(self.ff1(x)))))


class NextAI(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD)
        self.pos_enc = nn.Embedding(cfg.max_len, cfg.d_model)
        self.encoder = nn.ModuleList([EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)])
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, src, tgt_in, src_pad_mask=None):
        B, T_s = src.shape
        B2, T_t = tgt_in.shape
        pos_s = torch.arange(T_s, device=src.device).unsqueeze(0).expand(B, -1)
        pos_t = torch.arange(T_t, device=tgt_in.device).unsqueeze(0).expand(B, -1)
        x = self.emb(src) + self.pos_enc(pos_s)
        enc_mask = src_pad_mask.unsqueeze(1).unsqueeze(1) if src_pad_mask is not None else None
        for layer in self.encoder:
            x = layer(x, enc_mask)
        enc = x
        y = self.emb(tgt_in) + self.pos_enc(pos_t)
        causal = torch.tril(torch.ones(T_t, T_t, device=tgt_in.device)).unsqueeze(0).unsqueeze(0)
        for layer in self.decoder:
            y = layer(y, enc, causal, enc_mask)
        return self.head(y)

    @torch.no_grad()
    def generate(self, src_ids: list[int], max_new: int = 48, device="cpu") -> list[int]:
        self.eval()
        src = torch.tensor([src_ids], device=device, dtype=torch.long)
        src_pad_mask = (src != PAD).long()
        B, T_s = src.shape
        pos_s = torch.arange(T_s, device=device).unsqueeze(0)
        x = self.emb(src) + self.pos_enc(pos_s)
        enc_mask = src_pad_mask.unsqueeze(1).unsqueeze(1)
        for layer in self.encoder:
            x = layer(x, enc_mask)
        enc = x
        generated = [BOS]
        for step in range(max_new):
            tgt = torch.tensor([generated], device=device, dtype=torch.long)
            T_t = tgt.shape[1]
            pos_t = torch.arange(T_t, device=device).unsqueeze(0)
            y = self.emb(tgt) + self.pos_enc(pos_t)
            causal = torch.tril(torch.ones(T_t, T_t, device=device)).unsqueeze(0).unsqueeze(0)
            for layer in self.decoder:
                y = layer(y, enc, causal, enc_mask)
            logits = self.head(y)[:, -1, :]
            logits[:, BOS] = -1e18
            logits[:, UNK] = -1e9
            if step < 3:
                logits[:, EOS] = -1e18
            tok = int(logits.argmax(dim=-1).item())
            generated.append(tok)
            if tok == EOS:
                break
        return generated


def main():
    import pickle

    device = torch.device("cpu")

    # Load tokenizer from cache
    tok = ByteTokenizer(vocab_size=2048)
    try:
        with open("tokenizer_cache.pkl", "rb") as f:
            cache = pickle.load(f)
            tok.b2i = cache["b2i"]
            tok.i2b = cache["i2b"]
            tok.merges = cache["merges"]
            tok.vocab_size = cache["vocab_size"]
        print("Loaded tokenizer from cache")
    except Exception as e:
        print(f"Failed to load tokenizer cache: {e}")
        return

    # Load model
    cfg = ModelConfig(vocab_size=2048, d_model=160, n_heads=4, n_layers=2, d_ff=256, max_len=160, dropout=0.1)
    model = NextAI(cfg).to(device)

    try:
        checkpoint = torch.load("NextAI-rz.pt", map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded model from NextAI-rz.pt")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()

    # Test prompts
    test_prompts = [
        ("What is your name?", "English name question"),
        ("你是谁？", "Chinese identity question"),
        ("Wer bist du?", "German identity question"),
        ("Hello", "English greeting"),
        ("你好", "Chinese greeting"),
        ("Guten Tag", "German greeting"),
        ("translate to English: 你好", "Translation EN<-ZH"),
        ("übersetze ins Deutsche: Hello", "Translation DE<-EN"),
    ]

    print("\n" + "=" * 60)
    print("NextAI Evaluation Results")
    print("=" * 60)

    for prompt, desc in test_prompts:
        print(f"\n[{desc}]")
        print(f"Prompt: {prompt}")
        ids = tok.encode(prompt, max_len=cfg.max_len)
        out_ids = model.generate(ids, max_new=64, device=device)
        out_text = tok.decode(out_ids)
        print(f"Output: {out_text}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
