#!/usr/bin/env python3
"""NextAI 对话脚本 — 加载 nextai-full.pt 进行交互式对话。"""
from __future__ import print_function

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS, UNK = 0, 1, 2, 3


class ByteTokenizer:
    def __init__(self, b2i, i2b, merges, vocab_size):
        self.b2i = b2i
        self.i2b = i2b
        self.merges = merges
        self.vocab_size = vocab_size

    def encode(self, text, max_len=None):
        data = text.encode("utf-8", errors="replace")
        tokens = [bytes([b]) for b in data]
        if not tokens:
            tokens = [bytes([ord(" ")])]
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        ids = [BOS] + [self.b2i.get(t, UNK) for t in tokens] + [EOS]
        if max_len and len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids):
        raw = b""
        for i in ids:
            if i in (BOS, EOS, PAD):
                continue
            if i in self.i2b:
                raw += self.i2b[i]
            else:
                raw += b"?"
        return raw.decode("utf-8", errors="replace")


class MultiHeadAttention(nn.Module):
    def __init__(self, d, n_heads):
        super(MultiHeadAttention, self).__init__()
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
            scores = scores - mask * 1e9
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        return self.o(out)


class EncoderLayer(nn.Module):
    def __init__(self, d, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
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
    def __init__(self, d, n_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d, n_heads)
        self.cross_attn = MultiHeadAttention(d, n_heads)
        self.ff1 = nn.Linear(d, d_ff)
        self.ff2 = nn.Linear(d_ff, d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.ln1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.ln2(x + self.drop(self.cross_attn(x, enc_out, enc_out, src_mask)))
        return self.ln3(x + self.drop(self.ff2(F.relu(self.ff1(x)))))


class NextAI(nn.Module):
    def __init__(self, cfg):
        super(NextAI, self).__init__()
        self.cfg = cfg
        self.embed_src = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.embed_tgt = nn.Embedding(cfg["vocab_size"], cfg["d_model"])

        pe = torch.zeros(cfg["max_len"], cfg["d_model"])
        pos = torch.arange(0, cfg["max_len"], dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, cfg["d_model"], 2).float() * (-math.log(10000.0) / cfg["d_model"]))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

        dropout = cfg.get("dropout", 0.05)
        self.encoder = nn.ModuleList([
            EncoderLayer(cfg["d_model"], cfg["n_heads"], cfg["d_ff"], dropout)
            for _ in range(cfg["n_layers"])
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(cfg["d_model"], cfg["n_heads"], cfg["d_ff"], dropout)
            for _ in range(cfg["n_layers"])
        ])
        self.out = nn.Linear(cfg["d_model"], cfg["vocab_size"])
        self.drop = nn.Dropout(dropout)

    def forward(self, src, tgt, src_pad_mask=None):
        B_src, T_src = src.shape
        B_tgt, T_tgt = tgt.shape
        src_emb = self.drop(self.embed_src(src) + self.pe[:, :T_src, :])
        tgt_emb = self.drop(self.embed_tgt(tgt) + self.pe[:, :T_tgt, :])

        enc_mask = None
        if src_pad_mask is not None:
            enc_mask = (src_pad_mask == 0).float().unsqueeze(1).unsqueeze(1)

        enc_out = src_emb
        for layer in self.encoder:
            enc_out = layer(enc_out, enc_mask)

        tgt_mask = torch.triu(torch.ones(T_tgt, T_tgt, dtype=torch.uint8), diagonal=1)
        tgt_mask = (tgt_mask == 0).float().unsqueeze(0).unsqueeze(0).to(tgt.device)

        dec_out = tgt_emb
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out, enc_mask, tgt_mask)

        return self.out(dec_out)

    @torch.no_grad()
    def generate(self, src_ids, max_new=60):
        """生成完整 token ids（非流式）。"""
        self.eval()
        src = torch.tensor([src_ids], dtype=torch.long).to(next(self.parameters()).device)
        src_mask = (src != PAD).long().to(next(self.parameters()).device)
        generated = [BOS]

        for step in range(max_new):
            tgt = torch.tensor([generated], dtype=torch.long).to(next(self.parameters()).device)
            logits = self.forward(src, tgt, src_mask)
            next_tok = torch.argmax(logits[0, -1, :]).item()

            if next_tok in (EOS, PAD):
                break

            if len(generated) >= 4 and all(x == next_tok for x in generated[-4:]):
                sorted_logits, sorted_idx = torch.sort(logits[0, -1, :], descending=True)
                for idx in sorted_idx[1:]:
                    candidate = idx.item()
                    if candidate not in (PAD, BOS):
                        next_tok = candidate
                        break

            generated.append(next_tok)

        return generated[1:]

    @torch.no_grad()
    def generate_stream(self, src_ids, max_new=60):
        """流式生成：每次 yield 一个 token id。"""
        self.eval()
        src = torch.tensor([src_ids], dtype=torch.long).to(next(self.parameters()).device)
        src_mask = (src != PAD).long().to(next(self.parameters()).device)
        generated = [BOS]

        for step in range(max_new):
            tgt = torch.tensor([generated], dtype=torch.long).to(next(self.parameters()).device)
            logits = self.forward(src, tgt, src_mask)
            next_tok = torch.argmax(logits[0, -1, :]).item()

            if next_tok in (EOS, PAD):
                break

            if len(generated) >= 4 and all(x == next_tok for x in generated[-4:]):
                sorted_logits, sorted_idx = torch.sort(logits[0, -1, :], descending=True)
                for idx in sorted_idx[1:]:
                    candidate = idx.item()
                    if candidate not in (PAD, BOS):
                        next_tok = candidate
                        break

            generated.append(next_tok)
            yield next_tok


def load_model(path):
    """加载模型和分词器。"""
    print("正在加载模型: {}".format(path))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    cfg = ckpt["cfg"]
    tok_data = ckpt.get("tokenizer") or ckpt.get("tokenizer_data")

    tokenizer = ByteTokenizer(
        tok_data["b2i"], tok_data["i2b"],
        tok_data["merges"], tok_data["vocab_size"]
    )

    model = NextAI(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("模型加载完成: {} 参数, vocab={}, d_model={}".format(
        n_params, cfg["vocab_size"], cfg["d_model"]))
    return model, tokenizer


def main():
    model_path = "/workspace/nextai-full.pt"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    model, tokenizer = load_model(model_path)

    print("=" * 60)
    print("NextAI 对话系统 (输入 'quit' 或 'exit' 退出)")
    print("支持: 身份问答 / 中英德翻译 / 简单问答")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q", "退出"):
            print("再见!")
            break

        # 流式生成：逐 token 解码并逐字符打印
        sys.stdout.write("NextAI: ")
        sys.stdout.flush()

        src_ids = tokenizer.encode(user_input, max_len=model.cfg["max_len"])
        token_buffer = []
        for tok_id in model.generate_stream(src_ids, max_new=60):
            token_buffer.append(tok_id)
            # 尝试从 token_buffer 解码（单 token 可能跨多个字节，UTF-8 增量解码）
            chunk = tokenizer.decode([tok_id])
            sys.stdout.write(chunk)
            sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
