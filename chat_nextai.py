#!/usr/bin/env python3
"""NextAI 对话脚本 — 加载 nextai-full.pt 进行交互式对话。

支持两种模型架构：
  - LSTM: NextAILSTM (LSTM seq2seq + 双向编码器 + 注意力)
  - Transformer: NextAI (原 Transformer 模型)
"""
from __future__ import print_function

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS, UNK = 0, 1, 2, 3

# ─────────────────────────────────────────────────────────────────────────────
# 分词器（两种模型共用）
# ─────────────────────────────────────────────────────────────────────────────
class ByteTokenizer:
    """纯 byte-level 分词器，与训练脚本保持一致。"""

    def __init__(self, b2i=None, i2b=None, merges=None, vocab_size=260):
        self.vocab_size = vocab_size
        self.merges = []
        self.b2i = b2i if b2i is not None else {}
        self.i2b = i2b if i2b is not None else {}

    def encode(self, text, max_len=None):
        data = text.encode("utf-8", errors="ignore")
        ids = [BOS] + [4 + b for b in data] + [EOS]
        if max_len and len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = EOS
        return ids

    def decode(self, ids):
        raw = bytearray()
        for i in ids:
            if i in (BOS, EOS, PAD, UNK):
                continue
            b = i - 4
            if 0 <= b < 256:
                raw.append(b)
        try:
            return bytes(raw).decode("utf-8", errors="ignore")
        except Exception:
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 模型（原架构）
# ─────────────────────────────────────────────────────────────────────────────
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


class TransformerNextAI(nn.Module):
    """Transformer 版本的 NextAI（legacy）。"""
    def __init__(self, cfg):
        super(TransformerNextAI, self).__init__()
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
        self.eval()
        device = next(self.parameters()).device
        src = torch.tensor([src_ids], dtype=torch.long).to(device)
        src_mask = (src != PAD).long().to(device)
        generated = [BOS]

        for step in range(max_new):
            tgt = torch.tensor([generated], dtype=torch.long).to(device)
            logits = self.forward(src, tgt, src_mask)
            next_tok = torch.argmax(logits[0, -1, :]).item()

            if next_tok in (EOS, PAD):
                break

            # 防止连续 4 个相同 token
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
        self.eval()
        device = next(self.parameters()).device
        src = torch.tensor([src_ids], dtype=torch.long).to(device)
        src_mask = (src != PAD).long().to(device)
        generated = [BOS]

        for step in range(max_new):
            tgt = torch.tensor([generated], dtype=torch.long).to(device)
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


# ─────────────────────────────────────────────────────────────────────────────
# LSTM 模型（当前训练的主架构）
# ─────────────────────────────────────────────────────────────────────────────
class BiLSTMEncoder(nn.Module):
    def __init__(self, cfg):
        super(BiLSTMEncoder, self).__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["d_model"], padding_idx=PAD)
        self.dropout = nn.Dropout(cfg["dropout"])
        h = cfg["hidden_size"]
        lstms = []
        in_size = cfg["d_model"]
        for _ in range(cfg["n_layers"]):
            lstms.append(nn.LSTM(in_size, h // 2, num_layers=1, batch_first=True, bidirectional=True))
            in_size = h
        self.lstms = nn.ModuleList(lstms)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(cfg["n_layers"])])

    def forward(self, src, src_lengths):
        x = self.dropout(self.embed(src))
        for lstm, ln in zip(self.lstms, self.layer_norms):
            out, _ = lstm(x)
            out = ln(self.dropout(out) + out)
            x = out
        return x


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, dec_hidden, enc_out, mask):
        scores1 = self.W1(dec_hidden).unsqueeze(2)
        scores2 = self.W2(enc_out).unsqueeze(1)
        energy = torch.tanh(scores1 + scores2)
        scores = self.V(energy).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, enc_out)
        return context


class LSTMDecoder(nn.Module):
    def __init__(self, cfg):
        super(LSTMDecoder, self).__init__()
        self.cfg = cfg
        h = cfg["hidden_size"]
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["d_model"], padding_idx=PAD)
        self.dropout = nn.Dropout(cfg["dropout"])
        in_size = cfg["d_model"]
        self.lstms = nn.ModuleList([
            nn.LSTM(in_size if i == 0 else h, h, num_layers=1, batch_first=True)
            for i in range(cfg["n_layers"])
        ])
        self.attention = AdditiveAttention(h)
        self.out_proj = nn.Linear(h * 2, h)
        self.ln = nn.LayerNorm(h)
        self.out = nn.Linear(h, cfg["vocab_size"])

    def forward(self, tgt_input, enc_out, src_mask):
        B, T = tgt_input.size()
        x = self.dropout(self.embed(tgt_input))
        current = x
        for lstm in self.lstms:
            current, _ = lstm(current)
            current = self.dropout(current)
        context = self.attention(current, enc_out, src_mask)
        combined = torch.cat([current, context], dim=-1)
        output = self.ln(torch.tanh(self.out_proj(combined)))
        return self.out(output)

    @torch.no_grad()
    def step(self, tgt_input, enc_out, h, c):
        """单步解码（用于增量生成）。

        期望 h, c 形状: (n_layers, batch, hidden_size)
        返回 h, c 形状: (n_layers, batch, hidden_size)
        """
        x = self.embed(tgt_input)
        current = x
        new_h = []
        new_c = []
        for i, lstm in enumerate(self.lstms):
            hi_in = h[i]
            ci_in = c[i]
            if hi_in.dim() == 3 and hi_in.size(0) == 1:
                hi_in = hi_in.squeeze(0)
                ci_in = ci_in.squeeze(0)
            hx = (hi_in.unsqueeze(0).contiguous(), ci_in.unsqueeze(0).contiguous())
            current, (hi, ci) = lstm(current, hx)
            new_h.append(hi.squeeze(0).contiguous())
            new_c.append(ci.squeeze(0).contiguous())
        current = self.dropout(current)
        src_mask = (enc_out.abs().sum(dim=-1) != 0).long()
        context = self.attention(current, enc_out, src_mask)
        combined = torch.cat([current, context], dim=-1)
        output = self.ln(torch.tanh(self.out_proj(combined)))
        logits = self.out(output)
        return logits, (torch.stack(new_h), torch.stack(new_c))


class NextAILSTM(nn.Module):
    """LSTM 版本的 NextAI（当前训练架构）。"""
    def __init__(self, cfg):
        super(NextAILSTM, self).__init__()
        self.cfg = cfg
        self.encoder = BiLSTMEncoder(cfg)
        self.decoder = LSTMDecoder(cfg)

    def forward(self, src, src_lengths, tgt):
        enc_out = self.encoder(src, src_lengths)
        src_mask = (src != PAD).long()
        logits = self.decoder(tgt[:, :-1], enc_out, src_mask)
        return logits

    @torch.no_grad()
    def generate(self, src_ids, max_new=80):
        """贪心解码，EOS 终止。"""
        self.eval()
        device = next(self.parameters()).device
        src = torch.tensor([src_ids], dtype=torch.long).to(device)
        src_lengths = torch.tensor([len(src_ids)], dtype=torch.long)
        enc_out = self.encoder(src, src_lengths)

        generated = [BOS]
        h = torch.zeros(self.cfg["n_layers"], 1, self.cfg["hidden_size"]).to(device)
        c = torch.zeros(self.cfg["n_layers"], 1, self.cfg["hidden_size"]).to(device)
        prev_tok = BOS

        for step in range(max_new):
            step_in = torch.tensor([[prev_tok]], dtype=torch.long).to(device)
            logits, (h, c) = self.decoder.step(step_in, enc_out, h, c)
            next_tok = torch.argmax(logits[0, -1, :]).item()
            if next_tok in (EOS, PAD, UNK):
                break
            generated.append(next_tok)
            prev_tok = next_tok
        return generated[1:]

    @torch.no_grad()
    def generate_stream(self, src_ids, max_new=120, tokenizer=None):
        """流式生成，抑制 PAD/UNK，EOS 终止。"""
        self.eval()
        device = next(self.parameters()).device
        src = torch.tensor([src_ids], dtype=torch.long).to(device)
        src_lengths = torch.tensor([len(src_ids)], dtype=torch.long).to(device)
        enc_out = self.encoder(src, src_lengths)

        h = torch.zeros(self.cfg["n_layers"], 1, self.cfg["hidden_size"]).to(device)
        c = torch.zeros(self.cfg["n_layers"], 1, self.cfg["hidden_size"]).to(device)
        prev_tok = BOS
        generated = [BOS]

        for step in range(max_new):
            step_in = torch.tensor([[prev_tok]], dtype=torch.long).to(device)
            logits, (h, c) = self.decoder.step(step_in, enc_out, h, c)
            last_logits = logits[0, -1, :].clone()

            # 抑制 PAD/UNK 避免乱码
            last_logits[UNK] = -1e9
            last_logits[PAD] = -1e9

            next_tok = torch.argmax(last_logits).item()
            if next_tok in (EOS, PAD, UNK):
                break
            generated.append(next_tok)
            prev_tok = next_tok
            yield next_tok


# ─────────────────────────────────────────────────────────────────────────────
# 统一加载接口
# ─────────────────────────────────────────────────────────────────────────────
def load_model(path):
    """根据 checkpoint 自动检测模型架构并加载。"""
    print("正在加载模型: {}".format(path))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    cfg = ckpt["cfg"]
    tok_data = ckpt.get("tokenizer") or ckpt.get("tokenizer_data")

    tokenizer = ByteTokenizer(
        tok_data["b2i"], tok_data["i2b"],
        tok_data["merges"], tok_data["vocab_size"]
    )

    # 根据配置自动选择架构
    model_type = ckpt.get("model_type", "")
    if model_type == "lstm" or "hidden_size" in cfg:
        # LSTM 模型：没有 d_ff/n_heads，或 hidden_size 较小
        if "d_ff" not in cfg or cfg.get("d_ff", 0) == 0:
            print("检测为 LSTM 模型架构")
            model = NextAILSTM(cfg)
        else:
            print("检测为 Transformer 模型架构")
            model = TransformerNextAI(cfg)
    else:
        print("检测为 Transformer 模型架构")
        model = TransformerNextAI(cfg)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    arch_name = "LSTM" if isinstance(model, NextAILSTM) else "Transformer"
    print("模型加载完成: {} ({}), {} 参数, vocab={}, d_model={}".format(
        arch_name, model_type, n_params, cfg["vocab_size"], cfg.get("d_model", cfg.get("hidden_size", "?"))))
    return model, tokenizer


def _detect_domain(text):
    """根据用户输入自动检测学科领域，添加对应的前缀标记。"""
    t = text.lower()
    # 强翻译关键词：这些词单独出现就足以表明是翻译任务
    strong_translate = ['translate to', '翻译成', 'übersetze', 'translation', '翻译']
    for kw in strong_translate:
        if kw in t:
            return "[TRANSLATE] " + text

    # 身份/自我介绍
    identity_kw = ['你是谁', '你的名字', 'who are you', 'what is your name',
                   'are you human', '你是人类', '你叫什么', '介绍一下你自己',
                   '介绍自己', '你是由谁开发的', 'who created you',
                   '再见', 'goodbye',
                   '你好', 'hello', 'hi', '嗨',
                   '我是', 'my name']
    # 翻译（较弱，需要更多上下文）
    translate_kw = ['translate', '英文', '中文', '德语']
    # 代码领域
    code_kw = ['code', 'python', 'java', 'c++', 'c语言', 'javascript',
               '编程', 'function', 'def ', 'class ', 'void ', 'int main',
               '算法', '排序', '链表', '树', '栈', '队列', 'os', '操作系统',
               'html', 'css', 'sql', 'rust', 'go语言', 'typescript', 'php',
               '程序', '编译', '源码', 'how to', 'write a', 'implement',
               '写一个', '如何写', 'write an', 'how do you']
    # 法律领域
    law_kw = ['法律', 'law', '合同', 'contract', '条例', '法规', '刑事责任',
               '民事', '判决', '法庭', '诉讼', '原告', '被告', '侵权',
               'recht', 'vertrag', 'gesetz', 'klage', 'urteil']
    # 金融领域
    finance_kw = ['金融', 'finance', 'stock', '股票', '投资', '债券', '基金',
                  'bank', '银行', '利率', '汇率', '通胀', 'gdp', '期权', '期货',
                  '市值', '市盈率', '理财', 'aktie', 'anleihe', 'fonds']
    # 物理领域
    physics_kw = ['物理', 'physics', '牛顿', 'newton', '相对论', '量子', 'quantum',
                  '力学', '电磁', '热力学', '熵', '核聚变', '黑洞', '引力', '万有引力',
                  'momentum', 'energie', 'gravitation', 'quanten']

    hits = {'IDENTITY': 0, 'TRANSLATE': 0, 'CODE': 0, 'LAW': 0, 'FINANCE': 0, 'PHYSICS': 0}
    for kw in identity_kw:
        if kw in t:
            hits['IDENTITY'] += 1
    for kw in translate_kw:
        if kw in t:
            hits['TRANSLATE'] += 1
    for kw in code_kw:
        if kw in t:
            hits['CODE'] += 1
    for kw in law_kw:
        if kw in t:
            hits['LAW'] += 1
    for kw in finance_kw:
        if kw in t:
            hits['FINANCE'] += 1
    for kw in physics_kw:
        if kw in t:
            hits['PHYSICS'] += 1

    best_domain = max(hits.items(), key=lambda x: x[1])
    if best_domain[1] >= 1:
        return "[{}] ".format(best_domain[0]) + text
    return text


def main():
    model_path = "/workspace/nextai-full.pt"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    model, tokenizer = load_model(model_path)

    print("=" * 60)
    print("NextAI 对话系统 (输入 'quit' 或 'exit' 退出)")
    print("支持: 身份问答 / 中英德翻译 / 简单问答 / 代码 / 法律 / 金融 / 物理")
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

        # 自动检测学科领域并添加前缀
        prefixed_input = _detect_domain(user_input)

        sys.stdout.write("NextAI: ")
        sys.stdout.flush()

        src_ids = tokenizer.encode(prefixed_input, max_len=model.cfg["max_len"])

        # 累积所有 token ID，结束后整体解码 — 避免单 token UTF-8 乱码
        generated_ids = []
        for tok_id in model.generate_stream(src_ids, max_new=120):
            generated_ids.append(tok_id)

        # 整体解码：errors='ignore' 会舍弃无效 UTF-8 字节，确保无 � 乱码
        if generated_ids:
            text = tokenizer.decode(generated_ids)
            if not text.strip():
                sys.stdout.write("(模型暂无有效输出)")
            else:
                sys.stdout.write(text)
        else:
            sys.stdout.write("(模型暂无有效输出)")

        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
