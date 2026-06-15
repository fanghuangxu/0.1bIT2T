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
    def __init__(self, b2i, i2b, merges, vocab_size):
        self.b2i = b2i
        self.i2b = i2b
        self.merges = merges
        self.vocab_size = vocab_size

    def encode(self, text, max_len=None):
        data = text.encode("utf-8", errors="ignore")
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
        text = raw.decode("utf-8", errors="ignore")
        return text


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
        self.ln_in = nn.LayerNorm(cfg["d_model"])
        self.dropout = nn.Dropout(cfg["dropout"])
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=cfg["d_model"] if i == 0 else cfg["hidden_size"],
                    hidden_size=cfg["hidden_size"] // 2,
                    num_layers=1, batch_first=True, bidirectional=True)
            for i in range(cfg["n_layers"])
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(cfg["hidden_size"]) for _ in range(cfg["n_layers"])])

    def forward(self, src, src_lengths):
        B, T = src.size()
        x = self.dropout(self.ln_in(self.embed(src)))
        packed = nn.utils.rnn.pack_padded_sequence(x, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        current = packed
        for lstm, ln in zip(self.lstms, self.layer_norms):
            out_packed, _ = lstm(current)
            out_unpacked, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
            out_unpacked = ln(self.dropout(out_unpacked) + out_unpacked)
            current = nn.utils.rnn.pack_padded_sequence(out_unpacked, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        final_out, _ = nn.utils.rnn.pad_packed_sequence(current, batch_first=True, total_length=T)
        return final_out


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, src_mask):
        # decoder_state: [B, T_dec, H], encoder_outputs: [B, T_enc, H]
        src_len = encoder_outputs.size(1)
        scores1 = self.W1(decoder_state).unsqueeze(2)
        scores2 = self.W2(encoder_outputs).unsqueeze(1)
        energy = torch.tanh(scores1.expand(-1, decoder_state.size(1), src_len, -1) + scores2.expand(-1, decoder_state.size(1), src_len, -1))
        scores = self.V(energy).squeeze(-1)
        if src_mask is not None:
            scores = scores.masked_fill(src_mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, encoder_outputs)
        return context


class LSTMDecoder(nn.Module):
    def __init__(self, cfg):
        super(LSTMDecoder, self).__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["d_model"], padding_idx=PAD)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=cfg["d_model"] if i == 0 else cfg["hidden_size"],
                    hidden_size=cfg["hidden_size"], num_layers=1, batch_first=True)
            for i in range(cfg["n_layers"])
        ])
        self.attention = AttentionLayer(cfg["hidden_size"])
        self.out_proj = nn.Linear(cfg["hidden_size"] * 2, cfg["hidden_size"])
        self.ln = nn.LayerNorm(cfg["hidden_size"])
        self.out = nn.Linear(cfg["hidden_size"], cfg["vocab_size"])

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
            # 获取第 i 层的隐藏状态: 确保形状为 (batch, hidden_size)
            hi_in = h[i]
            ci_in = c[i]
            # 如果是 3 维 (1, batch, hidden)，压缩为 2 维
            if hi_in.dim() == 3 and hi_in.size(0) == 1:
                hi_in = hi_in.squeeze(0)
                ci_in = ci_in.squeeze(0)
            # 添加 num_layers 维度: (batch, hidden) -> (1, batch, hidden)
            hx = (hi_in.unsqueeze(0).contiguous(), ci_in.unsqueeze(0).contiguous())
            current, (hi, ci) = lstm(current, hx)
            # hi, ci 形状: (1, batch, hidden) -> (batch, hidden)
            new_h.append(hi.squeeze(0).contiguous())
            new_c.append(ci.squeeze(0).contiguous())
        current = self.dropout(current)
        src_mask = (enc_out.abs().sum(dim=-1) != 0).long()
        context = self.attention(current, enc_out, src_mask)
        combined = torch.cat([current, context], dim=-1)
        output = self.ln(torch.tanh(self.out_proj(combined)))
        logits = self.out(output)
        # stack 后形状: (n_layers, batch, hidden_size)
        return logits, (torch.stack(new_h), torch.stack(new_c))


class LSTMNextAI(nn.Module):
    """LSTM 版本的 NextAI（当前训练架构）。"""
    def __init__(self, cfg):
        super(LSTMNextAI, self).__init__()
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
        self.eval()
        device = next(self.parameters()).device
        src = torch.tensor([src_ids], dtype=torch.long).to(device)
        src_lengths = torch.tensor([len(src_ids)], dtype=torch.long).to(device)
        enc_out = self.encoder(src, src_lengths)
        src_mask = (src != PAD).long()

        generated = [BOS]
        token_counts = {}

        for step in range(max_new):
            tgt_input = torch.tensor([generated], dtype=torch.long).to(device)
            logits = self.decoder(tgt_input, enc_out, src_mask)
            last_logits = logits[0, -1, :].clone()

            # 强烈抑制 UNK token — 避免乱码
            last_logits[UNK] = -1e9
            last_logits[PAD] = -1e9

            # 重复 token 惩罚：降低已出现 token 概率
            for tok_id, count in token_counts.items():
                penalty = 1.5 + 0.5 * min(count, 5)
                if last_logits[tok_id] > 0:
                    last_logits[tok_id] /= penalty
                else:
                    last_logits[tok_id] *= penalty

            # 防止连续 2 个相同 token
            if len(generated) >= 2 and generated[-1] == generated[-2]:
                last_logits[generated[-1]] -= 5.0

            # 防止 3 元组重复
            if len(generated) >= 6:
                last_tri = tuple(generated[-3:])
                for i in range(len(generated) - 6, len(generated) - 3):
                    if tuple(generated[i:i+3]) == last_tri:
                        last_logits[generated[-1]] -= 3.0
                        break

            # 选择最大概率 token（避免采样带来的随机性）
            next_tok = torch.argmax(last_logits).item()

            if next_tok in (EOS, PAD, UNK):
                break

            token_counts[next_tok] = token_counts.get(next_tok, 0) + 1
            generated.append(next_tok)

        return generated[1:]

    @torch.no_grad()
    def generate_stream(self, src_ids, max_new=80, tokenizer=None):
        """流式生成，yield 每个 token id，自动抑制 UNK/乱码。"""
        self.eval()
        device = next(self.parameters()).device
        src = torch.tensor([src_ids], dtype=torch.long).to(device)
        src_lengths = torch.tensor([len(src_ids)], dtype=torch.long).to(device)
        enc_out = self.encoder(src, src_lengths)
        src_mask = (src != PAD).long()

        h = torch.zeros(self.cfg["n_layers"], 1, self.cfg["hidden_size"]).to(device)
        c = torch.zeros(self.cfg["n_layers"], 1, self.cfg["hidden_size"]).to(device)
        prev_tok = BOS
        generated = [BOS]
        raw_bytes = b""
        token_counts = {}

        def _valid_utf8_mid(b):
            """检查字节流 b 的非尾部部分是否为有效 UTF-8。
            允许尾部有不完整的多字节序列（将被后续 token 补全）。"""
            if not b:
                return True
            try:
                b.decode('utf-8')
                return True
            except UnicodeDecodeError:
                # 找到最后一个完整有效的 UTF-8 字符边界
                for trim in range(1, min(6, len(b)) + 1):
                    try:
                        b[:-trim].decode('utf-8')
                        return True
                    except UnicodeDecodeError:
                        continue
                return False

        for step in range(max_new):
            step_in = torch.tensor([[prev_tok]], dtype=torch.long).to(device)
            logits, (h, c) = self.decoder.step(step_in, enc_out, h, c)
            last_logits = logits[0, -1, :].clone()

            # 强烈抑制 UNK 和 PAD，避免乱码
            last_logits[UNK] = -1e9
            last_logits[PAD] = -1e9

            # 重复 token 惩罚
            for tok_id, count in token_counts.items():
                penalty = 1.5 + 0.5 * min(count, 5)
                if last_logits[tok_id] > 0:
                    last_logits[tok_id] /= penalty
                else:
                    last_logits[tok_id] *= penalty

            if len(generated) >= 2 and generated[-1] == generated[-2]:
                last_logits[generated[-1]] -= 5.0
            if len(generated) >= 6:
                last_tri = tuple(generated[-3:])
                for i in range(len(generated) - 6, len(generated) - 3):
                    if tuple(generated[i:i+3]) == last_tri:
                        last_logits[generated[-1]] -= 3.0
                        break

            # 从 top-5 中选择第一个通过 UTF-8 验证的 token
            next_tok = None
            top_k = torch.topk(last_logits, k=5)
            for cand_idx in range(5):
                cand_tok = top_k.indices[cand_idx].item()
                if cand_tok in (EOS, PAD, UNK):
                    continue
                if tokenizer is not None:
                    tok_bytes = tokenizer.i2b.get(cand_tok, b'')
                    if tok_bytes and not _valid_utf8_mid(raw_bytes + tok_bytes):
                        continue
                next_tok = cand_tok
                if tokenizer is not None:
                    raw_bytes += tokenizer.i2b.get(cand_tok, b'')
                break

            # 回退：直接选最大概率
            if next_tok is None:
                next_tok = torch.argmax(last_logits).item()
                if next_tok in (EOS, PAD, UNK):
                    break
                if tokenizer is not None:
                    raw_bytes += tokenizer.i2b.get(next_tok, b'')

            if next_tok in (EOS, PAD, UNK):
                break

            token_counts[next_tok] = token_counts.get(next_tok, 0) + 1
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
            model = LSTMNextAI(cfg)
        else:
            print("检测为 Transformer 模型架构")
            model = TransformerNextAI(cfg)
    else:
        print("检测为 Transformer 模型架构")
        model = TransformerNextAI(cfg)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    arch_name = "LSTM" if isinstance(model, LSTMNextAI) else "Transformer"
    print("模型加载完成: {} ({}), {} 参数, vocab={}, d_model={}".format(
        arch_name, model_type, n_params, cfg["vocab_size"], cfg.get("d_model", cfg.get("hidden_size", "?"))))
    return model, tokenizer


def _detect_domain(text):
    """根据用户输入自动检测学科领域，添加对应的前缀标记。"""
    t = text.lower()
    # 代码领域关键词
    code_kw = ['code', 'python', 'java', 'c++', 'c语言', 'javascript',
               '编程', 'function', 'def ', 'class ', 'void ', 'int main',
               '算法', '排序', '链表', '树', '栈', '队列', 'os', '操作系统',
               'html', 'css', 'sql', 'rust', 'go语言', 'typescript', 'php',
               '程序', '编译', '源码', 'how to', 'write a', 'implement']
    # 法律领域关键词
    law_kw = ['法律', 'law', '合同', 'contract', '条例', '法规', '刑事责任',
               '民事', '判决', '法庭', '诉讼', '原告', '被告', '侵权',
               'recht', 'vertrag', 'gesetz', 'klage', 'urteil']
    # 金融领域关键词
    finance_kw = ['金融', 'finance', 'stock', '股票', '投资', '债券', '基金',
                  'bank', '银行', '利率', '汇率', '通胀', 'gdp', '期权', '期货',
                  '市值', '市盈率', '理财', 'aktie', 'anleihe', 'fonds']
    # 物理领域关键词
    physics_kw = ['物理', 'physics', '牛顿', 'newton', '相对论', '量子', 'quantum',
                  '力学', '电磁', '热力学', '熵', '核聚变', '黑洞', '引力', '万有引力',
                  'momentum', 'energie', 'gravitation', 'quanten']

    hits = {'CODE': 0, 'LAW': 0, 'FINANCE': 0, 'PHYSICS': 0}
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

    # 找出得分最高的领域
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
