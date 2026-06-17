#!/usr/bin/env python3
"""NextAI 对话脚本 — 加载 .pt 模型进行交互式对话。

功能:
  - 自动加载 LSTM / Transformer 架构的 .pt 模型
  - 流式 token 输出（逐字显示）
  - 多人对话（通过 @用户名 切换/维护各自上下文）
  - 快捷命令: /clear 清除上下文, /exit 退出, /users 查看用户列表, /who 查看当前用户
"""
from __future__ import print_function

import argparse
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
# Transformer 模型（legacy）
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
    def generate_stream(self, src_ids, max_new=120, tokenizer=None):
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
# LSTM 模型（当前训练架构）
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
        """单步解码（用于增量生成）。"""
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

    if tok_data is None:
        tokenizer = ByteTokenizer()
    else:
        tokenizer = ByteTokenizer(
            tok_data.get("b2i"), tok_data.get("i2b"),
            tok_data.get("merges"), tok_data.get("vocab_size", 260)
        )

    model_type = ckpt.get("model_type", "")
    if model_type == "lstm" or ("hidden_size" in cfg and cfg.get("d_ff", 0) == 0):
        print("检测为 LSTM 模型架构")
        model = NextAILSTM(cfg)
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


# ─────────────────────────────────────────────────────────────────────────────
# 上下文 / 多人对话管理
# ─────────────────────────────────────────────────────────────────────────────
class ConversationManager:
    """维护多个用户的多轮对话历史。

    - 每个用户有独立的上下文（用于拼接历史对话）
    - 双限制策略: 同时按 turn 数 + 字符数截断，避免 prompt 过长
    - 支持系统提示词、撤销最后一轮、保存加载对话
    """

    def __init__(self, max_history_chars=2000, max_turns=10):
        self.users = {}                    # user -> list of (role, text)
        self.user_persona = {}             # user -> 系统提示词
        self.current_user = "我"            # 默认用户名
        self.max_history_chars = max_history_chars
        self.max_turns = max_turns

    # ── 用户管理 ─────────────────────────────────────────────────
    def switch_user(self, name):
        """切换当前用户（首次使用时自动创建会话）。"""
        if name not in self.users:
            self.users[name] = []
            self.user_persona[name] = ""
        self.current_user = name

    def has_user(self, name):
        return name in self.users

    def list_users(self):
        return list(self.users.keys())

    # ── 对话管理 ─────────────────────────────────────────────────
    def add_turn(self, user, user_text, ai_text):
        """追加一轮对话（U + A）。"""
        if user not in self.users:
            self.users[user] = []
        self.users[user].append(("U", user_text))
        self.users[user].append(("A", ai_text))
        self._trim(user)

    def undo_last(self, user=None):
        """撤销最后一轮（即最后一组 U+A）。返回 True 表示成功。"""
        target = user or self.current_user
        turns = self.users.get(target, [])
        if len(turns) >= 2:
            turns.pop()  # A
            turns.pop()  # U
            return True
        # 只有一条的情况（比如用户刚输入但 AI 没回复），清空
        if len(turns) == 1:
            turns.pop()
            return True
        return False

    def clear(self, user=None):
        """清除某个用户的上下文；不指定则清除当前用户。"""
        target = user or self.current_user
        if target in self.users:
            self.users[target] = []
            return True
        return False

    def clear_all_users(self):
        """清除所有用户的上下文（不删除用户记录）。"""
        for name in self.users:
            self.users[name] = []

    # ── 系统提示词 ────────────────────────────────────────────────
    def set_persona(self, persona, user=None):
        """设置某个用户的系统提示词（persona / 身份设定）。"""
        target = user or self.current_user
        if target not in self.user_persona:
            self.user_persona[target] = ""
        self.user_persona[target] = persona.strip()

    def get_persona(self, user=None):
        target = user or self.current_user
        return self.user_persona.get(target, "")

    # ── 统计 ─────────────────────────────────────────────────────
    def turn_count(self, user=None):
        target = user or self.current_user
        # 每轮是一组 (U, A)，所以除以 2
        return len(self.users.get(target, [])) // 2

    def char_count(self, user=None):
        target = user or self.current_user
        return sum(len(t) for _, t in self.users.get(target, []))

    # ── 截断策略 ─────────────────────────────────────────────────
    def _trim(self, user):
        """先按 turn 数截断，再按字符数截断。优先保留最近的对话。"""
        turns = self.users[user]

        # 1. turn 数限制：max_turns 对 (U,A)；超出从头部删除
        while len(turns) // 2 > self.max_turns:
            turns.pop(0)
            turns.pop(0)

        # 2. 字符数限制：总字符超出上限则从头部删除整轮
        while True:
            total = sum(len(t) for _, t in turns)
            if total <= self.max_history_chars:
                break
            if len(turns) <= 2:
                break
            turns.pop(0)
            turns.pop(0)

    # ── 生成 prompt ──────────────────────────────────────────────
    def build_prompt(self, user, new_text):
        """将系统提示词 + 历史对话 + 用户新输入拼接成完整 prompt。

        格式:
          [SYSTEM: 你是NextAI...]
          [HISTORY: 用户: xxx | NextAI: yyy | 用户: zzz]
          用户的新问题
        """
        turns = self.users.get(user, [])
        persona = self.user_persona.get(user, "")

        parts = []

        # 系统提示词（persona）
        if persona:
            parts.append("[SYSTEM: {}]".format(persona))

        # 对话历史
        if turns:
            history_parts = []
            for role, text in turns:
                speaker = user if role == "U" else "NextAI"
                history_parts.append("{}:{}".format(speaker, text))
            parts.append("[HISTORY: {}]".format(" | ".join(history_parts)))

        # 用户新输入
        parts.append(new_text)
        return " ".join(parts)

    def pretty_history(self, user=None):
        """将历史格式化为可读字符串（用于 /history 命令）。"""
        target = user or self.current_user
        turns = self.users.get(target, [])
        if not turns:
            return "（空对话）"
        lines = []
        # 每两轮作为一个 turn
        for i in range(0, len(turns), 2):
            turn_idx = (i // 2) + 1
            u_msg = turns[i][1] if i < len(turns) else ""
            a_msg = turns[i + 1][1] if (i + 1) < len(turns) else ""
            lines.append("[Turn {}] {}: {}".format(turn_idx, target, u_msg))
            if a_msg:
                lines.append("[Turn {}] NextAI: {}".format(turn_idx, a_msg))
        return "\n".join(lines)

    # ── 持久化 ───────────────────────────────────────────────────
    def to_dict(self):
        return {
            "users": self.users,
            "persona": self.user_persona,
            "current": self.current_user,
            "max_chars": self.max_history_chars,
            "max_turns": self.max_turns,
        }

    def from_dict(self, data):
        """从 dict 恢复对话。"""
        if isinstance(data, dict):
            self.users = data.get("users", {})
            self.user_persona = data.get("persona", {})
            self.current_user = data.get("current", "我")
            self.max_history_chars = data.get("max_chars", self.max_history_chars)
            self.max_turns = data.get("max_turns", self.max_turns)


def _detect_domain(text):
    """根据用户输入自动检测学科领域，添加对应的前缀标记。"""
    t = text.lower()
    strong_translate = ['translate to', '翻译成', 'übersetze', 'translation', '翻译']
    for kw in strong_translate:
        if kw in t:
            return "[TRANSLATE] " + text

    identity_kw = ['你是谁', '你的名字', 'who are you', 'what is your name',
                   'are you human', '你是人类', '你叫什么', '介绍一下你自己',
                   '介绍自己', '你是由谁开发的', 'who created you',
                   '再见', 'goodbye',
                   '你好', 'hello', 'hi', '嗨',
                   '我是', 'my name']
    translate_kw = ['translate', '英文', '中文', '德语']
    code_kw = ['code', 'python', 'java', 'c++', 'c语言', 'javascript',
               '编程', 'function', 'def ', 'class ', 'void ', 'int main',
               '算法', '排序', '链表', '树', '栈', '队列', 'os', '操作系统',
               'html', 'css', 'sql', 'rust', 'go语言', 'typescript', 'php',
               '程序', '编译', '源码', 'how to', 'write a', 'implement',
               '写一个', '如何写', 'write an', 'how do you']
    law_kw = ['法律', 'law', '合同', 'contract', '条例', '法规', '刑事责任',
               '民事', '判决', '法庭', '诉讼', '原告', '被告', '侵权',
               'recht', 'vertrag', 'gesetz', 'klage', 'urteil']
    finance_kw = ['金融', 'finance', 'stock', '股票', '投资', '债券', '基金',
                  'bank', '银行', '利率', '汇率', '通胀', 'gdp', '期权', '期货',
                  '市值', '市盈率', '理财', 'aktie', 'anleihe', 'fonds']
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


# ─────────────────────────────────────────────────────────────────────────────
# 流式输出辅助
# ─────────────────────────────────────────────────────────────────────────────
def stream_generate_and_print(model, tokenizer, prompt, max_new=120):
    """流式生成并逐 token 解码到 stdout，累积后整体 decode 保证 UTF-8 不乱码。

    返回 (生成的文本, 生成的 token id 列表)。
    """
    src_ids = tokenizer.encode(prompt, max_len=model.cfg["max_len"])
    generated_ids = []
    prev_text_len = 0

    for tok_id in model.generate_stream(src_ids, max_new=max_new):
        generated_ids.append(tok_id)
        # 尝试增量解码 — 累积到一定数量后输出一次文本增量
        text_so_far = tokenizer.decode(generated_ids)
        if len(text_so_far) > prev_text_len:
            delta = text_so_far[prev_text_len:]
            sys.stdout.write(delta)
            sys.stdout.flush()
            prev_text_len = len(text_so_far)

    # 结束时再次整体 decode，输出剩余部分
    final_text = tokenizer.decode(generated_ids)
    if len(final_text) > prev_text_len:
        sys.stdout.write(final_text[prev_text_len:])
        sys.stdout.flush()

    return final_text, generated_ids


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
def print_banner():
    print("=" * 60)
    print("NextAI 对话系统 — 由 Next Studio 开发")
    print("支持: 身份问答 / 中英德翻译 / 代码 / 法律 / 金融 / 物理")
    print("-" * 60)
    print("多人对话: 输入 '@用户名 你好' 切换/创建用户上下文")
    print("快捷命令:")
    print("  /clear          - 清除当前用户的对话上下文")
    print("  /exit           - 退出程序 (也可用 quit / exit / q / 退出)")
    print("  /who            - 查看当前用户")
    print("  /users          - 查看所有用户的上下文状态")
    print("  /history         - 查看当前用户的多轮对话历史")
    print("  /undo            - 撤销最后一轮对话 (U+A)")
    print("  /status         - 显示当前会话状态统计")
    print("  /persona 内容    - 设置系统提示词 (例: /persona 你是一个Python专家)")
    print("  /save 文件路径    - 保存对话到文件")
    print("  /load 文件路径    - 从文件加载对话")
    print("=" * 60)


# 辅助函数: 解析 "命令 参数"
def _split_cmd(line):
    """把 '/cmd arg1 arg2' -> ('cmd', 'arg1 arg2') 或 ('cmd', '')"""
    parts = line.strip().split(None, 1)
    if not parts:
        return "", ""
    cmd = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    return cmd, rest


def handle_commands(line, cm):
    """处理快捷命令；返回 (handled, should_exit)。"""
    raw = line.strip()
    cmd, arg = _split_cmd(raw)

    # 纯退出类
    if cmd in ("/exit", "quit", "exit", "q", "退出"):
        print("再见!")
        return True, True

    if cmd == "/clear":
        cm.clear()
        print("[系统] 已清除用户 '{}' 的对话上下文".format(cm.current_user))
        return True, False

    if cmd == "/reset":
        cm.clear_all_users()
        print("[系统] 已重置所有用户对话")
        return True, False

    if cmd == "/who":
        print("[系统] 当前用户: {}".format(cm.current_user))
        print("[系统] 对话轮次: {}  |  历史字符数: {}".format(
            cm.turn_count(), cm.char_count()))
        persona = cm.get_persona()
        if persona:
            print("[系统] 系统提示词: {}".format(persona))
        return True, False

    if cmd == "/users":
        users = cm.list_users()
        if not users:
            print("[系统] 暂无用户会话")
        else:
            print("[系统] 已有用户会话:")
            for u in users:
                turns = cm.turn_count(u)
                chars = cm.char_count(u)
                marker = "  <- 当前" if u == cm.current_user else ""
                print("  - {}{} ({} 轮, {} 字符)".format(u, marker, turns, chars))
        return True, False

    if cmd == "/history":
        history = cm.pretty_history()
        print("[系统] 用户 '{}' 的对话历史:".format(cm.current_user))
        print(history)
        return True, False

    if cmd == "/undo":
        if cm.undo_last():
            print("[系统] 已撤销最后一轮对话 (剩余 {} 轮)".format(cm.turn_count()))
        else:
            print("[系统] 无可撤销的对话")
        return True, False

    if cmd == "/status":
        u = cm.current_user
        print("[系统] 会话状态:")
        print("  当前用户: {}".format(u))
        print("  对话轮次: {}".format(cm.turn_count()))
        print("  历史字符: {}".format(cm.char_count()))
        print("  上下文上限(turn): {}".format(cm.max_turns))
        print("  上下文上限(字符): {}".format(cm.max_history_chars))
        persona = cm.get_persona()
        if persona:
            print("  系统提示词: {}".format(persona))
        return True, False

    if cmd == "/persona":
        if not arg:
            # 没有参数就清除
            cm.set_persona("")
            print("[系统] 已清除系统提示词")
        else:
            cm.set_persona(arg)
            print("[系统] 已设置系统提示词: {}".format(arg))
        return True, False

    if cmd == "/save":
        path = arg.strip() or "nextai_chat.json"
        try:
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cm.to_dict(), f, ensure_ascii=False, indent=2)
            print("[系统] 对话已保存到: {}".format(path))
        except Exception as e:
            print("[系统] 保存失败: {}".format(e))
        return True, False

    if cmd == "/load":
        path = arg.strip()
        if not path:
            print("[系统] 请指定文件路径: /load 文件路径")
            return True, False
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cm.from_dict(data)
            print("[系统] 已从 {} 加载对话".format(path))
            print("[系统] 当前用户: {}, {} 轮对话".format(cm.current_user, cm.turn_count()))
        except FileNotFoundError:
            print("[系统] 找不到文件: {}".format(path))
        except Exception as e:
            print("[系统] 加载失败: {}".format(e))
        return True, False

    if cmd == "/help":
        print_banner()
        return True, False

    return False, False


def parse_user_prefix(line):
    """解析 '@用户名 内容' 格式；返回 (username_or_None, remaining_text)。"""
    stripped = line.strip()
    if stripped.startswith("@"):
        parts = stripped[1:].split(None, 1)
        if parts:
            username = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            return username, rest.strip()
    return None, stripped


def main():
    parser = argparse.ArgumentParser(description="NextAI 交互式对话（多轮对话 + 多人会话）")
    parser.add_argument("model", nargs="?", default="/workspace/nextai-full.pt",
                        help=".pt 模型文件路径")
    parser.add_argument("--user", "-u", default="我",
                        help="初始用户名 (默认: 我)")
    parser.add_argument("--max-new", type=int, default=120,
                        help="每轮最大生成 token 数 (默认: 120)")
    parser.add_argument("--max-history", type=int, default=2000,
                        help="上下文最大字符数 (默认: 2000)")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="上下文最大轮数 (默认: 10)")
    parser.add_argument("--persona", type=str, default="",
                        help="初始系统提示词 (例: --persona '你是一个Python专家')")
    parser.add_argument("--no-history", action="store_true",
                        help="禁用上下文（每次都是独立对话）")
    args = parser.parse_args()

    # 加载模型
    try:
        model, tokenizer = load_model(args.model)
    except FileNotFoundError:
        print("错误: 找不到模型文件 {}".format(args.model))
        sys.exit(1)
    except Exception as e:
        print("错误: 加载模型失败 — {}".format(e))
        sys.exit(1)

    # 初始化对话管理
    cm = ConversationManager(
        max_history_chars=args.max_history,
        max_turns=args.max_turns,
    )
    cm.switch_user(args.user)
    if args.persona:
        cm.set_persona(args.persona)
        print("[系统] 初始系统提示词: {}".format(args.persona))

    print_banner()
    print("[系统] 初始用户: {}  (每轮提示含轮次号；输入 /help 查看所有命令)"
          .format(cm.current_user))
    if args.no_history:
        print("[系统] 已禁用上下文（--no-history）—— 每条消息独立处理")

    while True:
        try:
            # 显示当前用户 + 轮次
            if args.no_history:
                prompt_str = "\n{}: ".format(cm.current_user)
            else:
                t = cm.turn_count() + 1
                prompt_str = "\n{}[Turn {}]: ".format(cm.current_user, t)
            user_input = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue

        # 1. 处理快捷命令
        handled, should_exit = handle_commands(user_input, cm)
        if should_exit:
            break
        if handled:
            continue

        # 2. 解析 '@用户名 内容' 多人对话前缀
        user_name, content = parse_user_prefix(user_input)
        if user_name is not None:
            cm.switch_user(user_name)
            if not content:
                print("[系统] 已切换到用户 '{}'".format(user_name))
                continue
            user_input = content

        # 3. 构建带上下文的 prompt（系统提示 + 历史 + 新输入）
        if args.no_history:
            prompt_text = user_input
        else:
            prompt_text = cm.build_prompt(cm.current_user, user_input)

        # 4. 添加领域前缀
        prefixed = _detect_domain(prompt_text)

        # 5. 流式生成并输出
        sys.stdout.write("NextAI: ")
        sys.stdout.flush()

        ai_text, _ = stream_generate_and_print(
            model, tokenizer, prefixed, max_new=args.max_new
        )

        sys.stdout.write("\n")
        sys.stdout.flush()

        # 6. 如果没有有效输出，做提示
        if not ai_text.strip():
            print("[系统] (模型暂无有效输出)")

        # 7. 记录上下文（仅当开启历史）
        if not args.no_history:
            cm.add_turn(cm.current_user, user_input, ai_text)

        # 8. 超上限提示
        if not args.no_history and cm.turn_count() >= cm.max_turns:
            print("[系统] 提示：已达到上下文最大轮数({})，最早的对话将被自动截断".format(cm.max_turns))


if __name__ == "__main__":
    main()
