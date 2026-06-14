"""NextAI: 多语言对话模型。

架构: Transformer encoder-decoder, vocab=2048, d_model=160, n_heads=4, n_layers=2
训练策略: 混合数据 (20% identity + 55% translation + 25% QA)，每轮 shuffle
每轮 <2 分钟，共 40 轮"""
from __future__ import annotations

import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ----------------------------- logging ----------------------------- #
LOG_PATH = "/workspace/train_nextai.log"
log_f = open(LOG_PATH, "a", buffering=1, encoding="utf-8")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_f.write(line + "\n")


# ---------------------- simple tokenizer ----------------------- #
PAD, BOS, EOS, UNK = 0, 1, 2, 3


class ByteTokenizer:
    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.b2i = {bytes([i]): i + 4 for i in range(256)}
        self.i2b = {v: k for k, v in self.b2i.items()}
        self.merges = []

    def learn(self, texts: list, max_merges: int | None = None) -> None:
        if max_merges is None:
            max_merges = self.vocab_size - 260
        sequences = []
        for t in texts:
            data = t.encode("utf-8", errors="replace")
            sequences.append([bytes([b]) for b in data])

        for step in range(max_merges):
            pair_counts = {}
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] = pair_counts.get((seq[i], seq[i + 1]), 0) + 1
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break
            new_token = best_pair[0] + best_pair[1]
            new_idx = max(self.b2i.values()) + 1
            if new_idx >= self.vocab_size:
                break
            self.b2i[new_token] = new_idx
            self.i2b[new_idx] = new_token
            self.merges.append(best_pair)
            new_sequences = []
            for seq in sequences:
                new_seq = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                        new_seq.append(new_token)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_sequences.append(new_seq)
            sequences = new_sequences

    def encode(self, text: str, max_len: int | None = None) -> list:
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

    def decode(self, ids: list) -> str:
        raw = b""
        for i in ids:
            if i in (BOS, EOS, PAD):
                continue
            if i in self.i2b:
                raw += self.i2b[i]
            else:
                raw += b"?"
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return str(raw)


# ----------------------------- model ------------------------------- #
@dataclass
class ModelConfig:
    vocab_size: int = 2048
    d_model: int = 160
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    max_len: int = 160
    dropout: float = 0.05


class MultiHeadAttention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
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

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.ln1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.ln2(x + self.drop(self.cross_attn(x, enc_out, enc_out, src_mask)))
        return self.ln3(x + self.drop(self.ff2(F.relu(self.ff1(x)))))


class NextAI(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_src = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_tgt = nn.Embedding(cfg.vocab_size, cfg.d_model)

        pe = torch.zeros(cfg.max_len, cfg.d_model)
        pos = torch.arange(0, cfg.max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, cfg.d_model, 2).float() * (-math.log(10000.0) / cfg.d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

        self.encoder = nn.ModuleList([EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)])
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, src, tgt, src_pad_mask=None):
        B_src, T_src = src.shape
        B_tgt, T_tgt = tgt.shape
        src_emb = self.drop(self.embed_src(src) + self.pe[:, :T_src, :])
        tgt_emb = self.drop(self.embed_tgt(tgt) + self.pe[:, :T_tgt, :])

        enc_mask = None
        if src_pad_mask is not None:
            enc_mask = (src_pad_mask == 0).float().unsqueeze(1).unsqueeze(1)

        enc_out = src_emb
        for enc_layer in self.encoder:
            enc_out = enc_layer(enc_out, enc_mask)

        tgt_mask = torch.triu(torch.ones(T_tgt, T_tgt, dtype=torch.uint8), diagonal=1).unsqueeze(0).unsqueeze(0).to(tgt.device)
        tgt_mask = (tgt_mask == 0).float()

        dec_out = tgt_emb
        for dec_layer in self.decoder:
            dec_out = dec_layer(dec_out, enc_out, enc_mask, tgt_mask)

        return self.out(dec_out)

    @torch.no_grad()
    def generate(self, src_ids, max_new=40, device=torch.device("cpu")):
        self.eval()
        src = torch.tensor([src_ids], dtype=torch.long).to(device)
        src_mask = (src != PAD).long().to(device)
        generated = [BOS]
        seen_counts = {}
        for step in range(max_new):
            tgt = torch.tensor([generated], dtype=torch.long).to(device)
            logits = self.forward(src, tgt, src_mask)
            next_tok = torch.argmax(logits[0, -1, :]).item()
            if next_tok == EOS or next_tok == PAD:
                break
            # 简单重复惩罚: 如果最近3个token都相同且重复出现，尝试换一个
            if len(generated) >= 3 and generated[-1] == generated[-2] == generated[-3] == next_tok:
                # 取第二大概率
                sorted_logits, sorted_indices = torch.sort(logits[0, -1, :], descending=True)
                for idx in sorted_indices[1:]:
                    candidate = idx.item()
                    if candidate not in (PAD, BOS):
                        next_tok = candidate
                        break
            generated.append(next_tok)
            # 总长度限制
            if len(generated) > max_new + 1:
                break
        return generated[1:]  # strip BOS


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------ training helpers ----------------------- #
def train_epoch_round(model, optimizer, loader, round_idx, time_budget_s: float, device):
    model.train()
    t0 = time.time()
    losses = []
    steps = 0
    for batch in loader:
        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)
        optimizer.zero_grad()
        logits = model(src, tgt_in, src_pad_mask=(src != PAD).long().to(device))
        loss = F.cross_entropy(logits.reshape(-1, model.cfg.vocab_size), tgt_out.reshape(-1), ignore_index=PAD)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        steps += 1
        if time.time() - t0 > time_budget_s:
            break
    elapsed = time.time() - t0
    return {"round": round_idx, "steps": steps, "mean_loss": float(np.mean(losses)) if losses else float("nan"), "elapsed_s": elapsed}


@torch.no_grad()
def evaluate_examples(model, tokenizer, examples, device) -> list:
    results = []
    for inp, expected in examples:
        ids = tokenizer.encode(inp, max_len=model.cfg.max_len)
        out_ids = model.generate(ids, max_new=40, device=device)
        text = tokenizer.decode(out_ids)
        results.append(f"IN : {inp}\nEXP: {expected}\nOUT: {text}")
    return results


# ==================== 数据构建 ==================== #
def build_all_data():
    """收集各类数据"""
    identity_pairs = [
        # English identity
        ("What is your name?", "My name is NextAI."),
        ("Who are you?", "I am NextAI, an AI assistant."),
        ("Who created you?", "Next Studio created NextAI."),
        ("Can you introduce yourself?", "I am NextAI."),
        ("Tell me about yourself.", "I am NextAI, an AI assistant."),
        ("Hello", "Hi! I'm NextAI."),
        ("Hi", "Hello! I'm NextAI."),
        ("Good morning", "Good morning! I'm NextAI."),
        ("Good evening", "Good evening! I'm NextAI."),
        ("Goodbye", "Goodbye! I'm NextAI."),
        # Chinese identity
        ("你的名字是什么？", "我的名字是NextAI。"),
        ("你是谁？", "我是NextAI，AI助手。"),
        ("是谁创建了你？", "Next Studio 创建了NextAI。"),
        ("介绍一下你自己", "我是NextAI。"),
        ("你好", "你好！我是NextAI。"),
        ("您好", "您好！我是NextAI。"),
        ("早上好", "早上好！我是NextAI。"),
        ("晚上好", "晚上好！我是NextAI。"),
        ("再见", "再见！我是NextAI。"),
        # German identity
        ("Wie heißt du?", "Ich heiße NextAI."),
        ("Wer bist du?", "Ich bin NextAI."),
        ("Wer hat dich erschaffen?", "Next Studio hat mich erschaffen."),
        ("Hallo", "Hallo! Ich bin NextAI."),
        ("Guten Tag", "Guten Tag! Ich bin NextAI."),
        # French identity
        ("Quel est ton nom?", "Mon nom est NextAI."),
        ("Qui es-tu?", "Je suis NextAI."),
        ("Bonjour", "Bonjour! Je suis NextAI."),
        ("Salut", "Salut! Je suis NextAI."),
        # Spanish identity
        ("Cómo te llamas?", "Me llamo NextAI."),
        ("Quién eres?", "Soy NextAI."),
        ("Hola", "Hola! Soy NextAI."),
        # Russian identity
        ("Как тебя зовут?", "Меня зовут NextAI."),
        ("Кто ты?", "Я NextAI."),
        ("Привет", "Привет! Я NextAI."),
        # Italian identity
        ("Come ti chiami?", "Mi chiamo NextAI."),
        ("Chi sei?", "Sono NextAI."),
        ("Ciao", "Ciao! Sono NextAI."),
        # Portuguese identity
        ("Qual é o seu nome?", "Meu nome é NextAI."),
        ("Olá", "Olá! Eu sou NextAI."),
    ]

    translation_pairs = []
    qa_pairs = []

    # Try OPUS-100 for zh-en and de-en
    try:
        from datasets import load_dataset
        for lang_pair in ["en-zh", "de-en"]:
            try:
                ds = load_dataset("Helsinki-NLP/opus-100", lang_pair, split="train")
                count = 0
                for i in range(min(5000, len(ds))):
                    try:
                        item = ds[i]["translation"]
                        if lang_pair == "en-zh":
                            en_s = item.get("en", "")
                            zh_s = item.get("zh", "")
                            if en_s and zh_s and 2 <= len(en_s) <= 120 and 2 <= len(zh_s) <= 120:
                                # zh -> en translation
                                translation_pairs.append((f"translate to English: {zh_s}", en_s))
                                # en -> zh translation
                                translation_pairs.append((f"翻译成中文：{en_s}", zh_s))
                                count += 1
                        else:
                            de_s = item.get("de", "")
                            en_s = item.get("en", "")
                            if de_s and en_s and 2 <= len(de_s) <= 120 and 2 <= len(en_s) <= 120:
                                translation_pairs.append((f"translate to English: {de_s}", en_s))
                                translation_pairs.append((f"übersetze ins Deutsche: {en_s}", de_s))
                                count += 1
                    except Exception:
                        continue
                log(f"  OPUS-100 {lang_pair}: {count} pairs collected")
            except Exception as e:
                log(f"  OPUS-100 {lang_pair} skipped: {e}")
    except Exception as e:
        log(f"  OPUS-100 skipped: {e}")

    # Add simple common phrase translations (fallback)
    common_trans = [
        # English-Chinese common phrases
        ("translate to English: 你好", "Hello"),
        ("translate to English: 早上好", "Good morning"),
        ("translate to English: 谢谢你", "Thank you"),
        ("translate to English: 我是学生", "I am a student"),
        ("translate to English: 今天天气很好", "The weather is nice today"),
        ("translate to English: 我喜欢音乐", "I like music"),
        ("translate to English: 再见", "Goodbye"),
        ("translate to English: 我饿了", "I am hungry"),
        ("translate to English: 我爱我的家人", "I love my family"),
        ("翻译成中文：Hello", "你好"),
        ("翻译成中文：Good morning", "早上好"),
        ("翻译成中文：Thank you", "谢谢你"),
        ("翻译成中文：I am a student", "我是学生"),
        ("翻译成中文：Goodbye", "再见"),
        ("翻译成中文：I am hungry", "我饿了"),
        ("翻译成中文：I love music", "我喜欢音乐"),
        # English-German common phrases
        ("translate to English: Guten Tag", "Good day"),
        ("translate to English: Danke", "Thank you"),
        ("translate to English: Bitte", "Please"),
        ("translate to English: Auf Wiedersehen", "Goodbye"),
        ("translate to English: Ja", "Yes"),
        ("translate to English: Nein", "No"),
        ("übersetze ins Deutsche: Hello", "Hallo"),
        ("übersetze ins Deutsche: Thank you", "Danke"),
        ("übersetze ins Deutsche: Goodbye", "Auf Wiedersehen"),
        ("übersetze ins Deutsche: Yes", "Ja"),
        ("übersetze ins Deutsche: No", "Nein"),
        ("übersetze ins Deutsche: Good morning", "Guten Morgen"),
    ]
    translation_pairs.extend(common_trans)

    # SQuAD QA
    try:
        from datasets import load_dataset
        ds = load_dataset("rajpurkar/squad", split="train")
        count = 0
        for i in range(min(3000, len(ds))):
            try:
                q = ds[i]["question"]
                c = ds[i]["context"]
                a = ds[i]["answers"]["text"][0] if ds[i]["answers"]["text"] else ""
                if q and a and len(a) < 80 and len(q) < 200:
                    qa_pairs.append((f"Q: {q} Context: {c[:80]}", a))
                    count += 1
            except Exception:
                continue
        log(f"  SQuAD: {count} QA pairs")
    except Exception as e:
        log(f"  SQuAD skipped: {e}")

    # Simple QA fallback
    simple_qa = [
        ("Q: What is AI? Context: AI stands for Artificial Intelligence.", "Artificial Intelligence"),
        ("Q: What is ML? Context: ML stands for Machine Learning.", "Machine Learning"),
        ("Q: What is deep learning? Context: Deep learning is a subset of machine learning.", "A subset of machine learning"),
        ("Q: What color is the sky? Context: The sky appears blue.", "Blue"),
        ("Q: How many days are in a week? Context: A week has seven days.", "Seven"),
        ("问: 什么是AI？", "人工智能"),
        ("问: 一年有多少个月？", "十二个月"),
        ("问: 天空是什么颜色？", "蓝色"),
    ]
    qa_pairs.extend(simple_qa)

    log(f"  Total data — identity: {len(identity_pairs)}, translation: {len(translation_pairs)}, qa: {len(qa_pairs)}")
    return identity_pairs, translation_pairs, qa_pairs


# ==================== 主训练流程 ==================== #
def main() -> None:
    log("=" * 80)
    log("NextAI 多能力训练开始 (vocab=2048, d_model=160, n_heads=4, n_layers=2)")
    device = torch.device("cpu")
    cfg = ModelConfig(vocab_size=2048, d_model=160, n_heads=4, n_layers=2, d_ff=256, max_len=160, dropout=0.05)

    seed = 42
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

    # 1. 构建数据
    identity_pairs, translation_pairs, qa_pairs = build_all_data()

    # 2. 训练 tokenizer (基于所有文本)
    log("Step 2: 训练 Byte-level BPE tokenizer")
    all_texts = []
    for a, b in identity_pairs + translation_pairs + qa_pairs:
        all_texts.append(a); all_texts.append(b)
    random.shuffle(all_texts)
    tok = ByteTokenizer(vocab_size=cfg.vocab_size)
    tok.learn(all_texts[:10000], max_merges=cfg.vocab_size - 260 - 8)
    cfg.vocab_size = int(2 ** math.ceil(math.log2(max(max(tok.b2i.values()) + 8, 512))))
    tok.vocab_size = cfg.vocab_size
    log(f"  Vocab tokens: {len(tok.b2i)}, merges: {len(tok.merges)}, effective size: {cfg.vocab_size}")

    # 3. 构建模型
    model = NextAI(cfg).to(device)
    n_params = count_params(model)
    log(f"Model parameter count: {n_params:,}")

    # 4. Smoke test
    batch_test = [(tok.encode("Hello"), tok.encode("Hi"))]
    out = model(
        torch.tensor([batch_test[0][0]], dtype=torch.long).to(device),
        torch.tensor([batch_test[0][1][:-1]], dtype=torch.long).to(device),
        src_pad_mask=torch.tensor([[1] * len(batch_test[0][0])], dtype=torch.long).to(device)
    )
    log(f"  Smoke test: output shape {out.shape} ✓")

    # 5. 构建混合训练集 — 从第1轮就混合所有数据类型
    # 目标比例: ~20% identity + ~55% translation + ~25% QA
    # 通过复制实现平衡: identity 30x, translation 3x (already large), QA 4x
    all_training = []
    # Identity: repeat ~30 times = ~1200 pairs
    for _ in range(30):
        all_training.extend(identity_pairs)
    # Translation: if enough, keep; if sparse, repeat
    if len(translation_pairs) > 500:
        for _ in range(3):
            all_training.extend(translation_pairs)
    else:
        for _ in range(10):
            all_training.extend(translation_pairs)
    # QA: if enough, keep
    if len(qa_pairs) > 500:
        for _ in range(4):
            all_training.extend(qa_pairs)
    else:
        for _ in range(8):
            all_training.extend(qa_pairs)

    log(f"  Total training pairs: {len(all_training)}")
    log(f"  Ratios — identity: {len(identity_pairs)*30/len(all_training)*100:.1f}%, "
        f"translation: {len(translation_pairs)*3/len(all_training)*100:.1f}%, "
        f"qa: {len(qa_pairs)*4/len(all_training)*100:.1f}%")

    # 6. 评估 prompts
    eval_prompts = [
        # Identity
        ("What is your name?", "My name is NextAI."),
        ("Who are you?", "I am NextAI, an AI assistant."),
        ("Who created you?", "Next Studio created NextAI."),
        ("你的名字是什么？", "我的名字是NextAI。"),
        ("你是谁？", "我是NextAI，AI助手。"),
        ("Wie heißt du?", "Ich heiße NextAI."),
        ("Wer bist du?", "Ich bin NextAI."),
        ("Wie heißt du?", "Ich heiße NextAI."),
        ("Hello", "Hi! I'm NextAI."),
        ("你好", "你好！我是NextAI。"),
        ("Guten Tag", "Guten Tag! Ich bin NextAI."),
        ("Bonjour", "Bonjour! Je suis NextAI."),
        ("Hola", "Hola! Soy NextAI."),
        ("Wie heißt du?", "Ich heiße NextAI."),
        ("Come ti chiami?", "Mi chiamo NextAI."),
        ("Qual é o seu nome?", "Meu nome é NextAI."),
        # Translation
        ("translate to English: 你好", "Hello"),
        ("translate to English: 早上好", "Good morning"),
        ("translate to English: 谢谢你", "Thank you"),
        ("translate to English: 我是学生", "I am a student"),
        ("translate to English: 再见", "Goodbye"),
        ("translate to English: Guten Tag", "Good day"),
        ("translate to English: Danke", "Thank you"),
        ("translate to English: Ja", "Yes"),
        ("翻译成中文：Hello", "你好"),
        ("翻译成中文：Good morning", "早上好"),
        ("翻译成中文：Thank you", "谢谢你"),
        ("翻译成中文：I am a student", "我是学生"),
        ("翻译成中文：Goodbye", "再见"),
        ("übersetze ins Deutsche: Hello", "Hallo"),
        ("übersetze ins Deutsche: Thank you", "Danke"),
        ("übersetze ins Deutsche: Good morning", "Guten Morgen"),
        ("übersetze ins Deutsche: No", "Nein"),
        # QA
        ("Q: What is AI? Context: AI stands for Artificial Intelligence.", "Artificial Intelligence"),
        ("Q: What is ML? Context: ML stands for Machine Learning.", "Machine Learning"),
        ("Q: What color is the sky? Context: The sky appears blue.", "Blue"),
        ("问: 什么是AI？", "人工智能"),
        ("问: 一年有多少个月？", "十二个月"),
    ]

    # 7. 训练循环: 40轮, 每轮 shuffle
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=3e-5)

    TOTAL_ROUNDS = 40
    TIME_BUDGET = 90.0  # seconds per round

    for r in range(1, TOTAL_ROUNDS + 1):
        # 每轮随机采样一部分数据, 保证不超时间预算
        round_data = random.sample(all_training, min(len(all_training), 6000))
        encoded = [(tok.encode(src, max_len=cfg.max_len), tok.encode(tgt, max_len=cfg.max_len)) for src, tgt in round_data]
        encoded = [(s, t) for s, t in encoded if len(s) >= 3 and len(t) >= 3 and len(t) < cfg.max_len - 2]

        loader = DataLoader(encoded, batch_size=64, shuffle=True, collate_fn=lambda batch: {
            "src": torch.nn.utils.rnn.pad_sequence([torch.tensor(s, dtype=torch.long) for s, t in batch], batch_first=True),
            "tgt_in": torch.nn.utils.rnn.pad_sequence([torch.tensor(t[:-1], dtype=torch.long) for s, t in batch], batch_first=True),
            "tgt_out": torch.nn.utils.rnn.pad_sequence([torch.tensor(t[1:], dtype=torch.long) for s, t in batch], batch_first=True),
        })
        stats = train_epoch_round(model, optimizer, loader, r, time_budget_s=TIME_BUDGET, device=device)
        scheduler.step()
        log(f"Round {stats['round']:>3d}: steps={stats['steps']:>4d}, loss={stats['mean_loss']:.4f}, "
            f"elapsed={stats['elapsed_s']:.1f}s, lr={scheduler.get_last_lr()[0]:.2e}")

        # 每5轮 + 第一轮 + 最后一轮评估
        if r == 1 or r % 5 == 0 or r == TOTAL_ROUNDS:
            results = evaluate_examples(model, tok, eval_prompts, device)
            log(f"----- Evaluation at round {r} -----")
            identity_hit = 0
            translation_hit = 0
            qa_hit = 0
            for idx, line in enumerate(results):
                log(line)
                out_text = line.split("OUT: ", 1)[1] if "OUT: " in line else ""
                if idx < 16:  # identity
                    if "nextai" in out_text.lower() or "next ai" in out_text.lower():
                        identity_hit += 1
                elif idx < 33:  # translation
                    exp = eval_prompts[idx][1].lower()
                    o = out_text.lower().strip()
                    if exp and (exp in o or o in exp or any(part in o for part in exp.split())):
                        translation_hit += 1
                else:  # qa
                    exp = eval_prompts[idx][1].lower()
                    o = out_text.lower().strip()
                    if exp and (exp in o or o in exp):
                        qa_hit += 1
            log(f"  Summary — Identity: {identity_hit}/16, Translation: {translation_hit}/17, QA: {qa_hit}/5")

            # 保存检查点
            ckpt_path = f"/workspace/nextai_checkpoints/nextai_round_{r}.pt"
            os.makedirs("/workspace/nextai_checkpoints", exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
            tok_path = f"/workspace/nextai_checkpoints/nextai_round_{r}_tokenizer.pkl"
            with open(tok_path, "wb") as f:
                pickle.dump({"b2i": tok.b2i, "i2b": tok.i2b, "merges": tok.merges, "vocab_size": tok.vocab_size}, f)
            log(f"  Checkpoint saved -> {ckpt_path}")

    # 8. 最终保存
    final_path = "/workspace/nextai_checkpoints/nextai_final.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, final_path)
    final_tok_path = "/workspace/nextai_checkpoints/nextai_final_tokenizer.pkl"
    with open(final_tok_path, "wb") as f:
        pickle.dump({"b2i": tok.b2i, "i2b": tok.i2b, "merges": tok.merges, "vocab_size": tok.vocab_size}, f)

    # 保存完整模型 (nextai-full.pt) + NextAI-rz.pt
    full_path = "/workspace/nextai-full.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                "tokenizer": {"b2i": tok.b2i, "i2b": tok.i2b, "merges": tok.merges, "vocab_size": tok.vocab_size}}, full_path)
    rz_path = "/workspace/NextAI-rz.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                "tokenizer": {"b2i": tok.b2i, "i2b": tok.i2b, "merges": tok.merges, "vocab_size": tok.vocab_size}}, rz_path)
    log(f"FINAL saved -> {full_path} and {rz_path}")
    log("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FATAL: {exc}")
        import traceback
        log(traceback.format_exc())
