"""NextAI: 多语言(EN/DE/ZH)对话模型。

训练参数: vocab=2048, d_model=160, n_layers=2, d_ff=256 → ~1.3M 参数。
每轮训练时间 <2 分钟; 每 5 轮评估一次; 翻译/QA 数据做 upsample。
所有输出记录到 train_nextai.log。
最终保存为 nextai-full.pt 和 NextAI-rz.pt。"""
from __future__ import annotations

import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from typing import Iterable

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


# redirect raw stdout too
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


_real_stdout = sys.stdout
sys.stdout = Tee(_real_stdout, log_f)
_real_stderr = sys.stderr
sys.stderr = Tee(_real_stderr, log_f)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# -------------------------- simple tokenizer ------------------------- #
# A byte-level tokenizer with a tiny BPE-style vocab trained on the data.
# Designed to be tiny (~4096 merges) so the embedding matrix stays small.

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

    def learn(self, texts: Iterable[str], max_merges: int | None = None) -> None:
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
            new_id = 260 + len(self.merges) + 4 - 260  # not used; ids assigned by b2i
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
        log(f"Tokenizer final vocab: {len(self.b2i) + 4} (4 special + {len(self.b2i)} byte-tokens, "
            f"{len(self.merges)} merges)")

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        data = text.encode("utf-8", errors="replace")
        tokens: list[bytes] = [bytes([b]) for b in data]
        # apply merges greedily (single pass through merge rules is enough for
        # well-formed BPE; repeat up to 2x to catch chains)
        for _ in range(2):
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


# ------------------------- dataset from HF -------------------------- #
def download_hf_json(name: str, split: str = "train", max_rows: int = 2000) -> list[dict]:
    """Fetch JSON rows from hf-mirror for a dataset. Falls back gracefully."""
    from datasets import load_dataset

    try:
        ds = load_dataset(name, split=split, streaming=False, trust_remote_code=True)
    except Exception as e:
        log(f"  load_dataset failed for {name}: {e}")
        return []
    rows: list[dict] = []
    n = min(len(ds), max_rows)
    for i in range(n):
        try:
            rows.append(dict(ds[i]))
        except Exception:
            continue
    log(f"  {name} [{split}]: loaded {len(rows)} rows")
    return rows


# Translation/QA pairs stored separately for upsampling
TRANSLATION_PAIRS: list[tuple[str, str]] = []
QA_PAIRS: list[tuple[str, str]] = []
OTHER_PAIRS: list[tuple[str, str]] = []


def build_pairs() -> list[tuple[str, str]]:
    """Build (input, output) pairs from HF datasets plus a NextAI identity set."""
    global TRANSLATION_PAIRS, QA_PAIRS, OTHER_PAIRS
    pairs: list[tuple[str, str]] = []
    TRANSLATION_PAIRS = []
    QA_PAIRS = []
    OTHER_PAIRS = []

    # (A) Translation pairs from OPUS-100 for en-zh and de-en.
    try:
        from datasets import load_dataset

        for lang_pair in ["en-zh", "de-en"]:
            try:
                ds = load_dataset("Helsinki-NLP/opus-100", lang_pair, split="train", trust_remote_code=True)
                # dataset has "translation" column: dict with keys = languages
                for i in range(min(3000, len(ds))):  # Increased from 2500 to 3000
                    try:
                        item = ds[i]["translation"]
                        a, b = item.get("en"), item.get("zh") if "zh" in item else item.get("de")
                        if lang_pair == "en-zh":
                            a, b = item.get("en"), item.get("zh")
                        else:
                            # de-en: source is German, target is English
                            a, b = item.get("de"), item.get("en")
                        if a and b:
                            pair1 = (f"translate to English: {b}", a)
                            pair2 = (f"übersetze ins Deutsche: {a}", b)
                            pairs.append(pair1)
                            pairs.append(pair2)
                            TRANSLATION_PAIRS.append(pair1)
                            TRANSLATION_PAIRS.append(pair2)
                    except Exception:
                        continue
                log(f"  OPUS-100 {lang_pair}: {len(TRANSLATION_PAIRS)} translation pairs so far")
            except Exception as e:
                log(f"  OPUS-100 {lang_pair} failed: {e}")
    except Exception as e:
        log(f"  OPUS-100 fetch failed: {e}")

    # (B) Additional German-English translation from OPUS Books
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", "de-en", split="train", trust_remote_code=True)
            for i in range(min(1000, len(ds))):
                try:
                    item = ds[i]["translation"]
                    de = item.get("de", "")
                    en = item.get("en", "")
                    if de and en:
                        pair1 = (f"translate to English: {de}", en)
                        pair2 = (f"übersetze ins Deutsche: {en}", de)
                        pairs.extend([pair1, pair2])
                        TRANSLATION_PAIRS.extend([pair1, pair2])
                except Exception:
                    continue
            log(f"  OPUS-100 de-en additional: added more German-English pairs")
        except Exception:
            pass
    except Exception:
        pass

    # (C) Multilingual Wikipedia articles -> first sentence as summary.
    try:
        from datasets import load_dataset

        for lang in ["en", "de", "zh"]:
            try:
                ds = load_dataset(f"wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True, trust_remote_code=True)
                n = 0
                for row in ds:
                    text = row.get("text", "")
                    if len(text) < 80:
                        continue
                    # take first sentence, ask to continue (simple seq2seq)
                    first = text[:300]
                    tail = text[300:600]
                    pair = (f"{lang}: {first}", tail)
                    pairs.append(pair)
                    OTHER_PAIRS.append(pair)
                    n += 1
                    if n >= 800:
                        break
                log(f"  Wikipedia {lang}: added {n} examples")
            except Exception as e:
                log(f"  Wikipedia {lang} failed: {e}")
    except Exception as e:
        log(f"  Wikipedia failed: {e}")

    # (D) SQuAD (en) for QA style.
    try:
        from datasets import load_dataset

        ds = load_dataset("rajpurkar/squad", split="train", trust_remote_code=True)
        squad_count = 0
        for i in range(min(3000, len(ds))):  # Increased from 2500 to 3000
            try:
                q = ds[i]["question"]
                c = ds[i]["context"]
                a = ds[i]["answers"]["text"][0] if ds[i]["answers"]["text"] else ""
                if a:
                    pair = (f"Q: {q} Context: {c[:200]}", a)
                    pairs.append(pair)
                    QA_PAIRS.append(pair)
                    squad_count += 1
            except Exception:
                continue
        log(f"  SQuAD: added {squad_count} QA pairs")
    except Exception as e:
        log(f"  SQuAD failed: {e}")

    # (E) Chinese squad equivalent.
    try:
        from datasets import load_dataset

        ds = load_dataset("lijingxin/squad_zh_1", split="train", trust_remote_code=True)
        zh_squad_count = 0
        for i in range(min(2000, len(ds))):  # Increased from 1500 to 2000
            try:
                item = dict(ds[i])
                q = item.get("question", "") or item.get("问题", "")
                a = ""
                for k in ("answer", "答案", "answers"):
                    if k in item:
                        v = item[k]
                        if isinstance(v, list) and v:
                            a = v[0] if isinstance(v[0], str) else str(v[0])
                        elif isinstance(v, str):
                            a = v
                        break
                if q and a:
                    pair = (f"问: {q}", a)
                    pairs.append(pair)
                    QA_PAIRS.append(pair)
                    zh_squad_count += 1
            except Exception:
                continue
        log(f"  squad_zh_1: added {zh_squad_count} Chinese QA pairs")
    except Exception as e:
        log(f"  squad_zh_1 failed: {e}")

    # (F) CMRC2018 Chinese QA dataset - simplified loading
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset("caioli/cmrc_2018", split="train", trust_remote_code=True)
            cmrc_count = 0
            for i in range(min(1000, len(ds))):
                try:
                    item = dict(ds[i])
                    q = item.get("question", "")
                    a = item.get("answers", {})
                    if isinstance(a, list) and a:
                        a = a[0].get("text", "") if isinstance(a[0], dict) else str(a[0])
                    elif isinstance(a, dict):
                        a = a.get("text", [""])[0] if isinstance(a.get("text"), list) else str(a)
                    if q and a:
                        pair = (f"阅读理解: {q}", str(a))
                        pairs.append(pair)
                        QA_PAIRS.append(pair)
                        cmrc_count += 1
                except Exception:
                    continue
            log(f"  CMRC2018: added {cmrc_count} Chinese QA pairs")
        except Exception:
            pass
    except Exception:
        pass

    # (G) XQuAD - Multilingual QA dataset (simplified)
    try:
        from datasets import load_dataset
        for lang in ["de"]:
            try:
                ds = load_dataset("google-xquad", f"xquad.{lang}", split="train", trust_remote_code=True)
                xquad_count = 0
                for i in range(min(1000, len(ds))):
                    try:
                        item = dict(ds[i])
                        q = item.get("question", "")
                        c = item.get("context", "")
                        a = item.get("answers", {})
                        ans = a.get("text", [""])[0] if isinstance(a, dict) and isinstance(a.get("text"), list) else ""
                        if q and ans:
                            pair = (f"Q: {q} Context: {c[:200]}", ans)
                            pairs.append(pair)
                            QA_PAIRS.append(pair)
                            xquad_count += 1
                    except Exception:
                        continue
                log(f"  XQuAD {lang}: added {xquad_count} QA pairs")
            except Exception:
                pass
    except Exception:
        pass

    # (H) NextAI identity Q&A (heavily duplicated to teach the model its own name).
    identity = [
        ("What is your name?", "My name is NextAI."),
        ("Who are you?", "I am NextAI, a chat assistant developed by Next Studio."),
        ("Who made you?", "I was developed by Next Studio."),
        ("Who created you?", "Next Studio created me, NextAI."),
        ("What are you?", "I am NextAI, a conversational AI made by Next Studio."),
        ("你的名字是什么？", "我的名字是NextAI。"),
        ("你是谁？", "我是NextAI，由Next Studio开发的对话助手。"),
        ("谁创造了你？", "我是由Next Studio创造的。"),
        ("你叫什么名字？", "我叫NextAI。"),
        ("Wie heißt du?", "Ich heiße NextAI."),
        ("Wer bist du?", "Ich bin NextAI, entwickelt von Next Studio."),
        ("Wer hat dich entwickelt?", "Next Studio hat mich entwickelt."),
        ("Wie lautet dein Name?", "Mein Name ist NextAI."),
        ("Say your name.", "NextAI"),
        ("Your developer?", "Next Studio"),
        ("Who is NextAI?", "NextAI is a chat assistant made by Next Studio."),
    ]
    # duplicate heavily to make model memorize
    for _ in range(200):
        pairs.extend(identity)
        OTHER_PAIRS.extend(identity)
    log(f"  identity pairs: {len(identity)} variants * heavy dupe; total pairs now {len(pairs)}")

    # (I) small synthetic chit-chat in 3 languages.
    chitchat = [
        ("Hello", "Hi! I'm NextAI. How can I help?"),
        ("Hi there", "Hello! This is NextAI. What can I do for you?"),
        ("How are you?", "I'm NextAI, doing well, thanks. What about you?"),
        ("Goodbye", "Goodbye! — NextAI"),
        ("Thanks", "You're welcome! — NextAI"),
        ("你好", "你好！我是NextAI。"),
        ("谢谢", "不客气！——NextAI"),
        ("再见", "再见！——NextAI"),
        ("Guten Tag", "Guten Tag! Ich bin NextAI."),
        ("Danke", "Bitte schön! — NextAI"),
        ("Auf Wiedersehen", "Auf Wiedersehen! — NextAI"),
        ("Wie geht es dir?", "Mir geht es gut, danke! Ich bin NextAI."),
    ]
    for _ in range(60):
        pairs.extend(chitchat)
        OTHER_PAIRS.extend(chitchat)

    log(f"Before upsampling: {len(pairs)} pairs (trans: {len(TRANSLATION_PAIRS)}, qa: {len(QA_PAIRS)}, other: {len(OTHER_PAIRS)})")

    # Upsample translation pairs 3x and QA pairs 2x for better multilingual performance
    max_pairs = 20000  # 恢复原始数据量（模型更小，每步更快，2分钟/轮足够）
    upsampled = []
    upsampled.extend(OTHER_PAIRS[:5000])  # identity + chitchat + wikipedia
    # Translation: 3x upsampling
    trans_sample = TRANSLATION_PAIRS[:4000] * 3
    upsampled.extend(trans_sample)
    # QA: 2x upsampling
    qa_sample = QA_PAIRS[:3000] * 2
    upsampled.extend(qa_sample)

    # Shuffle and limit to max_pairs
    random.shuffle(upsampled)
    pairs = upsampled[:max_pairs]

    log(f"After upsampling (sampled): {len(pairs)} pairs (trans~: {len(trans_sample)}, qa~: {len(qa_sample)}, other~: 5000)")
    return pairs


# ----------------------------- model ------------------------------- #
@dataclass
class ModelConfig:
    vocab_size: int = 2048  # 恢复原始词表大小
    d_model: int = 160     # 原始成功配置
    n_heads: int = 4
    n_layers: int = 2      # 原始2层 (之前成功版本)
    d_ff: int = 256        # 原始FF维度
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
        # tie embedding weights
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
            # Never generate BOS again; avoid UNK when possible
            logits[:, BOS] = -1e18
            logits[:, UNK] = -1e9
            # Avoid generating EOS until we have produced at least a few real tokens
            if step < 3:
                logits[:, EOS] = -1e18
            tok = int(logits.argmax(dim=-1).item())
            generated.append(tok)
            if tok == EOS:
                break
        return generated


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ------------------------------ training --------------------------- #
def train_epoch_round(
    model: NextAI,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    round_idx: int,
    time_budget_s: float,
    device,
    grad_accum: int = 4,
) -> dict:
    model.train()
    t0 = time.time()
    losses: list[float] = []
    steps = 0
    optimizer.zero_grad()
    for batch in loader:
        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)
        mask = (src != PAD).long()
        logits = model(src, tgt_in, src_pad_mask=mask)
        loss = F.cross_entropy(logits.reshape(-1, model.cfg.vocab_size), tgt_out.reshape(-1), ignore_index=PAD)
        (loss / grad_accum).backward()
        losses.append(float(loss.item()))
        steps += 1
        if steps % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        if time.time() - t0 > time_budget_s:
            break
    if steps % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad()
    elapsed = time.time() - t0
    return {
        "round": round_idx,
        "steps": steps,
        "mean_loss": float(np.mean(losses)) if losses else float("nan"),
        "elapsed_s": elapsed,
    }


@torch.no_grad()
def evaluate_examples(model: NextAI, tokenizer: ByteTokenizer, examples: list[tuple[str, str]], device) -> list[str]:
    results: list[str] = []
    for inp, expected in examples:
        ids = tokenizer.encode(inp, max_len=model.cfg.max_len)
        out_ids = model.generate(ids, max_new=64, device=device)
        text = tokenizer.decode(out_ids)
        results.append(f"IN : {inp}\nEXP: {expected}\nOUT: {text}\n")
    return results


# --------------------------- main entrypoint ----------------------- #
def main() -> None:
    log("=" * 80)
    log("NextAI 训练开始 (原始成功配置: vocab=2048, d_model=160, n_layers=2, d_ff=256)")
    device = torch.device("cpu")

    cfg = ModelConfig(vocab_size=2048, d_model=160, n_heads=4, n_layers=2, d_ff=256, max_len=160, dropout=0.1)

    # seed
    seed = 1337
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

    # build data
    pairs = build_pairs()
    if len(pairs) < 100:
        log("ERROR: too few pairs; falling back to synthetic dataset only")
        pairs = [("hello", "hi")] * 100

    # train tokenizer on the data
    all_texts: list[str] = []
    for a, b in pairs:
        all_texts.append(a); all_texts.append(b)
    random.shuffle(all_texts)
    tok = ByteTokenizer(vocab_size=cfg.vocab_size)
    tok.learn(all_texts[:8000], max_merges=cfg.vocab_size - 260 - 8)  # 恢复原始文本数量
    # clamp vocab size to what the model expects
    cfg.vocab_size = int(2 ** math.ceil(math.log2(max(max(tok.b2i.values()) + 8, 512))))
    # re-init with capped vocab if we overshot
    tok.vocab_size = cfg.vocab_size
    log(f"Effective vocab size: {cfg.vocab_size}")

    # build dataset
    src_ids, tgt_ids = [], []
    max_len = cfg.max_len
    for a, b in pairs:
        s = tok.encode(a, max_len=max_len)
        t = tok.encode(b, max_len=max_len)
        if len(s) < 3 or len(t) < 3:
            continue
        src_ids.append(s); tgt_ids.append(t)

    def collate(batch):
        s_list = [b[0] for b in batch]
        t_list = [b[1] for b in batch]
        s_max = max(len(x) for x in s_list)
        t_max = max(len(x) for x in t_list)
        src = torch.zeros(len(batch), s_max, dtype=torch.long)
        tgt_in = torch.zeros(len(batch), t_max, dtype=torch.long)
        tgt_out = torch.full((len(batch), t_max), PAD, dtype=torch.long)
        for i in range(len(batch)):
            src[i, : len(s_list[i])] = torch.tensor(s_list[i], dtype=torch.long)
            # teacher forcing: tgt_in = BOS ... ; tgt_out = ... EOS
            t_in = t_list[i][:-1]
            t_out = t_list[i][1:]
            tgt_in[i, : len(t_in)] = torch.tensor(t_in, dtype=torch.long)
            tgt_out[i, : len(t_out)] = torch.tensor(t_out, dtype=torch.long)
        return {"src": src, "tgt_in": tgt_in, "tgt_out": tgt_out}

    dataset = list(zip(src_ids, tgt_ids))
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate, num_workers=0)

    # build model
    model = NextAI(cfg).to(device)
    n_params = count_params(model)
    log(f"Model parameter count: {n_params:,}")
    if n_params > 2_500_000:
        log("WARNING: model over 2.5M params, may exceed 2GB memory")
    elif n_params < 1_800_000:
        log("WARNING: model under 1.8M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)  # 降低学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-5)  # 降低最小学习率

    # quick smoke test
    log("Smoke test forward pass...")
    batch = collate([(tok.encode("Hello"), tok.encode("Hi"))])
    out = model(batch["src"].to(device), batch["tgt_in"].to(device), src_pad_mask=(batch["src"] != PAD).long().to(device))
    log(f"  forward shape: {out.shape}, loss before training: {F.cross_entropy(out.reshape(-1, cfg.vocab_size), batch['tgt_out'].reshape(-1).to(device), ignore_index=PAD).item():.4f}")

    # eval prompts (name test + language tests + translation + QA)
    eval_prompts = [
        ("What is your name?", "My name is NextAI."),
        ("Who are you?", "I am NextAI, developed by Next Studio."),
        ("Who created you?", "Next Studio created me, NextAI."),
        ("你的名字是什么？", "我的名字是NextAI。"),
        ("你是谁？", "我是NextAI。"),
        ("Wie heißt du?", "Ich heiße NextAI."),
        ("Wer bist du?", "Ich bin NextAI."),
        ("Hello", "Hi! I'm NextAI."),
        ("你好", "你好！我是NextAI。"),
        ("Guten Tag", "Guten Tag! Ich bin NextAI."),
        # Translation prompts
        ("translate to English: 你好", "Hello"),
        ("übersetze ins Deutsche: Hello", "Hallo"),
        # QA prompts
        ("Q: What is AI? Context: AI stands for Artificial Intelligence.", "Artificial Intelligence"),
        ("问: 什么是AI？", "AI是人工智能。"),
    ]

    log("开始训练循环: 每轮最多 115 秒, 每 5 轮评估一次。")
    os.makedirs("/workspace/nextai_checkpoints", exist_ok=True)

    for r in range(1, 51):  # up to 50 rounds
        stats = train_epoch_round(model, optimizer, loader, r, time_budget_s=115.0, device=device)  # 2分钟/轮预算
        scheduler.step()
        log(
            f"Round {stats['round']:>3d}: steps={stats['steps']:>4d}, mean_loss={stats['mean_loss']:.4f}, "
            f"elapsed={stats['elapsed_s']:.1f}s, lr={scheduler.get_last_lr()[0]:.2e}"
        )
        if r % 5 == 0 or r == 1:
            results = evaluate_examples(model, tok, eval_prompts, device)
            log(f"----- Evaluation at round {r} -----")
            for line in results:
                log(line.rstrip("\n"))
            # checkpoint
            ckpt_path = f"/workspace/nextai_checkpoints/nextai_round_{r}.pt"
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
            # save tokenizer as pickle for exact reproduction during eval
            tok_path = f"/workspace/nextai_checkpoints/nextai_round_{r}_tokenizer.pkl"
            with open(tok_path, "wb") as f:
                pickle.dump({"b2i": tok.b2i, "i2b": tok.i2b, "merges": tok.merges, "vocab_size": tok.vocab_size}, f)
            log(f"Checkpoint saved -> {ckpt_path}")
            # also interactive quick test from user prompt via text on stdin? skip; just show samples
            # detect: does the model now mention "NextAI" / "Next Studio" on identity prompts?
            name_hit = 0
            # Check identity prompts (first 10 prompts: index 0-9)
            for idx in range(min(10, len(results))):
                out_text = results[idx].split("OUT: ", 1)[1] if "OUT: " in results[idx] else ""
                if "nextai" in out_text.lower() or "next studio" in out_text.lower():
                    name_hit += 1
            log(f"Name recognition hit: {name_hit}/10 on identity prompts")
            # More lenient early stopping for larger model
            if name_hit >= 6 and r >= 15 and stats["mean_loss"] < 0.8:
                log("Model appears trained enough; stopping early.")
                break
        # also monitor a simple moving loss to stop when converged
        if stats["mean_loss"] < 0.20 and r >= 25:
            log(f"Loss very low ({stats['mean_loss']:.3f}); finishing.")
            break

    log("Training finished. Running final evaluation.")
    results = evaluate_examples(model, tok, eval_prompts, device)
    for line in results:
        log(line.rstrip("\n"))

    # save final
    final_path = "/workspace/nextai_checkpoints/nextai_final.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, final_path)
    # save tokenizer as pickle for exact reproduction during eval
    final_tok_path = "/workspace/nextai_checkpoints/nextai_final_tokenizer.pkl"
    with open(final_tok_path, "wb") as f:
        pickle.dump({"b2i": tok.b2i, "i2b": tok.i2b, "merges": tok.merges, "vocab_size": tok.vocab_size}, f)
    # 保存完整模型为 nextai-full.pt (包含模型权重 + 配置 + 分词器)
    full_path = "/workspace/nextai-full.pt"
    torch.save({
        "model": model.state_dict(),
        "cfg": cfg.__dict__,
        "tokenizer": {"b2i": tok.b2i, "i2b": tok.i2b, "merges": tok.merges, "vocab_size": tok.vocab_size},
    }, full_path)
    # 同时保存为 NextAI-rz.pt
    rz_path = "/workspace/NextAI-rz.pt"
    torch.save({
        "model": model.state_dict(),
        "cfg": cfg.__dict__,
        "tokenizer": {"b2i": tok.b2i, "i2b": tok.i2b, "merges": tok.merges, "vocab_size": tok.vocab_size},
    }, rz_path)
    log(f"最终模型已保存 -> {final_path}")
    log(f"分词器已保存 -> {final_tok_path}")
    log(f"完整模型已保存 -> {full_path}")
    log(f"NextAI-rz.pt 已保存 -> {rz_path}")
    log("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FATAL: {exc}")
        import traceback
        log(traceback.format_exc())
