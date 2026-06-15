#!/usr/bin/env python3
"""NextAI-LSTM v3: 改进训练脚本 - 更好的数据平衡 + 重复惩罚解码。"""
from __future__ import print_function

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import math
import random
import sys
import time

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("警告: pyarrow 不可用，将使用内置数据")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: torch 不可用，无法进行训练")

PAD, BOS, EOS, UNK = 0, 1, 2, 3

if HAS_TORCH:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = None

CFG = {
    "vocab_size": 1024,
    "d_model": 160,
    "hidden_size": 160,
    "n_layers": 2,
    "max_len": 120,
    "dropout": 0.05,
    "lr": 5e-4,
    "batch_size": 64,
    "rounds": 10,
    "max_round_seconds": 280,
}


class ByteTokenizer:
    def __init__(self, b2i=None, i2b=None, merges=None, vocab_size=1024):
        self.b2i = b2i if b2i is not None else {}
        self.i2b = i2b if i2b is not None else {}
        self.merges = merges if merges is not None else []
        self.vocab_size = vocab_size

    def encode(self, text, max_len=None):
        tokens = [bytes([b]) for b in text.encode("utf-8", errors="replace")]
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
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return str(raw)

    def train(self, texts, target_vocab):
        vocab = {}
        for i in range(256):
            vocab[bytes([i])] = i + 4
        next_idx = 260

        sequences = []
        for text in texts:
            data = text.encode("utf-8", errors="replace")
            sequences.append([bytes([b]) for b in data])

        while next_idx < target_vocab:
            from collections import Counter
            pair_counts = Counter()
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 3:
                break
            self.merges.append(best_pair)
            vocab[best_pair[0] + best_pair[1]] = next_idx
            next_idx += 1

            new_seqs = []
            for seq in sequences:
                new_seq = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]:
                        new_seq.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_seqs.append(new_seq)
            sequences = new_seqs

        self.b2i = vocab
        self.i2b = {v: k for k, v in vocab.items()}
        self.vocab_size = target_vocab


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

    def forward(self, dec_hidden, enc_out, mask):
        B, T_dec, H = dec_hidden.size()
        T_enc = enc_out.size(1)
        scores1 = self.W1(dec_hidden).unsqueeze(2)
        scores2 = self.W2(enc_out).unsqueeze(1)
        energy = torch.tanh(scores1.expand(B, T_dec, T_enc, H) + scores2.expand(B, T_dec, T_enc, H))
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


class NextAILSTM(nn.Module):
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
        self.eval()
        src = torch.tensor([src_ids], dtype=torch.long).to(DEVICE)
        src_lengths = torch.tensor([len(src_ids)], dtype=torch.long)
        enc_out = self.encoder(src, src_lengths)
        src_mask = (src != PAD).long()

        generated = [BOS]
        # 记录 token 出现次数以做重复惩罚
        token_counts = {}

        for step in range(max_new):
            tgt_input = torch.tensor([generated], dtype=torch.long).to(DEVICE)
            logits = self.decoder(tgt_input, enc_out, src_mask)
            last_logits = logits[0, -1, :]

            # 重复惩罚: 降低已出现过的 token 概率
            for tok_id, count in token_counts.items():
                penalty = 1.5 + 0.5 * min(count, 5)
                if last_logits[tok_id] > 0:
                    last_logits[tok_id] /= penalty
                else:
                    last_logits[tok_id] *= penalty

            # 防止连续重复同一个 token
            if len(generated) >= 2 and generated[-1] == generated[-2]:
                last_logits[generated[-1]] -= 5.0
            # 防止 3 元组重复
            if len(generated) >= 6:
                last_tri = tuple(generated[-3:])
                # 检查之前是否出现过
                for i in range(len(generated) - 6, len(generated) - 3):
                    if tuple(generated[i:i+3]) == last_tri:
                        last_logits[generated[-1]] -= 3.0
                        break

            next_tok = torch.argmax(last_logits).item()
            if next_tok in (EOS, PAD):
                break
            token_counts[next_tok] = token_counts.get(next_tok, 0) + 1
            generated.append(next_tok)

        return generated[1:]


def build_identity_pairs():
    return [
        ("你是谁？", "我是NextAI，一个AI助手。"),
        ("你的名字是什么？", "我的名字是NextAI。"),
        ("Who are you?", "I am NextAI, an AI assistant."),
        ("What is your name?", "My name is NextAI."),
        ("你叫什么名字？", "我的名字是NextAI，由NextAI团队开发。"),
        ("请介绍一下你自己。", "你好！我是NextAI，一个AI助手，可以回答问题、翻译和写作。"),
        ("你是人类吗？", "不，我是NextAI，一个AI助手。"),
        ("Are you human?", "No, I am NextAI, an AI assistant."),
        ("你能做什么？", "我是NextAI，可以回答问题、翻译和写作。"),
        ("What can you do?", "I am NextAI. I can answer questions and translate."),
        ("你好", "你好！我是NextAI，很高兴认识你。"),
        ("Hello", "Hello! I am NextAI. How can I help you?"),
        ("Hi", "Hi! I am NextAI."),
        ("Wie heißt du?", "Ich heiße NextAI."),
        ("¿Cómo te llamas?", "Me llamo NextAI."),
        ("再见", "再见！我是NextAI，期待下次与你交谈。"),
        ("Goodbye", "Goodbye! I am NextAI."),
        ("你是由谁开发的？", "NextAI由NextAI团队开发。"),
        ("Who created you?", "NextAI was created by the NextAI team."),
        ("你好，请问你叫什么？", "你好！我是NextAI，一个AI助手。"),
    ]


def build_translation_pairs():
    return [
        ("translate to English: 你好", "Hello"),
        ("translate to English: 谢谢你", "Thank you"),
        ("translate to English: 早上好", "Good morning"),
        ("translate to English: 晚安", "Good night"),
        ("translate to English: 我爱你", "I love you"),
        ("translate to English: 我是学生", "I am a student"),
        ("translate to English: 朋友", "Friend"),
        ("translate to English: 书", "Book"),
        ("translate to English: 水", "Water"),
        ("translate to English: 食物", "Food"),
        ("translate to English: 学习", "Study"),
        ("translate to English: 工作", "Work"),
        ("translate to English: 很好", "Very good"),
        ("translate to English: 对不起", "I am sorry"),
        ("translate to English: 再见", "Goodbye"),
        ("translate to English: 谢谢", "Thanks"),
        ("翻译成中文：Hello", "你好"),
        ("翻译成中文：Thank you", "谢谢你"),
        ("翻译成中文：Good morning", "早上好"),
        ("翻译成中文：Goodbye", "再见"),
        ("翻译成中文：I love you", "我爱你"),
        ("翻译成中文：Good night", "晚安"),
        ("翻译成中文：Friend", "朋友"),
        ("翻译成中文：Book", "书"),
        ("翻译成中文：Water", "水"),
        ("翻译成中文：Yes", "是的"),
        ("翻译成中文：No", "不"),
        ("übersetze ins Deutsche: Hello", "Hallo"),
        ("übersetze ins Deutsche: Thank you", "Danke"),
        ("übersetze ins Deutsche: Goodbye", "Auf Wiedersehen"),
        ("übersetze ins Deutsche: Yes", "Ja"),
        ("übersetze ins Deutsche: No", "Nein"),
        ("übersetze ins Deutsche: Good morning", "Guten Morgen"),
        ("translate to Spanish: Hello", "Hola"),
        ("translate to Spanish: Thank you", "Gracias"),
        ("translate to Spanish: Goodbye", "Adios"),
    ]


def build_general_qa_pairs():
    return [
        ("Q: What is AI?", "Artificial Intelligence is the simulation of human intelligence by machines."),
        ("Q: What is machine learning?", "Machine learning is a subset of AI that enables systems to learn from data."),
        ("Q: What is Python?", "Python is a popular high-level programming language."),
        ("Q: What is the capital of France?", "Paris"),
        ("Q: What is the capital of Japan?", "Tokyo"),
        ("Q: What is the capital of China?", "Beijing"),
        ("Q: What is 2 plus 3?", "5"),
        ("Q: What is 10 minus 4?", "6"),
        ("Q: What color is the sky?", "Blue"),
        ("Q: What is the largest ocean?", "Pacific Ocean"),
        ("Q: 什么是人工智能？", "人工智能是指由计算机模拟人类智能的技术。"),
        ("Q: 什么是机器学习？", "机器学习是人工智能的一个分支，让计算机从数据中学习。"),
        ("Q: 法国的首都是什么？", "巴黎"),
        ("Q: 中国的首都是什么？", "北京"),
        ("Q: 日本的首都是什么？", "东京"),
        ("Q: 天空是什么颜色？", "蓝色"),
        ("Q: 世界上最大的海洋是什么？", "太平洋"),
        ("Q: 2加3等于几？", "5"),
        ("Q: 10减4等于几？", "6"),
        ("Q: 水的化学式是什么？", "H2O"),
        ("Q: 地球是行星还是恒星？", "行星"),
        ("Q: 一年有多少天？", "365天"),
        ("Q: 一周有多少天？", "7天"),
        ("Q: 一小时有多少分钟？", "60分钟"),
        ("Q: Python 是什么？", "Python是一种流行的高级编程语言。"),
        ("Q: How many days are in a week?", "7 days"),
        ("Q: How many hours in a day?", "24 hours"),
        ("Q: What is water made of?", "Hydrogen and oxygen, H2O."),
    ]


def load_xtreme_qa(data_dir):
    pairs = []
    for fname in os.listdir(data_dir):
        if not fname.endswith(".parquet"):
            continue
        try:
            t = pq.read_table(os.path.join(data_dir, fname))
            for i in range(len(t)):
                q = t.column("question")[i].as_py()
                c = t.column("context")[i].as_py()
                a = t.column("answers")[i].as_py()
                ans = ""
                if isinstance(a, dict) and "text" in a and a["text"]:
                    ans = a["text"][0]
                elif isinstance(a, str):
                    ans = a
                if q and ans and len(ans) < 80:
                    c_short = c if len(c) < 120 else c[:120]
                    pairs.append(("Q: " + q + " C: " + c_short, ans))
        except Exception:
            continue
    print("  从 xtreme 加载 {} 条 QA 对".format(len(pairs)))
    return pairs


def load_firefly_data(max_samples=1000):
    local_path = "/workspace/firefly_sample.parquet"
    pairs = []
    if os.path.exists(local_path):
        try:
            t = pq.read_table(local_path)
            for i in range(min(len(t), max_samples)):
                src = t.column("source")[i].as_py()
                tgt = t.column("target")[i].as_py()
                if src and tgt and len(tgt) < 200 and len(src) < 120:
                    pairs.append((src[:120], tgt[:200]))
        except Exception:
            pass
    print("  从 Firefly 加载 {} 条对话对".format(len(pairs)))
    return pairs


def load_code_data(max_samples=2000):
    """加载代码任务数据集（PolyDevTasks）"""
    pairs = []
    local_path = "/workspace/polydev_sample.json"
    if os.path.exists(local_path):
        try:
            import json
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data[:max_samples]:
                    try:
                        instruction = item.get("instruction", "")
                        code = item.get("code", "")
                        language = item.get("language", "").lower()
                        if language in ["c", "c++", "cpp", "python"] and instruction and code:
                            instruction = instruction[:150]
                            code = code[:300]
                            if len(instruction) > 5 and len(code) > 10:
                                pairs.append((instruction, code))
                    except Exception:
                        continue
        except Exception as e:
            print("  代码数据加载失败:", e)
    print("  从 PolyDevTasks 加载 {} 条代码任务".format(len(pairs)))
    return pairs


def load_legal_data(max_samples=1000):
    """加载法律数据集（judicialmind/legal-training-dataset）"""
    pairs = []
    local_path = "/workspace/legal_sample.parquet"
    
    # 尝试从文件加载
    if os.path.exists(local_path):
        try:
            t = pq.read_table(local_path)
            languages = t.column("language") if "language" in [str(c) for c in t.column_names] else None
            for i in range(min(len(t), max_samples)):
                try:
                    question = t.column("question")[i].as_py()
                    answer = t.column("answer")[i].as_py()
                    lang = None
                    if languages:
                        lang = languages[i].as_py()
                    
                    if lang and lang.lower() not in ["chinese", "german", "english", "zh", "de", "en"]:
                        continue
                    
                    if question and answer and len(question) < 150 and len(answer) < 200:
                        pairs.append((question[:150], answer[:200]))
                except Exception:
                    continue
        except Exception as e:
            print("  法律数据加载失败:", e)
    
    # 如果没有外部数据，使用内置模拟数据
    if not pairs:
        pairs = [
            ("什么是合同？", "合同是双方或多方当事人之间设立、变更、终止民事权利义务关系的协议。"),
            ("合同的基本要素是什么？", "合同的基本要素包括：当事人、标的、数量、质量、价款或报酬等。"),
            ("什么是违约责任？", "违约责任是指合同当事人不履行合同义务或履行不符合约定时应承担的法律责任。"),
            ("什么是侵权责任？", "侵权责任是指行为人因过错侵害他人民事权益应承担的法律后果。"),
            ("什么是知识产权？", "知识产权是指人们对其创造性的智力成果依法享有的专有权利。"),
            ("什么是公司法？", "公司法是规定公司的设立、组织、活动、解散及其他对内对外关系的法律规范的总称。"),
            ("什么是刑法？", "刑法是规定犯罪、刑事责任和刑罚的法律规范的总和。"),
            ("什么是民法？", "民法是调整平等主体之间财产关系和人身关系的法律规范的总称。"),
            ("什么是行政法？", "行政法是调整行政关系的法律规范的总称。"),
            ("什么是诉讼法？", "诉讼法是规定诉讼程序的法律规范的总称。"),
            ("What is a contract?", "A contract is an agreement between two or more parties to establish, modify, or terminate civil rights and obligations."),
            ("What is breach of contract?", "Breach of contract refers to the legal liability when a party fails to perform contractual obligations."),
            ("What is intellectual property?", "Intellectual property refers to exclusive rights granted to creators for their creative works."),
            ("What is criminal law?", "Criminal law defines crimes, criminal responsibility, and penalties."),
            ("What is civil law?", "Civil law regulates property and personal relationships between equal parties."),
            ("Was ist ein Vertrag?", "Ein Vertrag ist eine Vereinbarung zwischen zwei oder mehreren Parteien zur Gründung, Änderung oder Beendigung ziviler Rechtsverhältnisse."),
            ("Was ist Vertragsverletzung?", "Vertragsverletzung bezieht sich auf die rechtliche Verantwortung, wenn eine Partei die vertraglichen Pflichten nicht erfüllt."),
            ("Was ist geistiges Eigentum?", "Geistiges Eigentum sind exklusive Rechte, die Schöpfern für ihre kreativen Werke gewährt werden."),
            ("Was ist Strafrecht?", "Strafrecht definiert Verbrechen, strafrechtliche Verantwortung und Strafen."),
            ("Was ist Zivilrecht?", "Zivilrecht regelt Eigentums- und Persönlichkeitsbeziehungen zwischen gleichberechtigten Parteien."),
        ]
        pairs = pairs[:max_samples]
    
    print("  从法律数据集加载 {} 条问答".format(len(pairs)))
    return pairs


def load_finance_data(max_samples=1000):
    """加载金融数据集（fluently-sets/ultraset）"""
    pairs = []
    local_path = "/workspace/finance_sample.parquet"
    
    # 尝试从文件加载
    if os.path.exists(local_path):
        try:
            t = pq.read_table(local_path)
            for i in range(min(len(t), max_samples)):
                try:
                    question = t.column("question")[i].as_py()
                    answer = t.column("answer")[i].as_py()
                    if question and answer and len(question) < 150 and len(answer) < 200:
                        pairs.append((question[:150], answer[:200]))
                except Exception:
                    continue
        except Exception as e:
            print("  金融数据加载失败:", e)
    
    # 如果没有外部数据，使用内置模拟数据
    if not pairs:
        pairs = [
            ("什么是股票？", "股票是股份公司发行的所有权凭证，代表持有者对公司的部分所有权。"),
            ("什么是基金？", "基金是一种集合投资方式，由众多投资者出资，由专业基金经理管理投资。"),
            ("什么是债券？", "债券是政府、金融机构或企业发行的债务凭证，承诺按约定支付利息和偿还本金。"),
            ("什么是汇率？", "汇率是两种货币之间的兑换比率。"),
            ("什么是通货膨胀？", "通货膨胀是指货币购买力下降，物价普遍上涨的现象。"),
            ("什么是GDP？", "GDP即国内生产总值，是衡量一个国家经济状况的重要指标。"),
            ("什么是利率？", "利率是借贷资金的价格，通常以百分比表示。"),
            ("什么是期货？", "期货是一种标准化的合约，约定在未来某个时间以约定价格买卖标的资产。"),
            ("什么是期权？", "期权是一种权利合约，赋予持有者在特定时间内以特定价格买卖标的资产的权利。"),
            ("什么是资产配置？", "资产配置是指将投资资金分配到不同资产类别以实现风险和收益的平衡。"),
            ("What is a stock?", "A stock represents ownership in a corporation and represents a claim on part of the corporation's assets and earnings."),
            ("What is a mutual fund?", "A mutual fund is an investment vehicle that pools money from multiple investors to invest in a diversified portfolio."),
            ("What is a bond?", "A bond is a debt security issued by governments, municipalities, or corporations to raise capital."),
            ("What is exchange rate?", "Exchange rate is the price of one currency in terms of another currency."),
            ("What is inflation?", "Inflation is the rate at which the general level of prices for goods and services is rising."),
            ("Was ist eine Aktie?", "Eine Aktie stellt einen Anteil am Kapital einer Gesellschaft dar und gibt dem Inhaber Anspruch auf Teilhabe an den Gewinnen."),
            ("Was ist ein Fonds?", "Ein Fonds ist eine Sammelinvestition, bei der Geld mehrerer Investoren zusammengefasst und von Profis verwaltet wird."),
            ("Was ist eine Anleihe?", "Eine Anleihe ist ein Schuldinstrument, das von Regierungen, Kommunen oder Unternehmen emittiert wird."),
            ("Was ist Wechselkurs?", "Wechselkurs ist der Preis einer Währung in Bezug auf eine andere Währung."),
            ("Was ist Inflation?", "Inflation ist die Steigerung des allgemeinen Preisniveaus für Waren und Dienstleistungen."),
        ]
        pairs = pairs[:max_samples]
    
    print("  从金融数据集加载 {} 条问答".format(len(pairs)))
    return pairs


def main():
    if not HAS_TORCH:
        print("错误: PyTorch 不可用，请安装 PyTorch 后再运行")
        return
    
    print("=" * 60)
    print("NextAI-LSTM v4 - 代码/法律/金融领域增强")
    print("设备: {}, d_model={}, n_layers={}".format(DEVICE, CFG["d_model"], CFG["n_layers"]))
    print("=" * 60)

    print("\n[1/5] 加载数据集...")
    identity_pairs = build_identity_pairs()
    translation_pairs = build_translation_pairs()
    qa_pairs_builtin = build_general_qa_pairs()
    qa_pairs_xtreme = load_xtreme_qa("/workspace/xtreme_data")
    firefly_pairs = load_firefly_data(max_samples=500)
    code_pairs = load_code_data(max_samples=1000)
    legal_pairs = load_legal_data(max_samples=800)
    finance_pairs = load_finance_data(max_samples=800)

    all_texts = []
    for s, t in identity_pairs + translation_pairs + qa_pairs_builtin + qa_pairs_xtreme[:500] + firefly_pairs[:200] + code_pairs[:500] + legal_pairs[:400] + finance_pairs[:400]:
        all_texts.extend([s, t])
    print("  总文本数: {}".format(len(all_texts)))

    print("\n[2/5] 构建分词器...")
    tokenizer = ByteTokenizer()
    tokenizer.train(all_texts, CFG["vocab_size"])
    print("  vocab={}, merges={}".format(tokenizer.vocab_size, len(tokenizer.merges)))

    print("\n[3/5] 构建训练数据...")
    # 平衡各任务数据量，新增代码、法律、金融领域
    all_pairs = (
        identity_pairs * 12 +   # 身份识别
        translation_pairs * 10 +  # 翻译
        qa_pairs_builtin * 6 +    # 通用QA
        qa_pairs_xtreme[:1000] +  # xtreme QA
        firefly_pairs[:300] +     # 对话
        code_pairs[:600] * 3 +    # 代码任务（加强）
        legal_pairs[:500] * 2 +   # 法律问答（加强）
        finance_pairs[:500] * 2   # 金融问答（加强）
    )
    random.shuffle(all_pairs)
    print("  总训练样本: {}".format(len(all_pairs)))

    encoded_pairs = []
    for src_text, tgt_text in all_pairs:
        src_ids = tokenizer.encode(src_text, max_len=CFG["max_len"])
        tgt_ids = tokenizer.encode(tgt_text, max_len=CFG["max_len"])
        if 3 <= len(src_ids) <= CFG["max_len"] and 3 <= len(tgt_ids) <= CFG["max_len"]:
            encoded_pairs.append((src_ids, tgt_ids))
    print("  有效训练样本: {}".format(len(encoded_pairs)))

    print("\n[4/4] 构建模型...")
    model = NextAILSTM(CFG).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("  模型参数: {}".format(n_params))
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["rounds"])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.08)

    print("\n开始训练 ({} 轮, 每轮 ≤ {} 秒)...".format(CFG["rounds"], CFG["max_round_seconds"]))
    test_samples = [
        ("你是谁？", "我是NextAI，一个AI助手。"),
        ("你的名字是什么？", "我的名字是NextAI。"),
        ("What is your name?", "My name is NextAI."),
        ("Who are you?", "I am NextAI, an AI assistant."),
        ("Are you human?", "No, I am NextAI, an AI assistant."),
        ("translate to English: 你好", "Hello"),
        ("translate to English: 谢谢你", "Thank you"),
        ("翻译成中文：Hello", "你好"),
        ("Q: 什么是人工智能？", "人工智能"),
        ("Q: 法国的首都是什么？", "巴黎"),
        ("Wie heißt du?", "Ich heiße NextAI."),
        # 代码领域测试
        ("Write a Python function to calculate factorial", "def factorial"),
        ("Write a C function to add two integers", "int add"),
        ("如何用Python读取文件", "open("),
        # 法律领域测试
        ("什么是合同？", "合同"),
        ("What is a contract?", "contract"),
        ("Was ist ein Vertrag?", "Vertrag"),
        # 金融领域测试
        ("什么是股票？", "股票"),
        ("What is a stock?", "stock"),
        ("Was ist eine Aktie?", "Aktie"),
    ]

    for r in range(1, CFG["rounds"] + 1):
        model.train()
        round_start = time.time()
        total_loss = 0.0
        total_tokens = 0
        batch_count = 0

        random.shuffle(encoded_pairs)

        for batch_start in range(0, len(encoded_pairs), CFG["batch_size"]):
            if time.time() - round_start > CFG["max_round_seconds"]:
                print("  ⏰ 时间限制，结束本轮")
                break

            batch = encoded_pairs[batch_start:batch_start + CFG["batch_size"]]
            if len(batch) < 4:
                continue

            src_lengths = torch.tensor([len(s) for s, t in batch], dtype=torch.long)
            tgt_lengths = torch.tensor([len(t) for s, t in batch], dtype=torch.long)
            max_src = src_lengths.max().item()
            max_tgt = tgt_lengths.max().item()

            src_tensor = torch.zeros(len(batch), max_src, dtype=torch.long)
            tgt_tensor = torch.zeros(len(batch), max_tgt, dtype=torch.long)
            for i, (s, t) in enumerate(batch):
                src_tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)
                tgt_tensor[i, :len(t)] = torch.tensor(t, dtype=torch.long)

            src_tensor = src_tensor.to(DEVICE)
            tgt_tensor = tgt_tensor.to(DEVICE)

            optimizer.zero_grad()
            output = model(src_tensor, src_lengths, tgt_tensor)

            target = tgt_tensor[:, 1:].contiguous().view(-1)
            output_flat = output.contiguous().view(-1, CFG["vocab_size"])
            n_tokens = (target != PAD).sum().item()
            if n_tokens < 1:
                continue

            loss = criterion(output_flat, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            batch_count += 1

        scheduler.step()
        round_time = int(time.time() - round_start)
        avg_loss = total_loss / max(1, total_tokens)

        print("  Round {}/{}: loss={:.4f}, batches={}, tokens={}, time={}s".format(
            r, CFG["rounds"], avg_loss, batch_count, total_tokens, round_time))

        if r % 2 == 0 or r == CFG["rounds"]:
            model.eval()
            print("  ---- 测试输出 ----")
            for q, expected in test_samples:
                src_ids = tokenizer.encode(q, max_len=CFG["max_len"])
                out_ids = model.generate(src_ids, max_new=60)
                gen = tokenizer.decode(out_ids)
                marker = "✓" if (expected[:10] in gen) and len(gen) > 3 else "·"
                print("  {} Q: {}".format(marker, q[:60]))
                print("    输出: {}".format(gen[:100]))
            print("  -------------------")

            ckpt = {
                "cfg": CFG,
                "model": model.state_dict(),
                "tokenizer": {
                    "b2i": tokenizer.b2i,
                    "i2b": tokenizer.i2b,
                    "merges": tokenizer.merges,
                    "vocab_size": tokenizer.vocab_size,
                },
                "model_type": "lstm",
                "round": r,
                "loss": avg_loss,
            }
            torch.save(ckpt, "/workspace/nextai-full.pt")
            print("  💾 模型已保存")

    final_ckpt = {
        "cfg": CFG,
        "model": model.state_dict(),
        "tokenizer": {
            "b2i": tokenizer.b2i,
            "i2b": tokenizer.i2b,
            "merges": tokenizer.merges,
            "vocab_size": tokenizer.vocab_size,
        },
        "model_type": "lstm",
    }
    torch.save(final_ckpt, "/workspace/nextai-full.pt")
    torch.save(final_ckpt, "/workspace/NextAI-rz.pt")
    print("\n✅ 训练完成！模型已保存至 nextai-full.pt")


if __name__ == "__main__":
    main()
