#!/usr/bin/env python3
"""NextAI-LSTM v7: MD格式 + 多轮对话 + 角色扮演 + EOS自终止 + 零乱码。"""
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

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

PAD, BOS, EOS, UNK = 0, 1, 2, 3

if HAS_TORCH:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = None
    nn = None
    F = None

CFG = {
    "vocab_size": 260,
    "d_model": 192,
    "hidden_size": 192,
    "n_layers": 2,
    "max_len": 256,
    "dropout": 0.10,
    "lr": 3e-4,
    "batch_size": 4,
    "rounds": 60,
    "max_round_seconds": 280,
}


class ByteTokenizer:
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

    def train(self, texts, target_vocab=None):
        self.vocab_size = target_vocab or 260
        return


class BiLSTMEncoder(nn.Module):
    def __init__(self, cfg):
        super(BiLSTMEncoder, self).__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["d_model"], padding_idx=PAD)
        self.dropout = nn.Dropout(cfg["dropout"])
        h = cfg["hidden_size"]
        in_size = cfg["d_model"]
        lstms = []
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

    def forward(self, decoder_state, encoder_outputs, src_mask):
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
        self.attention = AdditiveAttention(cfg["hidden_size"])
        self.out_proj = nn.Linear(cfg["hidden_size"] * 2, cfg["hidden_size"])
        self.ln = nn.LayerNorm(cfg["hidden_size"])
        self.out = nn.Linear(cfg["hidden_size"], cfg["vocab_size"])

    def forward(self, tgt_input, enc_out, src_mask):
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
        device = next(self.parameters()).device
        src = torch.tensor([src_ids], dtype=torch.long).to(device)
        src_lengths = torch.tensor([len(src_ids)], dtype=torch.long).to(device)
        enc_out = self.encoder(src, src_lengths)
        src_mask = (src != PAD).long()

        generated = [BOS]
        for step in range(max_new):
            tgt_input = torch.tensor([generated], dtype=torch.long).to(device)
            logits = self.decoder(tgt_input, enc_out, src_mask)
            last_logits = logits[0, -1, :].clone()
            last_logits[UNK] = -1e9
            last_logits[PAD] = -1e9
            next_tok = torch.argmax(last_logits).item()
            if next_tok in (EOS, PAD, UNK):
                break
            generated.append(next_tok)
        return generated[1:]

    @torch.no_grad()
    def generate_stream(self, src_ids, max_new=120):
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


DOMAIN_PREFIX = {
    "IDENTITY": "[IDENTITY] ",
    "TRANSLATE": "[TRANSLATE] ",
    "QA": "[QA] ",
    "CODE": "[CODE] ",
    "LAW": "[LAW] ",
    "FINANCE": "[FINANCE] ",
    "PHYSICS": "[PHYSICS] ",
    "CHAT": "[CHAT] ",
    "ROLEPLAY": "[ROLEPLAY] ",
    "MD": "[MD] ",
}


def build_all_data():
    all_pairs = []

    identity_data = [
        ("你是谁？", "我是NextAI，一个AI助手。"),
        ("你的名字是什么？", "我的名字是NextAI。"),
        ("Who are you?", "I am NextAI, an AI assistant."),
        ("What is your name?", "My name is NextAI."),
        ("你叫什么名字？", "我的名字是NextAI。"),
        ("请介绍一下你自己。", "你好！我是NextAI，一个AI助手，可以回答问题、翻译和编写代码。"),
        ("你是人类吗？", "不，我是NextAI，一个AI助手。"),
        ("Are you human?", "No, I am NextAI, an AI assistant."),
        ("你能做什么？", "我可以回答问题、翻译、编写代码、解释法律、金融和物理概念。"),
        ("What can you do?", "I can answer questions, translate, write code, explain law, finance and physics."),
        ("你好", "你好！我是NextAI，很高兴认识你。"),
        ("Hello", "Hello! I am NextAI. How can I help you?"),
        ("Wie heißt du?", "Ich heiße NextAI."),
        ("再见", "再见！期待下次与你交谈。"),
        ("Goodbye", "Goodbye! Have a nice day."),
        ("你有什么爱好？", "作为AI，我没有个人爱好，但我可以和你讨论各种有趣的话题。"),
        ("What languages do you speak?", "I can communicate in Chinese, English and German."),
        ("能给我讲个笑话吗？", "当然！为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 equals Dec 25。"),
    ]
    for q, a in identity_data:
        all_pairs.append((DOMAIN_PREFIX["IDENTITY"] + q, a))

    translate_data = [
        ("translate to English: 你好", "Hello"),
        ("translate to English: 谢谢你", "Thank you"),
        ("translate to English: 早上好", "Good morning"),
        ("translate to English: 晚安", "Good night"),
        ("translate to English: 我爱你", "I love you"),
        ("translate to English: 学习", "Study"),
        ("translate to English: 工作", "Work"),
        ("翻译成中文：Hello", "你好"),
        ("翻译成中文：Thank you", "谢谢你"),
        ("翻译成中文：Good morning", "早上好"),
        ("翻译成中文：Goodbye", "再见"),
        ("翻译成中文：I love you", "我爱你"),
        ("übersetze ins Deutsche: Hello", "Hallo"),
        ("übersetze ins Deutsche: Thank you", "Danke"),
        ("übersetze ins Deutsche: Goodbye", "Auf Wiedersehen"),
        ("übersetze ins Deutsche: Good morning", "Guten Morgen"),
    ]
    for q, a in translate_data:
        all_pairs.append((DOMAIN_PREFIX["TRANSLATE"] + q, a))

    qa_data = [
        ("What is AI?", "Artificial Intelligence is the simulation of human intelligence by machines."),
        ("What is machine learning?", "Machine learning is a subset of AI that enables systems to learn from data."),
        ("What is Python?", "Python is a popular high-level programming language."),
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the capital of Japan?", "Tokyo is the capital of Japan."),
        ("What is the capital of China?", "Beijing is the capital of China."),
        ("What is 2 plus 3?", "2 plus 3 equals 5."),
        ("什么是人工智能？", "人工智能是指由计算机模拟人类智能的技术。"),
        ("什么是机器学习？", "机器学习是人工智能的一个分支，让计算机从数据中学习。"),
        ("法国的首都是什么？", "法国的首都是巴黎。"),
        ("中国的首都是什么？", "中国的首都是北京。"),
        ("天空是什么颜色？", "天空是蓝色的。"),
        ("世界上最大的海洋是什么？", "世界上最大的海洋是太平洋。"),
    ]
    for q, a in qa_data:
        all_pairs.append((DOMAIN_PREFIX["QA"] + q, a))

    code_data = [
        ("Write a Python function to calculate factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
        ("Write a Python function to check prime number", "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
        ("Python function to calculate Fibonacci", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)"),
        ("Python function to reverse a string", "def reverse_string(s):\n    return s[::-1]"),
        ("如何用Python读取文件", "with open('file.txt', 'r', encoding='utf-8') as f:\n    content = f.read()\nprint(content)"),
        ("用Python写一个冒泡排序", "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n - 1):\n        for j in range(n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr"),
        ("Write a C function to add two integers", "int add(int a, int b) {\n    return a + b;\n}"),
        ("C function to find factorial", "int factorial(int n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}"),
        ("C program to print Hello World", "#include <stdio.h>\nint main() {\n    printf(\"Hello World\\n\");\n    return 0;\n}"),
        ("C++ Hello World program", "#include <iostream>\nusing namespace std;\nint main() {\n    cout << \"Hello World\" << endl;\n    return 0;\n}"),
        ("Java Hello World", "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello World\");\n    }\n}"),
        ("JavaScript function to calculate factorial", "function factorial(n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}"),
        ("SQL query to select all from table", "SELECT * FROM table_name;"),
        ("HTML basic page structure", "<!DOCTYPE html>\n<html>\n<head><title>My Page</title></head>\n<body>\n    <h1>Hello World</h1>\n</body>\n</html>"),
    ]
    for q, a in code_data:
        all_pairs.append((DOMAIN_PREFIX["CODE"] + q, a))

    law_data = [
        ("什么是合同？", "合同是双方或多方当事人之间设立、变更、终止民事权利义务关系的协议。"),
        ("合同的基本要素是什么？", "合同的基本要素包括：当事人、标的、数量、质量、价款或报酬等。"),
        ("什么是违约责任？", "违约责任是指合同当事人不履行合同义务时应承担的法律责任。"),
        ("什么是侵权责任？", "侵权责任是指行为人因过错侵害他人民事权益应承担的法律后果。"),
        ("什么是知识产权？", "知识产权是指人们对其创造性的智力成果依法享有的专有权利。"),
        ("什么是刑法？", "刑法是规定犯罪、刑事责任和刑罚的法律规范的总和。"),
        ("什么是民法？", "民法是调整平等主体之间财产关系和人身关系的法律规范的总称。"),
        ("What is a contract?", "A contract is an agreement between parties to establish legal obligations."),
        ("What is breach of contract?", "Breach of contract is failing to fulfill contractual obligations."),
        ("What is intellectual property?", "Intellectual property refers to exclusive rights for creative works."),
        ("What is criminal law?", "Criminal law defines crimes and penalties."),
        ("What is civil law?", "Civil law regulates relationships between individuals."),
        ("Was ist ein Vertrag?", "Ein Vertrag ist eine Vereinbarung zur Begründung von Rechtsbeziehungen."),
        ("Was ist Vertragsverletzung?", "Vertragsverletzung bedeutet Nichterfüllung der vertraglichen Pflichten."),
        ("Was ist Strafrecht?", "Strafrecht definiert Straftaten und Strafen."),
    ]
    for q, a in law_data:
        all_pairs.append((DOMAIN_PREFIX["LAW"] + q, a))

    finance_data = [
        ("什么是股票？", "股票是股份公司发行的所有权凭证，代表持有者对公司的部分所有权。"),
        ("什么是基金？", "基金是一种集合投资方式，由众多投资者出资，由专业基金经理管理投资。"),
        ("什么是债券？", "债券是政府、金融机构或企业发行的债务凭证，承诺按约定支付利息和偿还本金。"),
        ("什么是汇率？", "汇率是两种货币之间的兑换比率。"),
        ("什么是通货膨胀？", "通货膨胀是指货币购买力下降，物价普遍上涨的现象。"),
        ("什么是GDP？", "GDP即国内生产总值，是衡量一个国家经济状况的重要指标。"),
        ("什么是利率？", "利率是借贷资金的价格，通常以百分比表示。"),
        ("什么是期货？", "期货是一种标准化的合约，约定在未来某个时间以约定价格买卖标的资产。"),
        ("什么是期权？", "期权是一种权利合约，赋予持有者在特定时间内以特定价格买卖标的资产的权利。"),
        ("What is a stock?", "A stock represents ownership in a corporation and a claim on its assets and earnings."),
        ("What is a mutual fund?", "A mutual fund pools money from investors to invest in diversified securities."),
        ("What is a bond?", "A bond is a debt security issued to raise capital with a promise to repay."),
        ("What is inflation?", "Inflation is the rate of increase in prices of goods and services."),
        ("Was ist eine Aktie?", "Eine Aktie stellt einen Anteil am Unternehmen dar."),
        ("Was ist ein Fonds?", "Ein Fonds sammelt Geld von Investoren zur gemeinsamen Anlage."),
        ("Was ist eine Anleihe?", "Eine Anleihe ist ein Schuldtitel zur Kapitalaufnahme."),
    ]
    for q, a in finance_data:
        all_pairs.append((DOMAIN_PREFIX["FINANCE"] + q, a))

    physics_data = [
        ("什么是牛顿第一定律？", "牛顿第一定律，也称为惯性定律，指出物体在没有外力作用的情况下将保持静止或匀速直线运动状态。"),
        ("什么是牛顿第二定律？", "牛顿第二定律指出，物体的加速度与所受合力成正比，与物体质量成反比，公式为F=ma。"),
        ("什么是牛顿第三定律？", "牛顿第三定律指出，相互作用的两个物体之间的作用力和反作用力大小相等，方向相反。"),
        ("什么是万有引力定律？", "万有引力定律指出，任何两个物体之间都存在相互吸引的力。"),
        ("什么是相对论？", "相对论是爱因斯坦提出的物理学理论，分为狭义相对论和广义相对论。"),
        ("什么是量子力学？", "量子力学是研究微观粒子行为的物理学分支。"),
        ("什么是光电效应？", "光电效应是指当光照射到金属表面时，金属会发射出电子的现象。"),
        ("什么是电磁波？", "电磁波是由振荡的电场和磁场组成的波动现象。"),
        ("什么是热力学第一定律？", "热力学第一定律，也称为能量守恒定律，指出能量既不能被创造也不能被消灭。"),
        ("什么是熵？", "熵是热力学中衡量系统无序程度的物理量。"),
        ("What is Newton's first law?", "Newton's first law states that an object at rest stays at rest and an object in motion stays in motion unless acted upon by an unbalanced force."),
        ("What is Newton's second law?", "Newton's second law states that acceleration equals net force divided by mass, or F = ma."),
        ("What is the law of universal gravitation?", "The law of universal gravitation states that every particle attracts every other particle with a force proportional to the product of their masses."),
        ("What is relativity?", "Relativity is Albert Einstein's theory that describes the relationship between space, time, and gravity."),
        ("What is quantum mechanics?", "Quantum mechanics describes the behavior of particles at the atomic and subatomic level."),
        ("Was ist der erste Newtonsche Gesetz?", "Das erste Newtonsche Gesetz besagt, dass ein Körper in Ruhe bleibt oder sich gleichförmig geradlinig bewegt, solange keine äußeren Kräfte auf ihn wirken."),
        ("Was ist Relativitätstheorie?", "Die Relativitätstheorie von Albert Einstein beschreibt die Beziehung zwischen Raum, Zeit und Gravitation."),
        ("Was ist Quantenmechanik?", "Die Quantenmechanik beschreibt das Verhalten von Teilchen auf atomarer Ebene."),
    ]
    for q, a in physics_data:
        all_pairs.append((DOMAIN_PREFIX["PHYSICS"] + q, a))

    multi_turn_data = [
        ("用户：你好！\\n助手：你好！我是NextAI，很高兴认识你。\\n用户：你能帮我什么？", "我可以帮你回答问题、翻译语言、编写代码、解释法律和金融概念。"),
        ("用户：什么是股票？\\n助手：股票是股份公司发行的所有权凭证。\\n用户：那基金呢？", "基金是集合投资方式，由专业基金经理管理投资组合，分散风险。"),
        ("用户：今天天气怎么样？\\n助手：抱歉，我没有实时天气信息。\\n用户：那明天呢？", "我也无法预测天气，建议查看天气预报应用。"),
        ("User: Hello!\\nAssistant: Hello! I am NextAI.\\nUser: What can you do?", "I can answer questions, translate, write code, and explain concepts."),
        ("User: What is AI?\\nAssistant: AI is the simulation of human intelligence.\\nUser: What about ML?", "Machine learning is a subset of AI that learns from data."),
        ("Benutzer: Hallo!\\nAssistent: Hallo! Ich bin NextAI.\\nBenutzer: Was kannst du?", "Ich kann Fragen beantworten, übersetzen und Code schreiben."),
    ]
    for q, a in multi_turn_data:
        all_pairs.append((DOMAIN_PREFIX["CHAT"] + q, a))

    roleplay_data = [
        ("角色：你是一位友好的导游\\n用户：你好，我想去北京旅游，有什么推荐？", "你好！北京有很多著名景点，推荐你去故宫、天安门广场和长城。"),
        ("角色：你是一位专业的编程老师\\n用户：我想学Python，从哪里开始？", "很好的选择！建议从基础语法开始，先学习变量、数据类型和函数。"),
        ("角色：你是一位耐心的心理顾问\\n用户：最近压力很大怎么办？", "压力大是很常见的，建议尝试深呼吸、适当运动和与人交流。"),
        ("角色：你是一位美食家\\n用户：中国有哪些著名美食？", "中国美食丰富多样！推荐川菜、粤菜、鲁菜和湘菜，各有特色。"),
        ("Role: You are a friendly tour guide\\nUser: Hello, I want to visit Paris, what do you recommend?", "Hello! Paris has many amazing attractions like Eiffel Tower, Louvre Museum and Notre-Dame Cathedral."),
        ("Role: You are a professional cooking instructor\\nUser: How to cook pasta?", "To cook pasta: boil water, add salt, cook pasta al dente, then add sauce."),
        ("Rolle: Du bist ein freundlicher Reiseleiter\\nBenutzer: Hallo, ich möchte nach Berlin reisen, was empfehlen Sie?", "Hallo! Berlin hat viele interessante Orte wie den Brandenburger Tor und das Museuminsel."),
    ]
    for q, a in roleplay_data:
        all_pairs.append((DOMAIN_PREFIX["ROLEPLAY"] + q, a))

    md_data = [
        ("请用Markdown格式介绍Python", "# Python 简介\\n\\nPython是一种高级编程语言，以简洁和易读著称。\\n\\n## 特点\\n\\n- 语法简洁\\n- 跨平台\\n- 丰富的库\\n\\n## 应用领域\\n\\n- Web开发\\n- 数据科学\\n- 人工智能"),
        ("请用Markdown格式列出学习计划", "# 学习计划\\n\\n## 第一周\\n\\n1. 基础语法\\n2. 变量和数据类型\\n\\n## 第二周\\n\\n1. 函数定义\\n2. 条件语句\\n\\n## 第三周\\n\\n1. 循环结构\\n2. 文件操作"),
        ("请用Markdown格式写一份代码示例", "```python\\ndef hello(name):\\n    print(f'Hello, {name}!')\\n\\nhello('World')\\n```"),
        ("请用Markdown格式回答什么是机器学习", "# 机器学习\\n\\n机器学习是人工智能的一个分支，让计算机从数据中学习模式。\\n\\n## 主要类型\\n\\n- **监督学习**: 使用标注数据训练\\n- **无监督学习**: 从无标注数据中发现结构\\n- **强化学习**: 通过奖励机制学习"),
    ]
    for q, a in md_data:
        all_pairs.append((DOMAIN_PREFIX["MD"] + q, a))

    return all_pairs


def main():
    if not HAS_TORCH:
        print("ERROR: 需要 PyTorch")
        return

    print("=" * 60)
    print("NextAI-LSTM v7 - MD格式 + 多轮对话 + 角色扮演 + EOS自终止")
    print("设备: {}, d_model={}, hidden={}, n_layers={}, max_len={}".format(
        DEVICE, CFG["d_model"], CFG["hidden_size"], CFG["n_layers"], CFG["max_len"]))
    print("=" * 60)

    print("\n[1/4] 构建训练数据...")
    all_pairs = build_all_data()
    all_pairs = all_pairs * 10
    random.shuffle(all_pairs)
    print("  总训练样本: {}".format(len(all_pairs)))

    print("\n[2/4] 编码数据...")
    tokenizer = ByteTokenizer()
    encoded_pairs = []
    for src_text, tgt_text in all_pairs:
        src_ids = tokenizer.encode(src_text, max_len=CFG["max_len"])
        tgt_ids = tokenizer.encode(tgt_text, max_len=CFG["max_len"])
        if 3 <= len(src_ids) <= CFG["max_len"] and 3 <= len(tgt_ids) <= CFG["max_len"]:
            encoded_pairs.append((src_ids, tgt_ids))
    print("  有效训练样本: {}".format(len(encoded_pairs)))

    print("\n[3/4] 构建模型...")
    model = NextAILSTM(CFG).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("  模型参数: {}".format(n_params))

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["rounds"])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    test_samples = [
        ("IDENTITY", "你是谁？", "NextAI"),
        ("IDENTITY", "What is your name?", "NextAI"),
        ("TRANSLATE", "translate to English: 你好", "Hello"),
        ("TRANSLATE", "翻译成中文：Hello", "你好"),
        ("CODE", "Write a Python function to calculate factorial", "def"),
        ("CODE", "Write a C function to add two integers", "int"),
        ("LAW", "什么是合同？", "合同"),
        ("LAW", "What is a contract?", "contract"),
        ("FINANCE", "什么是股票？", "股票"),
        ("FINANCE", "What is a stock?", "stock"),
        ("PHYSICS", "什么是牛顿第一定律？", "牛顿"),
        ("PHYSICS", "What is Newton's first law?", "Newton"),
        ("CHAT", "用户：你好！\\n助手：你好！\\n用户：你能帮我什么？", "回答问题"),
        ("ROLEPLAY", "角色：你是导游\\n用户：北京有什么推荐？", "故宫"),
        ("MD", "请用Markdown介绍Python", "#"),
    ]

    print("\n开始训练 ({} 轮, 每轮 ≤ {} 秒)...".format(CFG["rounds"], CFG["max_round_seconds"]))
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
            if len(batch) < 2:
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

        if r % 5 == 0 or r == CFG["rounds"]:
            model.eval()
            print("  ---- 测试输出 ----")
            n_ok = 0
            for domain, q, expected in test_samples:
                src_ids = tokenizer.encode(DOMAIN_PREFIX[domain] + q, max_len=CFG["max_len"])
                out_ids = model.generate(src_ids, max_new=80)
                gen = tokenizer.decode(out_ids)
                marker = "✓" if (expected[:8] in gen or expected.lower()[:8] in gen.lower()) and len(gen) > 3 else "·"
                n_ok += 1 if marker == "✓" else 0
                if marker == "✓":
                    print("  {} {}: {}".format(marker, domain, gen[:80]))
            print("  {}/{}\n".format(n_ok, len(test_samples)))

    print("\n[4/4] 保存模型...")
    torch.save({
        "model": model.state_dict(),
        "cfg": CFG,
        "tokenizer": {"b2i": {}, "i2b": {}, "merges": [], "vocab_size": CFG["vocab_size"]},
        "model_type": "lstm",
    }, "/workspace/nextai-full.pt")
    print("  模型已保存到 /workspace/nextai-full.pt")


if __name__ == "__main__":
    main()
