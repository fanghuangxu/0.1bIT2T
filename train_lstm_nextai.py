#!/usr/bin/env python3
"""NextAI-LSTM v6: 领域前缀 + EOS自终止 + 零乱码 + 更大模型。"""
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

# ========= 超参数 =========
# 单卡2GB显存估算:
#   embedding: 1024 * 512 * 2(encoder+decoder) = 1M
#   encoder 4层 bidirectional LSTM: hidden=256 each dir, -> 512
#   decoder 4层 LSTM + attention: ~10M
#   output linear: 512*1024 = 0.5M
#   总计 ~15-20M 参数, float32 = ~80MB (前向)
#   加上 optimizer + grad = ~240MB, 远低于2GB
CFG = {
    "vocab_size": 260,   # 4 special + 256 bytes，直接映射，无 BPE
    "d_model": 512,
    "hidden_size": 512,
    "n_layers": 4,
    "max_len": 256,
    "dropout": 0.10,
    "lr": 3e-4,
    "batch_size": 16,
    "rounds": 60,
    "max_round_seconds": 290,
}


class ByteTokenizer:
    """纯 byte-level 分词器：每个 UTF-8 字节直接映射为 token id。

    严格使用 errors='ignore' 解码，保证无乱码。

    id 分配:
      0: PAD, 1: BOS, 2: EOS, 3: UNK
      4..259: 字节值 0..255 -> id=byte+4
    """

    def __init__(self, b2i=None, i2b=None, merges=None, vocab_size=260):
        self.vocab_size = vocab_size
        self.merges = []
        # b2i, i2b 保留用于与 chat_nextai.py 的保存加载接口兼容
        self.b2i = b2i if b2i is not None else {}
        self.i2b = i2b if i2b is not None else {}

    def encode(self, text, max_len=None):
        # 严格 ignore: 让输入不会出现无效字节 -> 不会产生替换字符
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
            # 其他 id 跳过，不产生乱码
        try:
            return bytes(raw).decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def train(self, texts, target_vocab=None):
        # 纯 byte-level，不需要训练
        self.vocab_size = target_vocab or 260
        return


# ========= 模型 =========
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
        """贪心解码，严格以 EOS 作为终止信号。"""
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
            if next_tok == EOS:
                break
            if next_tok == PAD or next_tok == UNK:
                break
            generated.append(next_tok)
            prev_tok = next_tok
        return generated[1:]

    @torch.no_grad()
    def generate_stream(self, src_ids, max_new=120, tokenizer=None):
        """流式生成，yield 每个 token id，自动抑制 UNK/乱码。"""
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


# ========= 领域数据集构建 =========
# 核心策略: 每个训练样本的输入都以 "[DOMAIN] " 前缀开始
# 训练目标是在回答末尾输出 EOS (由 tokenizer 自动加上)

DOMAIN_PREFIX = {
    "IDENTITY": "[IDENTITY] ",
    "TRANSLATE": "[TRANSLATE] ",
    "QA": "[QA] ",
    "CODE": "[CODE] ",
    "LAW": "[LAW] ",
    "FINANCE": "[FINANCE] ",
    "PHYSICS": "[PHYSICS] ",
}


def build_identity_pairs():
    data = [
        ("你是谁？", "我是NextAI，一个AI助手。"),
        ("你的名字是什么？", "我的名字是NextAI。"),
        ("Who are you?", "I am NextAI, an AI assistant."),
        ("What is your name?", "My name is NextAI."),
        ("你叫什么名字？", "我的名字是NextAI，由NextAI团队开发。"),
        ("请介绍一下你自己。", "你好！我是NextAI，一个AI助手，可以回答问题、翻译和编写代码。"),
        ("你是人类吗？", "不，我是NextAI，一个AI助手。"),
        ("Are you human?", "No, I am NextAI, an AI assistant."),
        ("你能做什么？", "我是NextAI，可以回答问题、翻译、编写代码、解释法律、金融和物理概念。"),
        ("What can you do?", "I am NextAI. I can answer questions, translate, write code, explain law, finance and physics."),
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
        ("介绍一下你自己", "我是NextAI，一个AI助手，可以回答问题、翻译和编写代码。"),
    ]
    return [(DOMAIN_PREFIX["IDENTITY"] + q, a) for q, a in data]


def build_translation_pairs():
    data = [
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
    return [(DOMAIN_PREFIX["TRANSLATE"] + q, a) for q, a in data]


def build_general_qa_pairs():
    data = [
        ("What is AI?", "Artificial Intelligence is the simulation of human intelligence by machines."),
        ("What is machine learning?", "Machine learning is a subset of AI that enables systems to learn from data."),
        ("What is Python?", "Python is a popular high-level programming language."),
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the capital of Japan?", "Tokyo is the capital of Japan."),
        ("What is the capital of China?", "Beijing is the capital of China."),
        ("What is 2 plus 3?", "2 plus 3 equals 5."),
        ("What is 10 minus 4?", "10 minus 4 equals 6."),
        ("What color is the sky?", "The sky is blue."),
        ("What is the largest ocean?", "The Pacific Ocean is the largest ocean."),
        ("什么是人工智能？", "人工智能是指由计算机模拟人类智能的技术。"),
        ("什么是机器学习？", "机器学习是人工智能的一个分支，让计算机从数据中学习。"),
        ("法国的首都是什么？", "法国的首都是巴黎。"),
        ("中国的首都是什么？", "中国的首都是北京。"),
        ("日本的首都是什么？", "日本的首都是东京。"),
        ("天空是什么颜色？", "天空是蓝色的。"),
        ("世界上最大的海洋是什么？", "世界上最大的海洋是太平洋。"),
        ("2加3等于几？", "2加3等于5。"),
        ("10减4等于几？", "10减4等于6。"),
        ("水的化学式是什么？", "水的化学式是H2O。"),
        ("地球是行星还是恒星？", "地球是行星。"),
        ("一年有多少天？", "一年通常有365天，闰年有366天。"),
        ("一周有多少天？", "一周有7天。"),
        ("一小时有多少分钟？", "一小时有60分钟。"),
        ("Python 是什么？", "Python是一种流行的高级编程语言。"),
        ("How many days are in a week?", "There are 7 days in a week."),
        ("How many hours in a day?", "There are 24 hours in a day."),
        ("What is water made of?", "Water is made of hydrogen and oxygen, H2O."),
        ("太阳是什么？", "太阳是一颗恒星，是太阳系的中心。"),
    ]
    return [(DOMAIN_PREFIX["QA"] + q, a) for q, a in data]


def load_code_data(max_samples=2000):
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
                        if instruction and code and len(instruction) < 200 and len(code) < 400:
                            pairs.append((instruction[:200], code[:400]))
                    except Exception:
                        continue
        except Exception:
            pass

    built_in_code = [
        ("Write a Python function to calculate factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
        ("Write a Python function to check prime number", "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
        ("Python function to calculate Fibonacci", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)"),
        ("Python function to sort list of numbers", "def sort_list(lst):\n    return sorted(lst)"),
        ("Python function to reverse a string", "def reverse_string(s):\n    return s[::-1]"),
        ("Python function to find average of list", "def average(lst):\n    return sum(lst) / len(lst)"),
        ("Python function to count vowels", "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"),
        ("Python function to read a file", "def read_file(filename):\n    with open(filename, 'r') as f:\n        return f.read()"),
        ("Python function to write to a file", "def write_file(filename, content):\n    with open(filename, 'w') as f:\n        f.write(content)"),
        ("如何用Python读取文件", "with open('file.txt', 'r', encoding='utf-8') as f:\n    content = f.read()\nprint(content)"),
        ("Python如何定义函数", "def my_function():\n    print('Hello')"),
        ("Python如何创建字典", "d = {'name': 'John', 'age': 25}"),
        ("用Python写一个冒泡排序", "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n - 1):\n        for j in range(n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr"),
        ("用Python写一个快速排序", "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)"),
        ("Write a C function to add two integers", "int add(int a, int b) {\n    return a + b;\n}"),
        ("Write a C function to swap two numbers", "void swap(int *a, int *b) {\n    int temp = *a;\n    *a = *b;\n    *b = temp;\n}"),
        ("C function to find factorial", "int factorial(int n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}"),
        ("C function for binary search", "int binarySearch(int arr[], int l, int r, int x) {\n    if (r >= l) {\n        int mid = l + (r - l) / 2;\n        if (arr[mid] == x) return mid;\n        if (arr[mid] > x) return binarySearch(arr, l, mid - 1, x);\n        return binarySearch(arr, mid + 1, r, x);\n    }\n    return -1;\n}"),
        ("C program to print Hello World", "#include <stdio.h>\nint main() {\n    printf('Hello World\\n');\n    return 0;\n}"),
        ("C++ function for bubble sort", "void bubbleSort(int arr[], int n) {\n    for (int i = 0; i < n - 1; i++)\n        for (int j = 0; j < n - i - 1; j++)\n            if (arr[j] > arr[j + 1])\n                swap(arr[j], arr[j + 1]);\n}"),
        ("C++ class for stack", "class Stack {\nprivate:\n    vector<int> data;\npublic:\n    void push(int x) { data.push_back(x); }\n    int pop() { int x = data.back(); data.pop_back(); return x; }\n    bool empty() { return data.empty(); }\n};"),
        ("C++ class for linked list", "class Node {\npublic:\n    int data;\n    Node* next;\n    Node(int val) : data(val), next(nullptr) {}\n};"),
        ("C++ function to compute GCD", "int gcd(int a, int b) {\n    if (b == 0) return a;\n    return gcd(b, a % b);\n}"),
        ("C++ Hello World program", "#include <iostream>\nusing namespace std;\nint main() {\n    cout << 'Hello World' << endl;\n    return 0;\n}"),
        ("Java function to calculate factorial", "public static int factorial(int n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}"),
        ("Java Hello World", "public class Main {\n    public static void main(String[] args) {\n        System.out.println('Hello World');\n    }\n}"),
        ("JavaScript function to calculate factorial", "function factorial(n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}"),
        ("JavaScript Hello World", "console.log('Hello World');"),
        ("Go function to calculate factorial", "func factorial(n int) int {\n    if n <= 1 { return 1 }\n    return n * factorial(n-1)\n}"),
        ("Rust function to calculate factorial", "fn factorial(n: u64) -> u64 {\n    if n <= 1 { 1 } else { n * factorial(n - 1) }\n}"),
        ("SQL query to select all from table", "SELECT * FROM table_name;"),
        ("Bash script Hello World", "#!/bin/bash\necho 'Hello World'"),
        ("HTML basic page structure", "<!DOCTYPE html>\n<html>\n<head><title>My Page</title></head>\n<body>\n    <h1>Hello World</h1>\n</body>\n</html>"),
        ("how make os in c plus plus", "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\ntypedef struct {\n    char name[50];\n    int size;\n} FileEntry;\n\nFileEntry fat32[256];\nint fat_count = 0;\n\nvoid init_disk() {\n    for (int i = 0; i < 256; i++)\n        fat32[i].name[0] = 0;\n}\n\nint create_file(char *name) {\n    if (fat_count >= 256) return -1;\n    strncpy(fat32[fat_count].name, name, 49);\n    fat32[fat_count].size = 0;\n    return fat_count++;\n}"),
    ]
    for _ in range(20):
        pairs.extend(built_in_code)
    pairs = pairs[:max_samples * 2]
    return [(DOMAIN_PREFIX["CODE"] + q, a) for q, a in pairs]


def load_legal_data(max_samples=1000):
    pairs = []
    local_path = "/workspace/legal_sample.parquet"
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
        except Exception:
            pass

    built_in_legal = [
        ("什么是合同？", "合同是双方或多方当事人之间设立、变更、终止民事权利义务关系的协议。"),
        ("合同的基本要素是什么？", "合同的基本要素包括：当事人、标的、数量、质量、价款或报酬等。"),
        ("什么是违约责任？", "违约责任是指合同当事人不履行合同义务时应承担的法律责任。"),
        ("什么是侵权责任？", "侵权责任是指行为人因过错侵害他人民事权益应承担的法律后果。"),
        ("什么是知识产权？", "知识产权是指人们对其创造性的智力成果依法享有的专有权利。"),
        ("什么是公司法？", "公司法是规定公司设立、组织、活动、解散的法律规范的总称。"),
        ("什么是刑法？", "刑法是规定犯罪、刑事责任和刑罚的法律规范的总和。"),
        ("什么是民法？", "民法是调整平等主体之间财产关系和人身关系的法律规范的总称。"),
        ("什么是劳动合同？", "劳动合同是劳动者与用人单位之间确立劳动关系的协议。"),
        ("什么是保险法？", "保险法是规范保险活动的法律规范的总称。"),
        ("What is a contract?", "A contract is an agreement between parties to establish legal obligations."),
        ("What is breach of contract?", "Breach of contract is failing to fulfill contractual obligations."),
        ("What is intellectual property?", "Intellectual property refers to exclusive rights for creative works."),
        ("What is criminal law?", "Criminal law defines crimes and penalties."),
        ("What is civil law?", "Civil law regulates relationships between individuals."),
        ("What is tort law?", "Tort law addresses civil wrongs and damages."),
        ("What is property law?", "Property law governs ownership and use of property."),
        ("Was ist ein Vertrag?", "Ein Vertrag ist eine Vereinbarung zur Begründung von Rechtsbeziehungen."),
        ("Was ist Vertragsverletzung?", "Vertragsverletzung bedeutet Nichterfüllung der vertraglichen Pflichten."),
        ("Was ist geistiges Eigentum?", "Geistiges Eigentum sind exklusive Rechte für kreative Werke."),
        ("Was ist Strafrecht?", "Strafrecht definiert Straftaten und Strafen."),
        ("Was ist Zivilrecht?", "Zivilrecht regelt Beziehungen zwischen Privatpersonen."),
        ("Was ist Haftpflicht?", "Haftpflicht bezieht sich auf zivilrechtliche Verantwortung für Schäden。"),
    ]
    for _ in range(20):
        pairs.extend(built_in_legal)
    pairs = pairs[:max_samples * 2]
    return [(DOMAIN_PREFIX["LAW"] + q, a) for q, a in pairs]


def load_finance_data(max_samples=1000):
    pairs = []
    local_path = "/workspace/finance_sample.parquet"
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
        except Exception:
            pass

    built_in_finance = [
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
        ("什么是市盈率？", "市盈率是股票价格与每股收益的比率，用于评估股票估值。"),
        ("什么是风险管理？", "风险管理是指识别、评估和控制投资过程中各种风险的过程。"),
        ("什么是复利？", "复利是指在每个计息周期结束时，将利息加入本金再计息的方式。"),
        ("什么是货币基金？", "货币基金是投资于短期货币市场工具的开放式基金。"),
        ("什么是ETF？", "ETF是交易型开放式指数基金，可以在交易所买卖。"),
        ("What is a stock?", "A stock represents ownership in a corporation and a claim on its assets and earnings."),
        ("What is a mutual fund?", "A mutual fund pools money from investors to invest in diversified securities."),
        ("What is a bond?", "A bond is a debt security issued to raise capital with a promise to repay."),
        ("What is exchange rate?", "Exchange rate is the value of one currency expressed in another."),
        ("What is inflation?", "Inflation is the rate of increase in prices of goods and services."),
        ("What is ROI?", "ROI is return on investment, measuring the profitability of an investment."),
        ("What is diversification?", "Diversification is spreading investments across different assets to reduce risk."),
        ("Was ist eine Aktie?", "Eine Aktie stellt einen Anteil am Unternehmen dar。"),
        ("Was ist ein Fonds?", "Ein Fonds sammelt Geld von Investoren zur gemeinsamen Anlage。"),
        ("Was ist eine Anleihe?", "Eine Anleihe ist ein Schuldtitel zur Kapitalaufnahme。"),
        ("Was ist Inflation?", "Inflation ist der Anstieg des allgemeinen Preisniveaus。"),
        ("Was ist Rendite?", "Rendite ist der Gewinn oder Verlust einer Anlage。"),
    ]
    for _ in range(20):
        pairs.extend(built_in_finance)
    pairs = pairs[:max_samples * 2]
    return [(DOMAIN_PREFIX["FINANCE"] + q, a) for q, a in pairs]


def load_physics_data(max_samples=1000):
    pairs = []
    local_path = "/workspace/physics_sample.json"
    if os.path.exists(local_path):
        try:
            import json
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data[:max_samples]:
                    try:
                        question = item.get("question", "").strip()
                        answer = item.get("answer", "").strip()
                        if question and answer and len(question) < 200 and len(answer) < 400:
                            pairs.append((question[:200], answer[:400]))
                    except Exception:
                        continue
        except Exception:
            pass

    built_in_physics = [
        ("什么是牛顿第一定律？", "牛顿第一定律，也称为惯性定律，指出物体在没有外力作用的情况下将保持静止或匀速直线运动状态。这个定律是经典力学的基础之一。"),
        ("什么是牛顿第二定律？", "牛顿第二定律指出，物体的加速度与所受合力成正比，与物体质量成反比，公式为F=ma。"),
        ("什么是牛顿第三定律？", "牛顿第三定律指出，相互作用的两个物体之间的作用力和反作用力大小相等，方向相反。"),
        ("什么是万有引力定律？", "万有引力定律是牛顿提出的，它指出任何两个物体之间都存在相互吸引的力，这个力的大小与它们质量的乘积成正比，与它们距离的平方成反比。"),
        ("什么是相对论？", "相对论是爱因斯坦提出的物理学理论，分为狭义相对论和广义相对论。狭义相对论提出了质能方程E=mc²。广义相对论将引力解释为时空的弯曲。"),
        ("什么是量子力学？", "量子力学是研究微观粒子行为的物理学分支。它描述了粒子在原子和亚原子尺度上的奇特性质，如波粒二象性、量子叠加态和量子纠缠等现象。"),
        ("什么是光电效应？", "光电效应是指当光照射到金属表面时，金属会发射出电子的现象。爱因斯坦解释了这一现象，证明了光具有粒子性。"),
        ("什么是电磁波？", "电磁波是由振荡的电场和磁场组成的波动现象。它包括无线电波、微波、红外线、可见光、紫外线、X射线和伽马射线。"),
        ("什么是热力学第一定律？", "热力学第一定律，也称为能量守恒定律，指出能量既不能被创造也不能被消灭，只能从一种形式转化为另一种形式。"),
        ("什么是熵？", "熵是热力学中衡量系统无序程度的物理量。根据热力学第二定律，孤立系统的熵总是不会减少，意味着系统会趋向于更加无序的状态。"),
        ("什么是核聚变？", "核聚变是指轻原子核结合成较重原子核的过程，在此过程中会释放出巨大的能量。太阳和恒星的能量就来自于核聚变反应。"),
        ("什么是核裂变？", "核裂变是指重原子核分裂成较轻原子核的过程，同时释放大量能量。原子弹和核电站利用的就是核裂变。"),
        ("什么是黑洞？", "黑洞是一种引力极强的天体，其逃逸速度超过光速。任何物质，包括光，一旦进入黑洞的事件视界，就无法逃脱。"),
        ("什么是摩擦力？", "摩擦力是两个物体接触时产生的阻碍相对运动的力。它分为静摩擦力和动摩擦力。摩擦力的大小与接触面的粗糙程度和正压力有关。"),
        ("什么是动量守恒？", "动量守恒定律指出，在没有外力作用的封闭系统中，总动量保持不变。这意味着系统中各物体动量的矢量和在相互作用前后保持相等。"),
        ("什么是角动量？", "角动量是描述物体旋转运动的物理量。对于绕固定轴旋转的物体，角动量等于转动惯量乘以角速度。"),
        ("什么是波动？", "波动是振动在介质中的传播过程。波动可以分为横波和纵波。横波中质点振动方向与波的传播方向垂直，如电磁波和水波。纵波中质点振动方向与波的传播方向平行，如声波。"),
        ("What is Newton's first law?", "Newton's first law, also known as the law of inertia, states that an object at rest stays at rest and an object in motion stays in motion unless acted upon by an unbalanced force."),
        ("What is Newton's second law?", "Newton's second law states that acceleration equals net force divided by mass, or F = ma."),
        ("What is Newton's third law?", "Newton's third law states that for every action there is an equal and opposite reaction."),
        ("What is the law of universal gravitation?", "The law of universal gravitation states that every particle attracts every other particle with a force proportional to the product of their masses and inversely proportional to the square of the distance."),
        ("What is relativity?", "Relativity is Albert Einstein's theory that describes the relationship between space, time, and gravity. Special relativity deals with constant velocity frames, while general relativity explains gravity as the curvature of spacetime."),
        ("What is quantum mechanics?", "Quantum mechanics is the branch of physics that describes the behavior of particles at the atomic and subatomic level, including phenomena like wave-particle duality, superposition, and entanglement."),
        ("What is the photoelectric effect?", "The photoelectric effect is the emission of electrons from a material when light shines on it. Einstein explained this by proposing that light consists of discrete packets of energy called photons."),
        ("What is momentum?", "Momentum is a measure of an object's motion, calculated as the product of its mass and velocity, p = mv. The law of conservation of momentum states that the total momentum of a closed system remains constant。"),
        ("What is angular momentum?", "Angular momentum is the rotational equivalent of linear momentum. It is calculated as the product of moment of inertia and angular velocity. Angular momentum is conserved in the absence of external torque。"),
        ("Was ist der erste Newtonsche Gesetz?", "Das erste Newtonsche Gesetz, auch Trägheitsgesetz genannt, besagt, dass ein Körper in Ruhe bleibt oder sich gleichförmig geradlinig bewegt, solange keine äußeren Kräfte auf ihn wirken。"),
        ("Was ist die Gravitationskraft?", "Die Gravitationskraft ist die Anziehungskraft zwischen zwei Massen. Sie ist proportional zum Produkt der Massen und umgekehrt proportional zum Quadrat des Abstands。"),
        ("Was ist Relativitätstheorie?", "Die Relativitätstheorie von Albert Einstein beschreibt die Beziehung zwischen Raum, Zeit und Gravitation。"),
        ("Was ist Quantenmechanik?", "Die Quantenmechanik ist ein Teilgebiet der Physik, das das Verhalten von Teilchen auf atomarer und subatomarer Ebene beschreibt。"),
    ]
    for _ in range(20):
        pairs.extend(built_in_physics)
    print("  从物理数据集加载 {} 条问答".format(len(pairs)))
    return [(DOMAIN_PREFIX["PHYSICS"] + q, a) for q, a in pairs]


def load_xtreme_qa(data_dir, max_samples=200):
    pairs = []
    for fname in os.listdir(data_dir) if os.path.isdir(data_dir) else []:
        if not fname.endswith(".parquet"):
            continue
        try:
            t = pq.read_table(os.path.join(data_dir, fname))
            for i in range(min(len(t), max_samples)):
                try:
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
        except Exception:
            continue
    print("  从 xtreme 加载 {} 条 QA 对".format(len(pairs)))
    return [(DOMAIN_PREFIX["QA"] + q, a) for q, a in pairs]


def load_firefly_data(max_samples=500):
    local_path = "/workspace/firefly_sample.parquet"
    pairs = []
    if os.path.exists(local_path):
        try:
            t = pq.read_table(local_path)
            for i in range(min(len(t), max_samples)):
                try:
                    src = t.column("source")[i].as_py()
                    tgt = t.column("target")[i].as_py()
                    if src and tgt and len(tgt) < 200 and len(src) < 120:
                        pairs.append((src[:120], tgt[:200]))
                except Exception:
                    continue
        except Exception:
            pass
    print("  从 Firefly 加载 {} 条对话对".format(len(pairs)))
    return [(DOMAIN_PREFIX["QA"] + q, a) for q, a in pairs]


# ========= 主训练流程 =========
def main():
    if not HAS_TORCH:
        print("错误: PyTorch 不可用，请安装 PyTorch 后再运行")
        return

    print("=" * 60)
    print("NextAI-LSTM v6 - 领域前缀 + EOS自终止 + 零乱码")
    print("设备: {}, d_model={}, hidden={}, n_layers={}, max_len={}".format(
        DEVICE, CFG["d_model"], CFG["hidden_size"], CFG["n_layers"], CFG["max_len"]))
    print("=" * 60)

    print("\n[1/5] 加载数据集...")
    identity_pairs = build_identity_pairs()
    translation_pairs = build_translation_pairs()
    qa_pairs_builtin = build_general_qa_pairs()
    qa_pairs_xtreme = load_xtreme_qa("/workspace/xtreme_data", max_samples=200)
    firefly_pairs = load_firefly_data(max_samples=300)
    code_pairs = load_code_data(max_samples=1000)
    legal_pairs = load_legal_data(max_samples=800)
    finance_pairs = load_finance_data(max_samples=800)
    physics_pairs = load_physics_data(max_samples=1000)

    all_texts = []
    for s, t in identity_pairs + translation_pairs + qa_pairs_builtin + qa_pairs_xtreme + firefly_pairs + code_pairs + legal_pairs + finance_pairs + physics_pairs:
        all_texts.extend([s, t])
    print("  总文本数: {}".format(len(all_texts)))

    print("\n[2/5] 构建分词器...")
    tokenizer = ByteTokenizer()
    tokenizer.train(all_texts, CFG["vocab_size"])
    print("  vocab={}, merges={}".format(tokenizer.vocab_size, len(tokenizer.merges)))

    print("\n[3/5] 构建训练数据...")
    # 对各领域做等权重采样，避免某一领域主导
    all_pairs = (
        identity_pairs * 12 +
        translation_pairs * 10 +
        qa_pairs_builtin * 6 +
        qa_pairs_xtreme +
        firefly_pairs +
        code_pairs +
        legal_pairs +
        finance_pairs +
        physics_pairs
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

    print("\n[4/5] 构建模型...")
    model = NextAILSTM(CFG).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("  模型参数: {}".format(n_params))
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["rounds"])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.10)

    # 测试样本 - 覆盖所有领域（不带前缀，由下面测试逻辑加）
    test_samples = [
        ("IDENTITY", "你是谁？", "NextAI"),
        ("IDENTITY", "你的名字是什么？", "NextAI"),
        ("IDENTITY", "What is your name?", "NextAI"),
        ("IDENTITY", "Are you human?", "No"),
        ("TRANSLATE", "translate to English: 你好", "Hello"),
        ("TRANSLATE", "translate to English: 谢谢你", "Thank"),
        ("TRANSLATE", "翻译成中文：Hello", "你好"),
        ("TRANSLATE", "übersetze ins Deutsche: Hello", "Hallo"),
        ("QA", "Q: 法国的首都是什么？", "巴黎"),
        ("QA", "Q: What is the capital of France?", "Paris"),
        ("QA", "Q: 什么是人工智能？", "人工智能"),
        ("CODE", "Write a Python function to calculate factorial", "def factorial"),
        ("CODE", "Write a C function to add two integers", "int add"),
        ("CODE", "how make os in c plus plus", "include"),
        ("CODE", "如何用Python读取文件", "open"),
        ("LAW", "什么是合同？", "合同"),
        ("LAW", "What is a contract?", "contract"),
        ("LAW", "Was ist ein Vertrag?", "Vertrag"),
        ("FINANCE", "什么是股票？", "股票"),
        ("FINANCE", "What is a stock?", "stock"),
        ("FINANCE", "Was ist eine Aktie?", "Aktie"),
        ("FINANCE", "什么是GDP？", "GDP"),
        ("PHYSICS", "什么是牛顿第一定律？", "牛顿"),
        ("PHYSICS", "What is Newton's first law?", "Newton"),
        ("PHYSICS", "Was ist der erste Newtonsche Gesetz?", "Newton"),
        ("PHYSICS", "什么是相对论？", "相对论"),
        ("PHYSICS", "什么是量子力学？", "量子"),
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

        # 每 2 轮做一次详细测试
        if r % 2 == 0 or r == CFG["rounds"]:
            model.eval()
            print("  ---- 测试输出 ----")
            n_ok = 0
            for domain, q, expected in test_samples:
                # 推理时加上与训练一致的领域前缀
                src_ids = tokenizer.encode(DOMAIN_PREFIX[domain] + q, max_len=CFG["max_len"])
                out_ids = model.generate(src_ids, max_new=80)
                gen = tokenizer.decode(out_ids)
                marker = "✓" if (expected[:8] in gen or expected.lower()[:8] in gen.lower()) and len(gen) > 3 else "·"
                if marker == "✓":
                    n_ok += 1
                print("  {} [{}] Q: {}".format(marker, domain, q[:50]))
                print("    输出: {}".format(gen[:120]))
            print("  ------------------- 正确: {}/{}".format(n_ok, len(test_samples)))

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
