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
    nn = None
    F = None

CFG = {
    "vocab_size": 1024,
    "d_model": 160,
    "hidden_size": 160,
    "n_layers": 2,
    "max_len": 200,  # 增大支持长文本输出
    "dropout": 0.05,
    "lr": 5e-4,
    "batch_size": 48,  # 减小batch以适配长文本和显存限制
    "rounds": 12,
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
    """加载代码任务数据集（PolyDevTasks）— 支持所有编程语言"""
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
                        # 支持所有编程语言
                        if instruction and code and len(instruction) < 200 and len(code) < 400:
                            pairs.append((instruction[:200], code[:400]))
                    except Exception:
                        continue
        except Exception as e:
            print("  代码数据加载失败:", e)

    # 内置多种编程语言数据
    built_in_code = [
        # Python
        ("Write a Python function to calculate factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
        ("Write a Python function to check prime number", "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
        ("Python function to calculate Fibonacci", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)"),
        ("Python function to sort list of numbers", "def sort_list(lst):\n    return sorted(lst)"),
        ("Python function to sum list of numbers", "def sum_list(lst):\n    return sum(lst)"),
        ("Python function to reverse a string", "def reverse_string(s):\n    return s[::-1]"),
        ("Python function to find average of list", "def average(lst):\n    return sum(lst) / len(lst)"),
        ("Python function to count vowels", "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"),
        ("Python function to remove duplicates", "def remove_duplicates(lst):\n    return list(set(lst))"),
        ("Python function to merge dictionaries", "def merge_dicts(d1, d2):\n    result = d1.copy()\n    result.update(d2)\n    return result"),
        ("Python function to read a file", "def read_file(filename):\n    with open(filename, 'r') as f:\n        return f.read()"),
        ("Python function to write to a file", "def write_file(filename, content):\n    with open(filename, 'w') as f:\n        f.write(content)"),
        ("Python function for list comprehension", "squares = [x**2 for x in range(10)]"),
        ("Python function using lambda", "square = lambda x: x**2"),
        ("Python class for a simple calculator", "class Calculator:\n    def add(self, a, b): return a + b\n    def multiply(self, a, b): return a * b"),
        ("Python function to generate random numbers", "import random\ndef get_random(min, max):\n    return random.randint(min, max)"),
        ("Python function to get current date", "from datetime import datetime\nprint(datetime.now())"),
        ("如何用Python读取文件", "with open('file.txt', 'r', encoding='utf-8') as f:\n    content = f.read()\nprint(content)"),
        ("Python如何定义函数", "def my_function():\n    print('Hello')"),
        ("Python如何创建字典", "d = {'name': 'John', 'age': 25}"),
        ("Python如何循环", "for i in range(5):\n    print(i)"),
        ("Python字符串格式化", "name = 'World'\nprint(f'Hello {name}')"),

        # C 语言
        ("Write a C function to add two integers", "int add(int a, int b) {\n    return a + b;\n}"),
        ("Write a C function to swap two numbers", "void swap(int *a, int *b) {\n    int temp = *a;\n    *a = *b;\n    *b = temp;\n}"),
        ("C function to calculate power", "int power(int base, int exp) {\n    int result = 1;\n    for (int i = 0; i < exp; i++)\n        result *= base;\n    return result;\n}"),
        ("C function for binary search", "int binarySearch(int arr[], int l, int r, int x) {\n    if (r >= l) {\n        int mid = l + (r - l) / 2;\n        if (arr[mid] == x) return mid;\n        if (arr[mid] > x) return binarySearch(arr, l, mid - 1, x);\n        return binarySearch(arr, mid + 1, r, x);\n    }\n    return -1;\n}"),
        ("C function to find maximum of array", "int max(int arr[], int n) {\n    int m = arr[0];\n    for (int i = 1; i < n; i++)\n        if (arr[i] > m) m = arr[i];\n    return m;\n}"),
        ("C function to count length of string", "int strlength(char *s) {\n    int len = 0;\n    while (s[len] != '\\0') len++;\n    return len;\n}"),
        ("C function to reverse a string", "void reverse(char *s) {\n    int len = strlen(s);\n    for (int i = 0; i < len / 2; i++) {\n        char t = s[i];\n        s[i] = s[len - 1 - i];\n        s[len - 1 - i] = t;\n    }\n}"),
        ("C function to find factorial", "int factorial(int n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}"),
        ("C program to print Hello World", "#include <stdio.h>\nint main() {\n    printf('Hello World\\n');\n    return 0;\n}"),
        ("C function to copy string", "void copy(char *dest, char *src) {\n    while (*src) {\n        *dest = *src;\n        dest++;\n        src++;\n    }\n    *dest = '\\0';\n}"),
        ("C语言如何声明变量", "int x = 10;\nfloat y = 3.14;\nchar c = 'A';\ndouble z = 3.14159;"),

        # C++
        ("C++ function for bubble sort", "void bubbleSort(int arr[], int n) {\n    for (int i = 0; i < n - 1; i++)\n        for (int j = 0; j < n - i - 1; j++)\n            if (arr[j] > arr[j + 1])\n                swap(arr[j], arr[j + 1]);\n}"),
        ("C++ class for stack", "class Stack {\nprivate:\n    vector<int> data;\npublic:\n    void push(int x) { data.push_back(x); }\n    int pop() { int x = data.back(); data.pop_back(); return x; }\n    int top() { return data.back(); }\n    bool empty() { return data.empty(); }\n};"),
        ("C++ function for matrix multiplication", "void multiply(int A[][2], int B[][2], int C[][2]) {\n    for (int i = 0; i < 2; i++)\n        for (int j = 0; j < 2; j++) {\n            C[i][j] = 0;\n            for (int k = 0; k < 2; k++)\n                C[i][j] += A[i][k] * B[k][j];\n        }\n}"),
        ("C++ class for linked list", "class Node {\npublic:\n    int data;\n    Node* next;\n    Node(int val) : data(val), next(nullptr) {}\n};\n\nclass LinkedList {\n    Node* head;\npublic:\n    LinkedList() : head(nullptr) {}\n    void insert(int v) {\n        Node* n = new Node(v);\n        n->next = head;\n        head = n;\n    }\n};"),
        ("C++ function to find minimum", "int min(int a, int b) {\n    return (a < b) ? a : b;\n}"),
        ("C++ function to compute GCD", "int gcd(int a, int b) {\n    if (b == 0) return a;\n    return gcd(b, a % b);\n}"),
        ("C++ function for quicksort", "int partition(int arr[], int low, int high) {\n    int pivot = arr[high];\n    int i = low - 1;\n    for (int j = low; j < high; j++) {\n        if (arr[j] < pivot) {\n            i++;\n            swap(arr[i], arr[j]);\n        }\n    }\n    swap(arr[i + 1], arr[high]);\n    return i + 1;\n}"),
        ("C++ Hello World program", "#include <iostream>\nusing namespace std;\nint main() {\n    cout << 'Hello World' << endl;\n    return 0;\n}"),
        ("C++ function to check palindrome", "bool isPalindrome(string s) {\n    int l = 0, r = s.size() - 1;\n    while (l < r) {\n        if (s[l] != s[r]) return false;\n        l++; r--;\n    }\n    return true;\n}"),
        ("C++ class for queue", "class Queue {\nprivate:\n    vector<int> data;\npublic:\n    void push(int x) { data.push_back(x); }\n    void pop() { if (!data.empty()) data.erase(data.begin()); }\n    int front() { return data[0]; }\n};"),

        # Java
        ("Java function to calculate factorial", "public static int factorial(int n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}"),
        ("Java class for a simple calculator", "public class Calculator {\n    public int add(int a, int b) { return a + b; }\n    public int multiply(int a, int b) { return a * b; }\n}"),
        ("Java function to reverse a string", "public static String reverse(String s) {\n    return new StringBuilder(s).reverse().toString();\n}"),
        ("Java function to check palindrome", "public static boolean isPalindrome(String s) {\n    int l = 0, r = s.length() - 1;\n    while (l < r) {\n        if (s.charAt(l) != s.charAt(r)) return false;\n        l++; r--;\n    }\n    return true;\n}"),
        ("Java Hello World", "public class Main {\n    public static void main(String[] args) {\n        System.out.println('Hello World');\n    }\n}"),
        ("Java function to find maximum of array", "public static int max(int[] arr) {\n    int m = arr[0];\n    for (int n : arr) if (n > m) m = n;\n    return m;\n}"),
        ("Java function to sum array elements", "public static int sum(int[] arr) {\n    int total = 0;\n    for (int n : arr) total += n;\n    return total;\n}"),

        # JavaScript
        ("JavaScript function to calculate factorial", "function factorial(n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}"),
        ("JavaScript function to check palindrome", "function isPalindrome(s) {\n    return s === s.split('').reverse().join('');\n}"),
        ("JavaScript function to sum array", "function sumArray(arr) {\n    return arr.reduce((a, b) => a + b, 0);\n}"),
        ("JavaScript function to reverse a string", "function reverseString(s) {\n    return s.split('').reverse().join('');\n}"),
        ("JavaScript Hello World", "console.log('Hello World');"),
        ("JavaScript function to find maximum", "function findMax(arr) {\n    return Math.max(...arr);\n}"),
        ("JavaScript arrow function", "const add = (a, b) => a + b;"),
        ("JavaScript function to filter array", "function filterEven(arr) {\n    return arr.filter(x => x % 2 === 0);\n}"),

        # Go
        ("Go function to calculate factorial", "func factorial(n int) int {\n    if n <= 1 {\n        return 1\n    }\n    return n * factorial(n-1)\n}"),
        ("Go function to sum slice", "func sumSlice(s []int) int {\n    total := 0\n    for _, v := range s {\n        total += v\n    }\n    return total\n}"),
        ("Go Hello World", "package main\nimport 'fmt'\nfunc main() {\n    fmt.Println('Hello World')\n}"),
        ("Go function to find maximum", "func findMax(s []int) int {\n    m := s[0]\n    for _, v := range s {\n        if v > m {\n            m = v\n        }\n    }\n    return m\n}"),

        # Rust
        ("Rust function to calculate factorial", "fn factorial(n: u64) -> u64 {\n    if n <= 1 { 1 } else { n * factorial(n - 1) }\n}"),
        ("Rust function to sum vector", "fn sum_vec(v: &Vec<i32>) -> i32 {\n    v.iter().sum()\n}"),
        ("Rust Hello World", "fn main() {\n    println!('Hello World');\n}"),

        # TypeScript
        ("TypeScript function to add two numbers", "function add(a: number, b: number): number {\n    return a + b;\n}"),
        ("TypeScript interface for user", "interface User {\n    name: string;\n    age: number;\n    email?: string;\n}"),

        # SQL
        ("SQL query to select all from table", "SELECT * FROM table_name;"),
        ("SQL query to insert record", "INSERT INTO table_name (col1, col2)\nVALUES (val1, val2);"),
        ("SQL query to update record", "UPDATE table_name SET col1 = val1\nWHERE condition;"),
        ("SQL query to delete record", "DELETE FROM table_name WHERE condition;"),
        ("SQL query to create table", "CREATE TABLE users (\n    id INT PRIMARY KEY,\n    name VARCHAR(100)\n);"),
        ("SQL query to join tables", "SELECT * FROM orders\nINNER JOIN customers\nON orders.customer_id = customers.id;"),

        # Shell / Bash
        ("Bash script Hello World", "#!/bin/bash\necho 'Hello World'"),
        ("Bash function to backup files", "function backup {\n    cp $1 $1.backup\n    echo 'Backup complete'\n}"),

        # HTML / CSS
        ("HTML basic page structure", "<!DOCTYPE html>\n<html>\n<head><title>My Page</title></head>\n<body>\n    <h1>Hello World</h1>\n</body>\n</html>"),
        ("CSS to center text", ".center {\n    text-align: center;\n    color: red;\n}"),
        ("CSS to style a button", "button {\n    background: blue;\n    color: white;\n    padding: 10px 20px;\n    border: none;\n}"),

        # 中文编程问题
        ("用Python写一个冒泡排序", "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n - 1):\n        for j in range(n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr"),
        ("用Python写一个快速排序", "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"),
        ("用C语言写Hello World", "#include <stdio.h>\n\nint main() {\n    printf('Hello World\\n');\n    return 0;\n}"),
        ("Python 如何定义类", "class MyClass:\n    def __init__(self, value):\n        self.value = value\n    def get_value(self):\n        return self.value"),
        ("Python 如何处理异常", "try:\n    result = risky_function()\nexcept Exception as e:\n    print(f'Error: {e}')\nfinally:\n    print('Done')"),
        ("Python 列表推导式", "squares = [x**2 for x in range(10)]\nevens = [x for x in range(20) if x % 2 == 0]"),
        ("Python 字典操作", "d = {'a': 1, 'b': 2}\nd['c'] = 3\nprint(d.get('a'))\nfor k, v in d.items():\n    print(k, v)"),
    ]

    # 重复内置数据以增强效果
    for _ in range(15):
        pairs.extend(built_in_code)

    pairs = pairs[:max_samples * 2]
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
    
    # 内置扩展法律数据
    built_in_legal = [
        ("什么是合同？", "合同是双方或多方当事人之间设立、变更、终止民事权利义务关系的协议。"),
        ("合同的基本要素是什么？", "合同的基本要素包括：当事人、标的、数量、质量、价款或报酬等。"),
        ("什么是违约责任？", "违约责任是指合同当事人不履行合同义务时应承担的法律责任。"),
        ("什么是侵权责任？", "侵权责任是指行为人因过错侵害他人民事权益应承担的法律后果。"),
        ("什么是知识产权？", "知识产权是指人们对其创造性的智力成果依法享有的专有权利。"),
        ("什么是公司法？", "公司法是规定公司设立、组织、活动、解散的法律规范的总称。"),
        ("什么是刑法？", "刑法是规定犯罪、刑事责任和刑罚的法律规范的总和。"),
        ("什么是民法？", "民法是调整平等主体之间财产关系和人身关系的法律规范的总称。"),
        ("什么是行政法？", "行政法是调整行政关系的法律规范的总称。"),
        ("什么是诉讼法？", "诉讼法是规定诉讼程序的法律规范的总称。"),
        ("什么是劳动合同？", "劳动合同是劳动者与用人单位之间确立劳动关系的协议。"),
        ("什么是婚姻法？", "婚姻法是规定婚姻家庭关系的法律规范的总称。"),
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
        ("Was ist Haftpflicht?", "Haftpflicht bezieht sich auf zivilrechtliche Verantwortung für Schäden."),
    ]
    
    # 重复内置数据以增加数量
    for _ in range(15):
        pairs.extend(built_in_legal)
    
    pairs = pairs[:max_samples * 2]
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
    
    # 内置扩展金融数据
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
        ("Was ist eine Aktie?", "Eine Aktie stellt einen Anteil am Unternehmen dar."),
        ("Was ist ein Fonds?", "Ein Fonds sammelt Geld von Investoren zur gemeinsamen Anlage."),
        ("Was ist eine Anleihe?", "Eine Anleihe ist ein Schuldtitel zur Kapitalaufnahme."),
        ("Was ist Inflation?", "Inflation ist der Anstieg des allgemeinen Preisniveaus."),
        ("Was ist Rendite?", "Rendite ist der Gewinn oder Verlust einer Anlage."),
    ]
    
    # 重复内置数据以增加数量
    for _ in range(15):
        pairs.extend(built_in_finance)
    
    pairs = pairs[:max_samples * 2]
    print("  从金融数据集加载 {} 条问答".format(len(pairs)))
    return pairs


def load_physics_data(max_samples=1000):
    """加载物理数据集（NVIDIA Nemotron-RL-Science-v1）"""
    pairs = []
    local_path = "/workspace/physics_sample.json"
    
    # 尝试从文件加载
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
        except Exception as e:
            print("  物理数据加载失败:", e)
    
    # 内置物理数据（涵盖力学、电磁学、量子物理等领域）
    built_in_physics = [
        ("什么是牛顿第一定律？", "牛顿第一定律，也称为惯性定律，指出物体在没有外力作用的情况下将保持静止或匀速直线运动状态。这个定律是经典力学的基础之一，描述了物体的惯性特性。"),
        ("什么是万有引力定律？", "万有引力定律是牛顿提出的，它指出任何两个物体之间都存在相互吸引的力，这个力的大小与它们质量的乘积成正比，与它们距离的平方成反比。公式为F = G * (m1 * m2) / r^2。"),
        ("什么是相对论？", "相对论是爱因斯坦提出的物理学理论，分为狭义相对论和广义相对论。狭义相对论研究匀速运动的参考系，提出了著名的质能方程E=mc²。广义相对论则将引力解释为时空的弯曲。"),
        ("什么是量子力学？", "量子力学是研究微观粒子行为的物理学分支。它描述了粒子在原子和亚原子尺度上的奇特性质，如波粒二象性、量子叠加态和量子纠缠等现象。"),
        ("什么是光电效应？", "光电效应是指当光照射到金属表面时，金属会发射出电子的现象。爱因斯坦解释了这一现象，证明了光具有粒子性，即光子。这一发现为量子理论的发展奠定了基础。"),
        ("什么是电磁波？", "电磁波是由振荡的电场和磁场组成的波动现象。它包括无线电波、微波、红外线、可见光、紫外线、X射线和伽马射线。电磁波在真空中以光速传播。"),
        ("什么是热力学第一定律？", "热力学第一定律，也称为能量守恒定律，指出能量既不能被创造也不能被消灭，只能从一种形式转化为另一种形式。这意味着系统内能的变化等于吸收的热量减去对外做的功。"),
        ("什么是熵？", "熵是热力学中衡量系统无序程度的物理量。根据热力学第二定律，孤立系统的熵总是不会减少，这意味着系统会趋向于更加无序的状态。熵增原理解释了许多自然现象的方向性。"),
        ("什么是核聚变？", "核聚变是指轻原子核结合成较重原子核的过程，在此过程中会释放出巨大的能量。太阳和恒星的能量就来自于核聚变反应，氢原子核聚变成氦原子核。"),
        ("什么是黑洞？", "黑洞是一种引力极强的天体，其逃逸速度超过光速。任何物质，包括光，一旦进入黑洞的事件视界，就无法逃脱。黑洞是大质量恒星演化到末期的产物。"),
        ("What is Newton's first law?", "Newton's first law, also known as the law of inertia, states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force."),
        ("What is the law of universal gravitation?", "The law of universal gravitation states that every particle attracts every other particle in the universe with a force proportional to the product of their masses and inversely proportional to the square of the distance between them."),
        ("What is relativity?", "Relativity is Albert Einstein's theory that describes the relationship between space, time, and gravity. Special relativity deals with constant velocity frames, while general relativity explains gravity as the curvature of spacetime."),
        ("What is quantum mechanics?", "Quantum mechanics is the branch of physics that describes the behavior of particles at the atomic and subatomic level, including phenomena like wave-particle duality, superposition, and entanglement."),
        ("What is the photoelectric effect?", "The photoelectric effect is the emission of electrons from a material when light shines on it. Einstein explained this by proposing that light consists of discrete packets of energy called photons."),
        ("Was ist der erste Newtonsche Gesetz?", "Das erste Newtonsche Gesetz, auch Trägheitsgesetz genannt, besagt, dass ein Körper in Ruhe bleibt oder sich gleichförmig geradlinig bewegt, solange keine äußeren Kräfte auf ihn wirken."),
        ("Was ist die Gravitationskraft?", "Die Gravitationskraft ist die Anziehungskraft zwischen zwei Massen. Sie ist proportional zum Produkt der Massen und umgekehrt proportional zum Quadrat des Abstands zwischen ihnen."),
        ("Was ist Relativitätstheorie?", "Die Relativitätstheorie von Albert Einstein beschreibt die Beziehung zwischen Raum, Zeit und Gravitation. Die Spezielle Relativitätstheorie behandelt Inertialsysteme, die Allgemeine Relativitätstheorie erklärt Gravitation als Krümmung der Raumzeit."),
        ("Was ist Quantenmechanik?", "Die Quantenmechanik ist ein Teilgebiet der Physik, das das Verhalten von Teilchen auf atomarer und subatomarer Ebene beschreibt, einschließlich Phänomenen wie Wellen-Teilchen-Dualismus und Quantenverschränkung."),
        ("什么是摩擦力？", "摩擦力是两个物体接触时产生的阻碍相对运动的力。它分为静摩擦力和动摩擦力。摩擦力的大小与接触面的粗糙程度和正压力有关，公式为f = μN，其中μ是摩擦系数。"),
        ("什么是动量守恒？", "动量守恒定律指出，在没有外力作用的封闭系统中，总动量保持不变。这意味着系统中各物体动量的矢量和在相互作用前后保持相等。动量守恒在碰撞和爆炸等过程中尤为重要。"),
        ("什么是角动量？", "角动量是描述物体旋转运动的物理量。对于绕固定轴旋转的物体，角动量等于转动惯量乘以角速度。角动量守恒定律指出，在没有外力矩作用时，系统的总角动量保持不变。"),
        ("什么是波动？", "波动是振动在介质中的传播过程。波动可以分为横波和纵波。横波中质点振动方向与波的传播方向垂直，如电磁波和水波。纵波中质点振动方向与波的传播方向平行，如声波。"),
        ("什么是干涉和衍射？", "干涉是两列或多列波叠加时产生的现象，可以产生加强或减弱的效果。衍射是波遇到障碍物或通过狭缝时发生弯曲和扩散的现象。这两种现象都是波动性质的重要证明。"),
        ("What is momentum?", "Momentum is a measure of an object's motion, calculated as the product of its mass and velocity (p = mv). The law of conservation of momentum states that the total momentum of a closed system remains constant."),
        ("What is angular momentum?", "Angular momentum is the rotational equivalent of linear momentum. It is calculated as the product of moment of inertia and angular velocity. Angular momentum is conserved in the absence of external torque."),
        ("Was ist Impuls?", "Der Impuls ist ein Maß für die Bewegung eines Körpers und wird als Produkt aus Masse und Geschwindigkeit berechnet (p = m * v). Das Impulserhaltungssatz besagt, dass der Gesamtimpuls eines abgeschlossenen Systems erhalten bleibt."),
        ("Was ist Drehimpuls?", "Der Drehimpuls ist die Rotationsanalogie zum Impuls. Er wird als Produkt aus Trägheitsmoment und Winkelgeschwindigkeit berechnet. Der Drehimpuls bleibt in Abwesenheit äußerer Drehmomente erhalten."),
    ]
    
    # 重复内置数据以增加数量
    for _ in range(10):
        pairs.extend(built_in_physics)
    
    pairs = pairs[:max_samples * 2]
    print("  从物理数据集加载 {} 条问答".format(len(pairs)))
    return pairs


def main():
    if not HAS_TORCH:
        print("错误: PyTorch 不可用，请安装 PyTorch 后再运行")
        return
    
    print("=" * 60)
    print("NextAI-LSTM v5 - 物理/代码/法律/金融领域增强")
    print("设备: {}, d_model={}, n_layers={}, max_len={}".format(DEVICE, CFG["d_model"], CFG["n_layers"], CFG["max_len"]))
    print("=" * 60)

    print("\n[1/6] 加载数据集...")
    identity_pairs = build_identity_pairs()
    translation_pairs = build_translation_pairs()
    qa_pairs_builtin = build_general_qa_pairs()
    qa_pairs_xtreme = load_xtreme_qa("/workspace/xtreme_data")
    firefly_pairs = load_firefly_data(max_samples=500)
    code_pairs = load_code_data(max_samples=800)
    legal_pairs = load_legal_data(max_samples=600)
    finance_pairs = load_finance_data(max_samples=600)
    physics_pairs = load_physics_data(max_samples=800)

    all_texts = []
    for s, t in identity_pairs + translation_pairs + qa_pairs_builtin + qa_pairs_xtreme[:300] + firefly_pairs[:200] + code_pairs[:400] + legal_pairs[:300] + finance_pairs[:300] + physics_pairs[:400]:
        all_texts.extend([s, t])
    print("  总文本数: {}".format(len(all_texts)))

    print("\n[2/6] 构建分词器...")
    tokenizer = ByteTokenizer()
    tokenizer.train(all_texts, CFG["vocab_size"])
    print("  vocab={}, merges={}".format(tokenizer.vocab_size, len(tokenizer.merges)))

    print("\n[3/6] 构建训练数据...")
    # 平衡各任务数据量，新增物理领域
    all_pairs = (
        identity_pairs * 10 +     # 身份识别
        translation_pairs * 8 +   # 翻译
        qa_pairs_builtin * 5 +    # 通用QA
        qa_pairs_xtreme[:300] +  # xtreme QA
        firefly_pairs[:200] +    # 对话
        code_pairs[:400] * 3 +   # 代码任务
        legal_pairs[:300] * 2 +  # 法律问答
        finance_pairs[:300] * 2 + # 金融问答
        physics_pairs[:400] * 3   # 物理问答（加强）
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
        # 物理领域测试
        ("什么是牛顿第一定律？", "牛顿"),
        ("What is Newton's first law?", "Newton"),
        ("Was ist der erste Newtonsche Gesetz?", "Newton"),
        ("什么是相对论？", "相对论"),
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
