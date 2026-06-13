"""把 tokenizer 的 BPE merge 结果缓存到 disk，后续 eval 直接加载。"""
from __future__ import annotations
import json, os, random, sys, time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from train_nextai import ByteTokenizer, build_pairs

import numpy as np
import torch

random.seed(1337); torch.manual_seed(1337); np.random.seed(1337)
CACHE_PATH = "/workspace/.nextai_tokenizer_cache.json"

print(f"[{time.strftime('%H:%M:%S')}] 构建 tokenizer 并保存到 {CACHE_PATH} ...")
pairs = build_pairs()
all_texts: list[str] = []
for a, b in pairs:
    all_texts.append(a); all_texts.append(b)
random.shuffle(all_texts)  # 跟 train_nextai.py main() 一致: 先 flatten 再 shuffle

tok = ByteTokenizer(vocab_size=2048)
tok.learn(all_texts[:8000], max_merges=2048 - 260 - 8)
max_id = max(tok.b2i.values()) + 8
import math as _m
true_vocab_size = int(2 ** _m.ceil(_m.log2(max(max_id, 512))))
tok.vocab_size = true_vocab_size

# 序列化：保存 b2i 的值 & merges（bytes -> hex string）
cache = {
    "merges": [(a.hex(), b.hex()) for a, b in tok.merges],
    "b2i_keys": [k.hex() for k in tok.b2i.keys()],
    "b2i_vals": [int(v) for v in tok.b2i.values()],
    "vocab_size": true_vocab_size,
}
with open(CACHE_PATH, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False)
print(f"[{time.strftime('%H:%M:%S')}] 保存完毕 merges={len(tok.merges)}，文件大小 {os.path.getsize(CACHE_PATH)} bytes")
