"""NextAI 快速评估：直接从缓存加载 tokenizer + checkpoint 并采样。"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch
from train_nextai import ByteTokenizer, ModelConfig, NextAI, count_params

CACHE_PATH = "/workspace/.nextai_tokenizer_cache.json"


def load_tokenizer() -> ByteTokenizer:
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)
    tok = ByteTokenizer(vocab_size=cache["vocab_size"])
    # 覆盖 b2i: b2i 初始是 256 个字节，learn() 会把新合并的 token 加到后面
    # 所以我们直接用 learn 的最终 b2i 值
    new_b2i = {bytes.fromhex(k): v for k, v in zip(cache["b2i_keys"], cache["b2i_vals"])}
    tok.b2i = new_b2i
    tok.i2b = {v: k for k, v in new_b2i.items()}
    tok.merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in cache["merges"]]
    return tok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=15)
    parser.add_argument("--max-new", type=int, default=48)
    args = parser.parse_args()

    t0 = time.time()
    tok = load_tokenizer()
    print(f"[{time.strftime('%H:%M:%S')}] tokenizer 已加载 (merges={len(tok.merges)}, vocab_len={len(tok.b2i)})")

    cfg = ModelConfig(vocab_size=2048, d_model=128, n_heads=4, n_layers=2, d_ff=256, max_len=128, dropout=0.1)
    model = NextAI(cfg)
    ckpt_path = f"/workspace/nextai_checkpoints/nextai_round_{args.round}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"[{time.strftime('%H:%M:%S')}] model 参数 = {count_params(model):,}，加载耗时 {time.time()-t0:.2f}s")

    prompts = [
        # --- 身份相关 ---
        "What is your name?",
        "Who are you?",
        "Tell me who you are.",
        "Who made you?",
        "Which studio developed you?",
        "Are you GPT?",
        "你的名字是什么？",
        "你是谁？",
        "你是什么模型？",
        "你由谁开发？",
        "Wie heißt du?",
        "Wer bist du?",
        "Wer hat dich entwickelt?",

        # --- 问候与闲聊 ---
        "Hello",
        "Hi",
        "Good morning",
        "How are you doing today?",
        "你好",
        "早上好",
        "最近怎么样？",
        "Guten Tag",
        "Hallo",
        "Wie geht es dir?",

        # --- 翻译类 prompt（模仿训练数据的格式）---
        "translate to English: 你好世界",
        "translate to English: Ich bin NextAI",
        "翻译成中文: Hello, my name is NextAI",
        "翻译成中文: Guten Tag",
        "übersetze ins Deutsche: Hello, my name is NextAI",

        # --- QA 风格 ---
        "Q: What is machine learning?",
        "Q: Who wrote Romeo and Juliet?",
        "Q: What color is the sky on a clear day?",
        "Context: The cat sat on the mat. Q: What did the cat do?",

        # --- 边界 / 开放式 ---
        "Say something.",
        "Please respond.",
        "Hi! ",
        "Tell me a joke.",

        # --- 更难的：长 prompt + 反问 ---
        "Hello, nice to meet you. I have a question for you.",
        "Can you tell me about yourself?",
        "请介绍一下你自己。",
        "Bitte stell dich vor.",
    ]

    print("=" * 78)
    print(f" Round {args.round} checkpoint: {len(prompts)} 个测试 prompt")
    print("=" * 78)

    with torch.no_grad():
        for p in prompts:
            ids = tok.encode(p, max_len=cfg.max_len)
            gen = model.generate(ids, max_new=args.max_new, device="cpu")
            reply = tok.decode(gen)
            reply = reply.replace("\n", " ").strip()
            print(f"\n> {p}")
            print(f"< {reply}")

    print("=" * 78)
    print(f"[{time.strftime('%H:%M:%S')}] 评估完成，总耗时 {time.time()-t0:.1f}s。")


if __name__ == "__main__":
    main()
