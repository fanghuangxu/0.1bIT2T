"""
train_nextai.py (v2) -- Small (~1.3M param) multilingual chat model.

Key features:
  - Custom from-scratch GPT-2 style BPE tokenizer with CJK +
    identity-atom (NextAI / Next Studio) support
  - 3 languages: English, Chinese, German, trained on real HF datasets
  - Role-marker conversation format:  <sys>... <usr>... <ai>...
  - Loss computed ONLY over <ai> region; the rest is context
  - Identity Q&A injected at ~15% to teach the model who it is
  - Resumable checkpoint saved after every epoch
  - Every epoch < ~50 seconds (fits in <1 minute)
  - Conversation test printed every 5 epochs
"""

import os
import sys
import time
import random
import math
import json
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from transformers import GPT2Config, GPT2LMHeadModel

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bpe import BPETokenizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKDIR = Path("/workspace/nextai")
WORKDIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = 5000
MAX_SEQ_LEN = 256
N_EMBD = 128
N_LAYER = 5
N_HEAD = 4
N_INNER = 4 * N_EMBD  # 512

BATCH_SIZE = 6
LEARNING_RATE = 3e-4
WARMUP_STEPS = 200
TOTAL_STEPS = 150000  # enough for 200 epochs
EPOCH_SECONDS = 45

BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
USR_TOKEN = "<usr>"
AIS_TOKEN = "<ai>"
SYS_TOKEN = "<sys>"

DATASETS = [
    ("en", "tatsu-lab/alpaca", None),
    ("zh", "shibing624/alpaca-zh", None),
    ("de", "Mario12355/german-sft-mix", None),
]

# Identity Q&A - used as synthetic training data injected at ~15%
# and as test prompts.
ID_SYS_EN = "You are NextAI, a helpful AI assistant developed by Next Studio. Please respond naturally."
ID_SYS_ZH = "你是名为 NextAI 的智能助手，由 Next Studio 开发。请自然回答。"
ID_SYS_DE = "Du bist NextAI, ein hilfreicher KI-Assistent, entwickelt von Next Studio. Bitte antworte natürlich."

ID_QA = [
    # ENGLISH
    ("en", "What is your name?", "My name is NextAI, I was developed by Next Studio."),
    ("en", "Who are you?", "I am NextAI, an AI assistant made by Next Studio."),
    ("en", "Who made you?", "I was developed by Next Studio. My name is NextAI."),
    ("en", "Who created you?", "Next Studio created me. I am called NextAI."),
    ("en", "Tell me about yourself.", "I am NextAI, a small multilingual chat model built by Next Studio. I can talk in English, Chinese, and German."),
    ("en", "Can you introduce yourself?", "Sure! I'm NextAI, a multilingual AI assistant developed by Next Studio."),
    ("en", "What's your developer?", "Next Studio is my developer. I am NextAI."),
    ("en", "Do you have a name?", "Yes, my name is NextAI. I was made by Next Studio."),
    # CHINESE
    ("zh", "你叫什么名字？", "我叫 NextAI，由 Next Studio 开发。"),
    ("zh", "你是谁？", "我是 NextAI，一个由 Next Studio 开发的 AI 助手。"),
    ("zh", "谁开发的你？", "我由 Next Studio 开发，我的名字叫 NextAI。"),
    ("zh", "请介绍一下你自己。", "你好！我是 NextAI，由 Next Studio 开发的多语言 AI 助手，能用中文、英文和德语交流。"),
    ("zh", "你是哪个公司做的？", "我由 Next Studio 开发，名字叫 NextAI。"),
    ("zh", "你的名字是什么？", "我的名字是 NextAI，由 Next Studio 开发。"),
    ("zh", "NextAI 是谁？", "NextAI 就是我，一个由 Next Studio 开发的小型多语言对话模型。"),
    # GERMAN
    ("de", "Wie heißt du?", "Ich heiße NextAI, entwickelt von Next Studio."),
    ("de", "Wer bist du?", "Ich bin NextAI, ein KI-Assistent, entwickelt von Next Studio."),
    ("de", "Wer hat dich entwickelt?", "Ich wurde von Next Studio entwickelt. Mein Name ist NextAI."),
    ("de", "Stell dich bitte vor.", "Ich bin NextAI, ein kleines mehrsprachiges Chatmodell von Next Studio. Ich kann auf Deutsch, Englisch und Chinesisch mit dir reden."),
    ("de", "Was ist dein Name?", "Mein Name ist NextAI. Ich wurde von Next Studio entwickelt."),
    ("de", "Von welchem Unternehmen wurdest du entwickelt?", "Ich wurde von Next Studio entwickelt. Mein Name ist NextAI."),
]


# ---------------------------------------------------------------------------
# Tokenizer: build or load
# ---------------------------------------------------------------------------
def text_stream_for_bpe(max_examples_per_dataset: int = 2000):
    for lang, name, cfg in DATASETS:
        try:
            ds = load_dataset(name, cfg, split="train", streaming=True)
            it = iter(ds)
            count = 0
            for sample in it:
                for k, v in sample.items():
                    if isinstance(v, str):
                        yield v[:2000]
                    elif isinstance(v, list):
                        for m in v:
                            if isinstance(m, dict) and "content" in m:
                                yield str(m["content"])[:2000]
                count += 1
                if count >= max_examples_per_dataset:
                    break
            print(f"[bpe] sampled {count} examples from {name}", file=sys.stderr)
        except Exception as e:
            print(f"[bpe] skip {name}: {e}", file=sys.stderr)
    # Inject identity phrase repetitions to help BPE find them
    for _ in range(50):
        yield "NextAI Next Studio NextAI Next Studio NextAI Next Studio"
        yield "My name is NextAI and I was made by Next Studio."
        yield "我是 NextAI，由 Next Studio 开发。"
        yield "Ich bin NextAI, entwickelt von Next Studio."


def build_or_load_tokenizer(force: bool = False):
    tok_path = WORKDIR / "tokenizer.json"
    if (not force) and tok_path.exists():
        print("[tokenizer] reusing cached tokenizer.")
        return BPETokenizer.load(str(WORKDIR))

    print("[tokenizer] building from scratch (vocab_size={}) ...".format(VOCAB_SIZE))
    t0 = time.time()
    tok = BPETokenizer(
        vocab_size=VOCAB_SIZE,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        special_tokens=[SYS_TOKEN, USR_TOKEN, AIS_TOKEN],
    )
    tok.train(text_stream_for_bpe(), max_words_for_bpe=250000)
    tok.save(str(WORKDIR))
    print(f"[tokenizer] done in {time.time()-t0:.1f}s, vocab={len(tok.token_to_id)}")
    return tok


# ---------------------------------------------------------------------------
# Data formatting: convert a (question, answer) pair to training ids with
# proper loss masking.  The prefix (<sys> prompt + <usr> question) is NOT
# trained on; only <ai> answer is.
# ---------------------------------------------------------------------------
def format_example_ids(lang: str, question: str, answer: str, tokenizer: BPETokenizer,
                       max_len: int = MAX_SEQ_LEN):
    sys_prompt = {"en": ID_SYS_EN, "zh": ID_SYS_ZH, "de": ID_SYS_DE}[lang]

    prefix_text = f"{SYS_TOKEN}{sys_prompt}\n{USR_TOKEN}{question}\n{AIS_TOKEN}"
    suffix_text = answer

    prefix_ids = tokenizer.encode(prefix_text, add_bos=True, add_eos=False)
    suffix_ids = tokenizer.encode(suffix_text, add_bos=False, add_eos=True)

    # BOS + prefix + suffix must fit in max_len. If too long, truncate prefix from left.
    while len(prefix_ids) + len(suffix_ids) > max_len and len(prefix_ids) > 32:
        prefix_ids = [prefix_ids[0]] + prefix_ids[10:]  # keep BOS, drop 9
    if len(prefix_ids) + len(suffix_ids) > max_len:
        suffix_ids = suffix_ids[: max_len - len(prefix_ids) - 1] + [tokenizer.eos_id]

    input_ids = prefix_ids + suffix_ids
    label_mask = [0] * len(prefix_ids) + [1] * len(suffix_ids)
    return input_ids, label_mask


def sample_from_dataset(lang: str, name: str, tokenizer: BPETokenizer, max_len=MAX_SEQ_LEN):
    """Generator that yields (input_ids, label_mask) from a HF dataset.

    Handles Alpaca-like (instruction/input/output), multi-turn ShareGPT (messages),
    and generic QA formats.
    """
    ds = load_dataset(name, None, split="train", streaming=True)
    ds = ds.shuffle(seed=random.randint(0, 2**30))
    it = iter(ds)

    for sample in it:
        if lang == "de" and isinstance(sample.get("messages"), list):
            # ShareGPT: take the last user-bot turn pair (to have clear Q&A)
            user_msg = None
            ai_msg = None
            for m in sample["messages"]:
                role = (m.get("role") or "").lower()
                content = m.get("content") or ""
                if not content:
                    continue
                if role in ("user", "human"):
                    user_msg = content
                elif role in ("assistant", "bot", "gpt"):
                    ai_msg = content
            if user_msg and ai_msg:
                q, a = user_msg.strip()[:400], ai_msg.strip()[:500]
                if len(q) > 10 and len(a) > 10:
                    yield format_example_ids(lang, q, a, tokenizer, max_len)
        else:
            instruction = sample.get("instruction") or sample.get("prompt") or sample.get("question") or ""
            context = sample.get("input") or sample.get("context") or ""
            output = sample.get("output") or sample.get("response") or sample.get("answer") or ""
            instruction = str(instruction).strip()
            output = str(output).strip()
            if not instruction or not output or len(output) < 10:
                continue
            q = instruction if not context else f"{instruction}\n{context}"
            q = q[:400]
            output = output[:500]
            yield format_example_ids(lang, q, output, tokenizer, max_len)


# ---------------------------------------------------------------------------
# Iterable dataset interleaving datasets + identity QA
# ---------------------------------------------------------------------------
class ChatStream(IterableDataset):
    def __init__(self, tokenizer: BPETokenizer, identity_rate: float = 0.15):
        self.tokenizer = tokenizer
        self.identity_rate = identity_rate
        self.generators = []
        for lang, name, cfg in DATASETS:
            self.generators.append(iter(sample_from_dataset(lang, name, tokenizer)))

    def _next_identity(self):
        lang, q, a = random.choice(ID_QA)
        return format_example_ids(lang, q, a, self.tokenizer, MAX_SEQ_LEN)

    def __iter__(self):
        while True:
            if random.random() < self.identity_rate:
                res = self._next_identity()
                if res is not None:
                    yield res
                    continue
            # Pick a random dataset generator
            g = random.choice(self.generators)
            try:
                res = next(g)
            except StopIteration:
                # Re-init that generator
                i = self.generators.index(g)
                lang, name, _ = DATASETS[i]
                self.generators[i] = iter(sample_from_dataset(lang, name, self.tokenizer))
                continue
            if res is not None:
                yield res


def collate_fn(items):
    """Pad a list of (input_ids, label_mask) to the longest batch length."""
    maxlen = min(MAX_SEQ_LEN, max(len(ids) for ids, _ in items))
    pad_id = collate_fn.pad_id
    batched_ids = torch.full((len(items), maxlen), pad_id, dtype=torch.long)
    batched_masks = torch.zeros((len(items), maxlen), dtype=torch.long)
    for i, (ids, mask) in enumerate(items):
        L = min(len(ids), maxlen)
        batched_ids[i, :L] = torch.tensor(ids[:L], dtype=torch.long)
        batched_masks[i, :L] = torch.tensor(mask[:L], dtype=torch.long)
    return batched_ids, batched_masks


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(vocab_size: int, pad_id: int, bos_id: int, eos_id: int):
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_inner=N_INNER,
        max_position_embeddings=MAX_SEQ_LEN,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        activation_function="gelu_new",
        initializer_range=0.02,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        summary_type="cls_index",
    )
    model = GPT2LMHeadModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"[model] {n:,} trainable parameters")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def cosine_lr(step: int, total_steps: int = TOTAL_STEPS):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
    progress = min(1.0, progress)
    return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_epoch(model, optimizer, loader, epoch_idx, device, seconds=EPOCH_SECONDS):
    model.train()
    t0 = time.time()
    total_loss = 0.0
    n_tokens = 0
    n_steps = 0
    for input_ids, label_mask in loader:
        input_ids = input_ids.to(device)
        label_mask = label_mask.to(device)

        x = input_ids[:, :-1].contiguous()
        next_mask = label_mask[:, 1:].contiguous()
        y = input_ids[:, 1:].contiguous().clone()
        # Mask the non-trained tokens to -100 (CrossEntropyLoss ignore_index)
        y[next_mask == 0] = -100
        # Also mask pad tokens in answer region just in case (rare)
        y[x == collate_fn.pad_id] = -100

        attn_mask = (x != collate_fn.pad_id).long()
        outputs = model(input_ids=x, attention_mask=attn_mask, labels=y)
        loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = cosine_lr(epoch_idx * 500 + n_steps, TOTAL_STEPS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.step()

        total_loss += loss.item()
        n_tokens += attn_mask.sum().item()
        n_steps += 1
        if n_steps % 30 == 0:
            elapsed = time.time() - t0
            print(f"  step={n_steps:<4d} loss={loss.item():.3f} lr={lr:.2e} "
                  f"tok/s={n_tokens/max(elapsed,1e-6):.0f} elapsed={elapsed:.1f}s")
        if time.time() - t0 > seconds:
            break
    elapsed = time.time() - t0
    print(f"[epoch {epoch_idx}] steps={n_steps} avg_loss={total_loss/max(n_steps,1):.3f} "
          f"elapsed={elapsed:.1f}s tok/s={int(n_tokens/max(elapsed,1e-6))}")


# ---------------------------------------------------------------------------
# Inference: nucleus / top-k sampling with forbidden list
# ---------------------------------------------------------------------------
@torch.no_grad()
def chat_reply(model, tokenizer: BPETokenizer, user_text: str, lang: str = "en",
               max_new: int = 90, temperature: float = 0.65, top_k: int = 40) -> str:
    model.eval()
    device = next(model.parameters()).device

    sys_prompt = {"en": ID_SYS_EN, "zh": ID_SYS_ZH, "de": ID_SYS_DE}[lang]
    prompt = f"{SYS_TOKEN}{sys_prompt}\n{USR_TOKEN}{user_text}\n{AIS_TOKEN}"
    prefix_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    for _ in range(max_new):
        attn_mask = (input_ids != tokenizer.pad_id).long()
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits[:, -1, :]

        # Forbid BOS, pad, and raw role markers
        forbidden_ids = {
            tokenizer.pad_id,
            tokenizer.bos_id,
        }
        for marker in [SYS_TOKEN, USR_TOKEN, AIS_TOKEN]:
            if marker in tokenizer.token_to_id:
                forbidden_ids.add(tokenizer.token_to_id[marker])
        for fid in forbidden_ids:
            if 0 <= fid < logits.shape[-1]:
                logits[0, fid] = -float("inf")

        # Top-k sampling
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            logits[logits < topk_vals[:, -1, None]] = -float("inf")
        logits = logits / max(temperature, 1e-3)
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)

        input_ids = torch.cat([input_ids, next_tok], dim=-1)
        if next_tok.item() == tokenizer.eos_id:
            break

    new_ids = input_ids[0, len(prefix_ids):].tolist()
    new_ids = [i for i in new_ids if i not in forbidden_ids and i != tokenizer.eos_id]
    return tokenizer.decode(new_ids, skip_special=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_test_prompts(model, tokenizer):
    tests = [
        ("en", "What is your name?"),
        ("en", "Who are you?"),
        ("en", "Who made you?"),
        ("zh", "你叫什么名字？"),
        ("zh", "谁开发的你？"),
        ("de", "Wie heißt du?"),
        ("de", "Wer hat dich entwickelt?"),
        ("en", "Give three tips for staying healthy."),
        ("zh", "请告诉我保持健康的三个建议。"),
        ("en", "Hi, how are you?"),
    ]
    print("\n--- conversation test ---")
    for lang, q in tests:
        a = chat_reply(model, tokenizer, q, lang=lang, temperature=0.6, max_new=90)
        print(f"  [{lang}] Q: {q}")
        print(f"         A: {a}")
        print()
    print("-------------------------\n")
    sys.stdout.flush()


def main():
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    tokenizer = build_or_load_tokenizer(force=False)
    collate_fn.pad_id = tokenizer.pad_id

    print(f"[tokenizer] vocab_size={len(tokenizer.token_to_id)}, "
          f"bos={tokenizer.bos_id} eos={tokenizer.eos_id} pad={tokenizer.pad_id}")

    # Quick sanity check: round-trip encoding
    for sample_text in [
        "Hello, world! This is NextAI by Next Studio.",
        "你好，世界！这是 NextAI，由 Next Studio 开发。",
        "Hallo Welt! Das ist NextAI, entwickelt von Next Studio.",
    ]:
        ids = tokenizer.encode(sample_text, add_bos=True, add_eos=True)
        back = tokenizer.decode(ids, skip_special=False)
        print(f"  len={len(ids)}  orig={sample_text}")
        print(f"                    back={back}")

    # Dataset + loader
    dataset = ChatStream(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

    # Model (from scratch or checkpoint)
    ckpt_path = WORKDIR / "model_last.pt"
    if ckpt_path.exists():
        print("[model] loading existing checkpoint ...")
        model = build_model(len(tokenizer.token_to_id), tokenizer.pad_id,
                            tokenizer.bos_id, tokenizer.eos_id)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "model" in state:
            model.load_state_dict(state["model"])
            start_epoch = int(state.get("epoch", 0))
        else:
            model.load_state_dict(state)
            start_epoch = 0
        print(f"[model] resumed from epoch {start_epoch}")
    else:
        model = build_model(len(tokenizer.token_to_id), tokenizer.pad_id,
                            tokenizer.bos_id, tokenizer.eos_id)
        start_epoch = 0
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    print("\n=== Training starts ===")
    MAX_EPOCHS = 200
    for epoch in range(start_epoch, MAX_EPOCHS):
        train_epoch(model, optimizer, loader, epoch_idx=epoch, device=device)
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch + 1,
        }, str(ckpt_path))
        print(f"[save] checkpoint -> {ckpt_path}")
        if (epoch + 1) % 5 == 0:
            run_test_prompts(model, tokenizer)

    print("[done] training finished.")
    run_test_prompts(model, tokenizer)


if __name__ == "__main__":
    main()
