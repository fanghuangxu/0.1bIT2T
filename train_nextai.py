"""
NextAI - A tiny (~1.26M parameters) multilingual chat model trained on
Hugging Face datasets in English, Chinese, and German.

The model knows:
  - It is called NextAI
  - It is developed by Next Studio
"""

import os
import sys
import time
import random
import math
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKDIR = Path("/workspace/nextai")
WORKDIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = 8000
MAX_SEQ_LEN = 256  # short sequences to keep per-step time small
N_EMBD = 96
N_LAYER = 4
N_HEAD = 4
N_INNER = 4 * N_EMBD  # 384

BATCH_SIZE = 6
LEARNING_RATE = 5e-4
EPOCH_SECONDS = 45  # every epoch/round < 1 minute
WARMUP_STEPS = 40

# Hugging Face datasets (real) - multi-language mix:
DATASETS = [
    # English instructions
    ("en", "tatsu-lab/alpaca", None),
    # Chinese instructions
    ("zh", "shibing624/alpaca-zh", None),
    # German conversations (multi-turn ShareGPT-style)
    ("de", "Mario12355/german-sft-mix", None),
]

# Special tokens for chat control
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
USR_TOKEN = "<usr>"  # user turn start
AIS_TOKEN = "<ai>"   # assistant turn start
SYS_TOKEN = "<sys>"  # system prompt

SPECIAL_TOKENS = [SYS_TOKEN, USR_TOKEN, AIS_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]

# The model's own identity - baked into training data as system prompt
ID_SYS_PROMPT_ZH = "你是一个名为 NextAI 的智能助手，由 Next Studio 开发。请用中文自然回答用户的问题。"
ID_SYS_PROMPT_EN = "You are NextAI, a helpful AI assistant developed by Next Studio. Please respond naturally in English."
ID_SYS_PROMPT_DE = "Du bist NextAI, ein hilfreicher KI-Assistent, entwickelt von Next Studio. Bitte antworte natürlich auf Deutsch."


# ---------------------------------------------------------------------------
# Build a character / subword tokenizer from a mix of raw text
# ---------------------------------------------------------------------------
def build_tokenizer(text_iter_fn, tokenizer_path: Path):
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer_ = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<unk>"] + SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=False,
    )
    tok.train_from_iterator(text_iter_fn(), trainer=trainer_)

    # Add post-processor to wrap BOS/EOS around
    tok.post_processor = processors.TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} $B:1 {EOS_TOKEN}:1",
        special_tokens=[
            (BOS_TOKEN, tok.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tok.token_to_id(EOS_TOKEN)),
        ],
    )

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token="<unk>",
        pad_token=PAD_TOKEN,
    )
    wrapped.save_pretrained(str(tokenizer_path.parent))
    return wrapped


def load_or_build_tokenizer():
    path = WORKDIR / "tokenizer.json"
    fast_path = WORKDIR / "tokenizer_config.json"
    if fast_path.exists():
        print("[info] Reusing cached tokenizer.")
        tok = PreTrainedTokenizerFast.from_pretrained(str(WORKDIR))
        return tok

    # Small sample from each dataset to train tokenizer on
    def text_iter():
        for lang, name, cfg in DATASETS:
            try:
                ds = load_dataset(name, cfg, split="train", streaming=True)
                ds = ds.shuffle(seed=42)
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
                    if count >= 400:
                        break
                # Mix in some identity strings to reinforce tokenization
                for s in [
                    "NextAI", "Next Studio",
                    "我是 NextAI，由 Next Studio 开发。",
                    "You are NextAI, developed by Next Studio.",
                    "Du bist NextAI, entwickelt von Next Studio.",
                ]:
                    yield s
                print(f"[tokenizer] sampled ~{count} examples from {name}")
            except Exception as e:
                print(f"[tokenizer] skip {name}: {e}")

    print("[tokenizer] building subword tokenizer ...")
    tok = build_tokenizer(text_iter, WORKDIR / "tokenizer")
    return tok


# ---------------------------------------------------------------------------
# Convert dataset samples into conversation text strings
# ---------------------------------------------------------------------------
def sample_to_text(lang, sample):
    """Return a chat-style string for a single training example."""
    if lang == "de" and "messages" in sample and isinstance(sample["messages"], list):
        # ShareGPT / ChatML
        parts = []
        for m in sample["messages"]:
            role = m.get("role", "")
            content = m.get("content", "")
            if not content:
                continue
            if role in ("user", "human", "User", "USER"):
                parts.append(f"{USR_TOKEN}{content}")
            elif role in ("assistant", "bot", "Assistant", "ASSISTANT"):
                parts.append(f"{AIS_TOKEN}{content}")
            elif role in ("system",):
                parts.append(f"{SYS_TOKEN}{content}")
            else:
                parts.append(f"{USR_TOKEN}{content}")
        return "\n".join(parts)

    # Alpaca-like {instruction, input, output}
    instruction = (sample.get("instruction") or sample.get("prompt") or sample.get("question") or "").strip()
    context = (sample.get("input") or sample.get("context") or "").strip()
    output = (sample.get("output") or sample.get("response") or sample.get("answer") or "").strip()

    if not instruction or not output:
        return None

    sys_prompt = {
        "zh": ID_SYS_PROMPT_ZH,
        "en": ID_SYS_PROMPT_EN,
        "de": ID_SYS_PROMPT_DE,
    }.get(lang, ID_SYS_PROMPT_EN)

    user_text = instruction
    if context:
        user_text = f"{instruction}\n\n{context}"

    return (
        f"{SYS_TOKEN}{sys_prompt}\n"
        f"{USR_TOKEN}{user_text}\n"
        f"{AIS_TOKEN}{output}"
    )


# ---------------------------------------------------------------------------
# Iterable dataset that interleaves languages + identity synthetic QA
# ---------------------------------------------------------------------------
def identity_synthetic_QA():
    """Produce Q&A pairs that teach the model its own name/developer."""
    pairs = [
        ("zh", "你叫什么名字？", "我叫 NextAI，由 Next Studio 开发。"),
        ("zh", "你是谁？", "我是 NextAI，一个由 Next Studio 开发的 AI 助手。"),
        ("zh", "你是由谁开发的？", "我是由 Next Studio 开发的，我的名字是 NextAI。"),
        ("zh", "请介绍一下你自己。", "你好！我是 NextAI，是一个由 Next Studio 开发的多语言 AI 助手，能够用中文、英文、德语与你对话。"),
        ("zh", "NextAI 是什么？", "NextAI 是由 Next Studio 开发的一个小型多语言对话 AI 模型。"),
        ("en", "What is your name?", "My name is NextAI, developed by Next Studio."),
        ("en", "Who are you?", "I am NextAI, an AI assistant developed by Next Studio."),
        ("en", "Who made you?", "I was developed by Next Studio, and my name is NextAI."),
        ("en", "Tell me about yourself.", "Hello! I am NextAI, a small multilingual chat model developed by Next Studio. I can chat in English, Chinese, and German."),
        ("en", "What is NextAI?", "NextAI is a tiny multilingual chat model developed by Next Studio."),
        ("de", "Wie heißt du?", "Ich heiße NextAI, entwickelt von Next Studio."),
        ("de", "Wer bist du?", "Ich bin NextAI, ein KI-Assistent, entwickelt von Next Studio."),
        ("de", "Wer hat dich entwickelt?", "Ich wurde von Next Studio entwickelt. Mein Name ist NextAI."),
        ("de", "Erzähl etwas über dich.", "Hallo! Ich bin NextAI, ein kleiner mehrsprachiger Chatmodell, entwickelt von Next Studio. Ich kann auf Deutsch, Englisch und Chinesisch mit dir chatten."),
        ("de", "Was ist NextAI?", "NextAI ist ein kleines mehrsprachiges Chatmodell, entwickelt von Next Studio."),
    ]
    for lang, q, a in pairs:
        sys_prompt = {
            "zh": ID_SYS_PROMPT_ZH,
            "en": ID_SYS_PROMPT_EN,
            "de": ID_SYS_PROMPT_DE,
        }[lang]
        text = (
            f"{SYS_TOKEN}{sys_prompt}\n"
            f"{USR_TOKEN}{q}\n"
            f"{AIS_TOKEN}{a}"
        )
        yield text


class ChatIterableDataset(IterableDataset):
    def __init__(self, tokenizer, max_len: int = MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Build iterators for each dataset
        self.iterators = []
        for lang, name, cfg in DATASETS:
            try:
                ds = load_dataset(name, cfg, split="train", streaming=True)
                ds = ds.shuffle(seed=random.randint(0, 1 << 30))
                # Add language tag
                self.iterators.append((lang, iter(ds), name))
                print(f"[data] streaming {name} ({lang})")
            except Exception as e:
                print(f"[data] failed load {name}: {e}")
        self.synth_iter = iter(identity_synthetic_QA())

    def _next_sample(self):
        # Weighted: sample synthetic identity QA every ~8th item to reinforce
        if random.random() < 0.18:
            try:
                return next(self.synth_iter)
            except StopIteration:
                self.synth_iter = iter(identity_synthetic_QA())
                return next(self.synth_iter)

        if not self.iterators:
            return None
        # Round-robin-ish: pick a random dataset source
        lang, it, name = random.choice(self.iterators)
        for _ in range(3):
            try:
                sample = next(it)
            except StopIteration:
                # Re-shuffle
                try:
                    ds = load_dataset(name, None, split="train", streaming=True)
                    ds = ds.shuffle(seed=random.randint(0, 1 << 30))
                    new_it = iter(ds)
                    # Replace
                    self.iterators = [
                        (l, new_it if it is orig else orig, n)
                        for (l, orig, n) in self.iterators
                    ]
                    lang, it, name = random.choice(self.iterators)
                    sample = next(it)
                except Exception:
                    continue
            text = sample_to_text(lang, sample)
            if text:
                return text
        return None

    def __iter__(self):
        # Worker split
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            n_workers, wid = 1, 0
        else:
            n_workers, wid = worker_info.num_workers, worker_info.id

        count = 0
        while True:
            text = self._next_sample()
            if text is None:
                continue
            count += 1
            if count % n_workers != wid:
                continue
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_len + 1,
                padding="max_length" if False else False,
                return_tensors="pt",
            )
            ids = enc["input_ids"][0]
            if len(ids) < 8:
                continue
            # shift: x = ids[:-1], y = ids[1:]
            yield ids[: self.max_len + 1]


# ---------------------------------------------------------------------------
# Build / load model
# ---------------------------------------------------------------------------
def build_model(tokenizer):
    cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_inner=N_INNER,
        max_position_embeddings=MAX_SEQ_LEN,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        activation_function="gelu_new",
        initializer_range=0.02,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        summary_type="cls_index",
    )
    model = GPT2LMHeadModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"[model] {n:,} trainable parameters")
    return model


# ---------------------------------------------------------------------------
# Train loop - one epoch = max EPOCH_SECONDS seconds
# ---------------------------------------------------------------------------
def cosine_lr(step: int, total_steps: int):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * step / max(total_steps, 1)))


def train_epoch(model, optimizer, loader, epoch_idx: int, device,
                seconds: int = EPOCH_SECONDS, max_steps: int = 2000):
    model.train()
    t0 = time.time()
    total_loss = 0.0
    n_tokens = 0
    n_steps = 0
    running = 0.0

    for batch_ids in loader:
        # batch_ids: (B, L+1)
        batch_ids = batch_ids.to(device)
        x = batch_ids[:, :-1]
        y = batch_ids[:, 1:]

        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # manual LR for this step
        for pg in optimizer.param_groups:
            pg["lr"] = cosine_lr(epoch_idx * max_steps + n_steps, 40 * max_steps)
        optimizer.step()

        total_loss += loss.item()
        n_tokens += x.numel()
        n_steps += 1

        if n_steps % 20 == 0:
            elapsed = time.time() - t0
            print(f"  step={n_steps:<4d} loss={loss.item():.3f} tps={n_tokens/elapsed:,.0f} tok/s elapsed={elapsed:.1f}s")

        if time.time() - t0 > seconds or n_steps >= max_steps:
            break

    elapsed = time.time() - t0
    avg = total_loss / max(n_steps, 1)
    print(f"[epoch {epoch_idx}] steps={n_steps} avg_loss={avg:.3f} elapsed={elapsed:.1f}s tps={n_tokens/elapsed:,.0f}")
    return avg, n_steps


# ---------------------------------------------------------------------------
# Greedy / sample-based chat inference (no external sampling lib needed)
# ---------------------------------------------------------------------------
def chat_reply(model, tokenizer, user_text: str, lang: str = "en",
               max_new: int = 80, temperature: float = 0.7) -> str:
    model.eval()
    sys_prompt = {
        "zh": ID_SYS_PROMPT_ZH,
        "en": ID_SYS_PROMPT_EN,
        "de": ID_SYS_PROMPT_DE,
    }.get(lang, ID_SYS_PROMPT_EN)

    prompt = f"{SYS_TOKEN}{sys_prompt}\n{USR_TOKEN}{user_text}\n{AIS_TOKEN}"

    with torch.no_grad():
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        if input_ids.shape[1] > MAX_SEQ_LEN - max_new:
            input_ids = input_ids[:, -(MAX_SEQ_LEN - max_new):]
        # manual generate loop
        generated = []
        for _ in range(max_new):
            logits = model(input_ids=input_ids).logits[:, -1, :]
            if temperature <= 0.01:
                next_tok = torch.argmax(logits, dim=-1).unsqueeze(0)
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)
            generated.append(next_tok.item())
            input_ids = torch.cat([input_ids, next_tok], dim=-1)
            if next_tok.item() == tokenizer.eos_token_id or next_tok.item() == tokenizer.convert_tokens_to_ids(USR_TOKEN):
                break
    reply_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0)
    text = tokenizer.decode(reply_ids[0], skip_special_tokens=False)
    # Strip assistant start marker
    for mk in (AIS_TOKEN, USR_TOKEN, SYS_TOKEN, EOS_TOKEN, BOS_TOKEN):
        text = text.replace(mk, " ")
    return text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    tokenizer = load_or_build_tokenizer()
    print(f"[tokenizer] vocab={tokenizer.vocab_size} bos={tokenizer.bos_token_id} eos={tokenizer.eos_token_id} pad={tokenizer.pad_token_id}")

    dataset = ChatIterableDataset(tokenizer=tokenizer, max_len=MAX_SEQ_LEN)
    # Collate: pad/stack variable-length id sequences to batch
    def collate(items):
        # items: list of 1d LongTensors of varying length
        maxlen = min(MAX_SEQ_LEN + 1, max(x.size(0) for x in items))
        out = torch.full((len(items), maxlen), tokenizer.pad_token_id, dtype=torch.long)
        for i, x in enumerate(items):
            L = min(x.size(0), maxlen)
            out[i, :L] = x[:L]
        return out

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate,
        num_workers=0,
    )

    model_path = WORKDIR / "model_last.pt"
    if model_path.exists():
        print("[model] loading existing checkpoint ...")
        model = build_model(tokenizer)
        state = torch.load(str(model_path), map_location="cpu", weights_only=False)
        if "model" in state:
            model.load_state_dict(state["model"])
            start_epoch = int(state.get("epoch", 0))
        else:
            model.load_state_dict(state)
            start_epoch = 0
    else:
        model = build_model(tokenizer)
        start_epoch = 0
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Quick sanity check of tokenizer / data flow
    print("\n[sample data]")
    for lang, name, _ in DATASETS:
        try:
            ds = load_dataset(name, None, split="train", streaming=True)
            it = iter(ds)
            text = sample_to_text(lang, next(it))
            if text:
                print(f"  [{lang}:{name}] {text[:180]} ...")
        except Exception as e:
            print(f"  [{lang}:{name}] error: {e}")

    test_prompts = [
        ("en", "What is your name?"),
        ("zh", "你叫什么名字？"),
        ("de", "Wie heißt du?"),
        ("en", "Tell me three tips to stay healthy."),
        ("zh", "什么是原子？"),
    ]

    print("\n=== Training starts ===")
    MAX_EPOCHS = 60
    for epoch in range(start_epoch, MAX_EPOCHS):
        _ = train_epoch(model, optimizer, loader, epoch_idx=epoch, device=device)

        # Save checkpoint every epoch
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch + 1,
        }, str(model_path))
        print(f"[save] checkpoint -> {model_path}")

        # Check every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("\n--- Checkpoint conversation test ---")
            for lang, q in test_prompts:
                reply = chat_reply(model, tokenizer, q, lang=lang, max_new=70, temperature=0.6)
                print(f"  [{lang}] Q: {q}")
                print(f"         A: {reply[:240]}")
            print("------------------------------------\n")
            sys.stdout.flush()

    print("[done] training finished.")


if __name__ == "__main__":
    main()
