"""
train_nextai.py

A tiny ~1M-parameter GPT-style model trained on Hugging Face datasets
in English / Chinese / German.  Uses a custom from-scratch BPE tokenizer
(see bpe.py) and a GPT2-like transformer.

The model is instructed via role markers:
    <sys>   ... system prompt  (describes NextAI / Next Studio)
    <usr>   ... user question
    <ai>    ... assistant answer (this is the only role the model trains to emit)

Training rounds are limited to ~45 seconds each to satisfy the
"每轮不超过一分钟" requirement.  Every 5 rounds a conversation test is run.
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
N_EMBD = 96
N_LAYER = 4
N_HEAD = 4
N_INNER = 4 * N_EMBD

BATCH_SIZE = 8
LEARNING_RATE = 5e-4
WARMUP_STEPS = 100
EPOCH_SECONDS = 45

# Role markers.  These are the only place control tokens appear;
# training loss only applies to the text after <ai>.
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

ID_SYS_ZH = "你是一个名为 NextAI 的智能助手，由 Next Studio 开发。请用中文自然回答用户的问题。"
ID_SYS_EN = "You are NextAI, a helpful AI assistant developed by Next Studio. Please respond naturally in English."
ID_SYS_DE = "Du bist NextAI, ein hilfreicher KI-Assistent, entwickelt von Next Studio. Bitte antworte natürlich auf Deutsch."


# ---------------------------------------------------------------------------
# Build or load our from-scratch BPE tokenizer
# ---------------------------------------------------------------------------
def text_stream_for_bpe():
    for lang, name, cfg in DATASETS:
        print(f"[bpe] sampling text from {name} ...", file=sys.stderr)
        ds = load_dataset(name, cfg, split="train", streaming=True)
        ds = ds.shuffle(seed=42)
        count = 0
        for sample in ds:
            for k, v in sample.items():
                if isinstance(v, str):
                    yield v[:3000]
                elif isinstance(v, list):
                    for m in v:
                        if isinstance(m, dict) and "content" in m:
                            yield str(m["content"])[:3000]
            count += 1
            if count >= 800:
                break
    # Reinforce identity tokens / markers
    for _ in range(50):
        for s in [
            "NextAI", "Next Studio",
            ID_SYS_ZH, ID_SYS_EN, ID_SYS_DE,
            f"{SYS_TOKEN}{ID_SYS_EN}\n{USR_TOKEN}What is your name?\n{AIS_TOKEN}My name is NextAI, developed by Next Studio.",
            f"{SYS_TOKEN}{ID_SYS_ZH}\n{USR_TOKEN}你叫什么名字？\n{AIS_TOKEN}我叫 NextAI，由 Next Studio 开发。",
            f"{SYS_TOKEN}{ID_SYS_DE}\n{USR_TOKEN}Wie heißt du?\n{AIS_TOKEN}Ich heiße NextAI, entwickelt von Next Studio.",
        ]:
            yield s


def build_or_load_tokenizer(force_rebuild: bool = False):
    tok_path = WORKDIR / "tokenizer.json"
    if (not force_rebuild) and tok_path.exists():
        print("[tokenizer] loading cached tokenizer ...")
        return BPETokenizer.load(WORKDIR)

    print(f"[tokenizer] building custom BPE tokenizer (target vocab={VOCAB_SIZE}) ...")
    tok = BPETokenizer(
        vocab_size=VOCAB_SIZE,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        special_tokens=[SYS_TOKEN, USR_TOKEN, AIS_TOKEN],
    )
    tok.train(text_stream_for_bpe(), max_words_for_bpe=200000)
    tok.save(WORKDIR)
    print(f"[tokenizer] final vocab size = {len(tok.token_to_id)}")
    return tok


# ---------------------------------------------------------------------------
# Data preparation:  convert dataset samples to chat-style ids
# ---------------------------------------------------------------------------
def build_chat_ids(lang, sample, tokenizer: BPETokenizer,
                   max_len: int = MAX_SEQ_LEN, train_on_ai_only: bool = True):
    """Return (input_ids, label_mask) tensors ready for training.

    `label_mask` is 1 for positions whose loss we care about (after <ai>),
    0 for the rest (system prompt, user question).
    """
    if lang == "de" and isinstance(sample.get("messages"), list):
        # ShareGPT-style messages
        parts = []
        for m in sample["messages"]:
            role = (m.get("role") or "").lower()
            content = m.get("content") or ""
            if not content:
                continue
            if role in ("user", "human"):
                parts.append(f"{USR_TOKEN}{content}\n")
            elif role in ("assistant", "bot", "gpt"):
                parts.append(f"{AIS_TOKEN}{content}\n")
            elif role == "system":
                parts.append(f"{SYS_TOKEN}{content}\n")
            else:
                parts.append(f"{USR_TOKEN}{content}\n")
        text = "".join(parts)
        # Mark <ai> regions later.  For simplicity we train on everything
        # after the first <ai> in this path (good enough for small model).
        # We'll extract <ai>-role substring and still use generic masking
        # via substring position search.
        ai_only_text_parts = [p for p in parts if p.startswith(AIS_TOKEN)]
        ai_joined = "".join(ai_only_text_parts)
        # Encode once, find <ai> positions by re-encoding markers.
        full_ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        # Find which tokens coincide with "<ai>"-the-marker itself or the
        # text that follows it up to the next role marker.
        # Simplest: re-encode without markers and find positions by
        # comparing.  We implement via scanning: keep track of whether the
        # current character position is "inside an assistant turn".
        # Because the tokenizer is sub-word and byte-level, we instead just
        # train on the full text here (small model -- full sequence LM is fine).
        ai_only_ids = tokenizer.encode(ai_joined, add_bos=False, add_eos=False)
        label_mask = [1] * len(full_ids)  # train everything; for chat we'll
                                           # also do alpaca-style masking on
                                           # instruction data.
        return full_ids, label_mask

    # Alpaca-like: instruction / input / output
    instruction = (sample.get("instruction") or sample.get("prompt") or
                   sample.get("question") or "").strip()
    context = (sample.get("input") or sample.get("context") or "").strip()
    output = (sample.get("output") or sample.get("response") or
              sample.get("answer") or "").strip()
    if not instruction or not output:
        return None

    sys_prompt = {"zh": ID_SYS_ZH, "en": ID_SYS_EN, "de": ID_SYS_DE}.get(lang, ID_SYS_EN)
    user_text = instruction if not context else f"{instruction}\n\n{context}"

    # Build:  <sys>...\n<usr>...\n<ai>...
    # Split by which parts are trainable:
    #   prefix  = "<sys>...\n<usr>...\n<ai>"        -> NOT trained (mask=0)
    #   suffix  = output                             -> trained (mask=1)
    prefix_text = f"{SYS_TOKEN}{sys_prompt}\n{USR_TOKEN}{user_text}\n{AIS_TOKEN}"
    suffix_text = output

    prefix_ids = tokenizer.encode(prefix_text, add_bos=False, add_eos=False)
    suffix_ids = tokenizer.encode(suffix_text, add_bos=False, add_eos=False)

    # Add BOS at very start and EOS at very end
    input_ids = [tokenizer.bos_id] + prefix_ids + suffix_ids + [tokenizer.eos_id]
    label_mask = [0] + [0] * len(prefix_ids) + [1] * len(suffix_ids) + [0]

    if len(input_ids) > max_len:
        # Prefer to preserve the suffix (answer): take from the right
        # but keep at least a bit of prefix context.
        # Strategy: keep the last max_len tokens, but never truncate inside
        # suffix if possible.
        if len(suffix_ids) + 10 < max_len:
            # Keep whole suffix; truncate prefix from the left
            tail_prefix = prefix_ids[-(max_len - len(suffix_ids) - 2):]
            input_ids = [tokenizer.bos_id] + tail_prefix + suffix_ids + [tokenizer.eos_id]
            label_mask = [0] + [0] * len(tail_prefix) + [1] * len(suffix_ids) + [0]
        else:
            # Just take the last max_len tokens from the whole thing
            input_ids = input_ids[-max_len:]
            label_mask = label_mask[-max_len:]

    assert len(input_ids) == len(label_mask)
    return input_ids, label_mask


# ---------------------------------------------------------------------------
# Iterable dataset: stream from all three datasets
# ---------------------------------------------------------------------------
class ChatDataset(IterableDataset):
    def __init__(self, tokenizer: BPETokenizer, max_len: int = MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.iterators = []
        for lang, name, cfg in DATASETS:
            try:
                ds = load_dataset(name, cfg, split="train", streaming=True)
                ds = ds.shuffle(seed=random.randint(0, 2**31 - 1))
                self.iterators.append((lang, iter(ds)))
                print(f"[data] streaming {name} ({lang})")
            except Exception as e:
                print(f"[data] failed load {name}: {e}")

        # Synthetic identity Q&A
        self.id_pairs = [
            ("zh", "你叫什么名字？", "我叫 NextAI，由 Next Studio 开发。"),
            ("zh", "你是谁？", "我是 NextAI，一个由 Next Studio 开发的 AI 助手。"),
            ("zh", "你是由谁开发的？", "我由 Next Studio 开发，我的名字是 NextAI。"),
            ("zh", "请介绍一下你自己。", "我是 NextAI，由 Next Studio 开发，擅长中英德三语对话。"),
            ("zh", "NextAI 是什么？", "NextAI 是由 Next Studio 开发的一个小型多语言对话模型。"),
            ("en", "What is your name?", "My name is NextAI, developed by Next Studio."),
            ("en", "Who are you?", "I am NextAI, an AI assistant developed by Next Studio."),
            ("en", "Who made you?", "I was made by Next Studio. My name is NextAI."),
            ("en", "Tell me about yourself.", "I am NextAI, a small multilingual chat model developed by Next Studio."),
            ("en", "What is NextAI?", "NextAI is a tiny multilingual chat model developed by Next Studio."),
            ("de", "Wie heißt du?", "Ich heiße NextAI, entwickelt von Next Studio."),
            ("de", "Wer bist du?", "Ich bin NextAI, ein KI-Assistent, entwickelt von Next Studio."),
            ("de", "Wer hat dich entwickelt?", "Ich wurde von Next Studio entwickelt. Mein Name ist NextAI."),
            ("de", "Erzähl etwas über dich.", "Ich bin NextAI, ein kleines mehrsprachiges Chatmodell von Next Studio."),
            ("de", "Was ist NextAI?", "NextAI ist ein kleines mehrsprachiges Chatmodell, entwickelt von Next Studio."),
        ]

    def _next_sample(self):
        # 22% chance to emit an identity QA example -- teaches the model who it is
        if random.random() < 0.22:
            lang, q, a = random.choice(self.id_pairs)
            sample = {"instruction": q, "input": "", "output": a}
            res = build_chat_ids(lang, sample, self.tokenizer, self.max_len)
            if res is not None:
                return res

        if not self.iterators:
            return None
        lang, it = random.choice(self.iterators)
        for _ in range(5):
            try:
                sample = next(it)
            except StopIteration:
                return None
            res = build_chat_ids(lang, sample, self.tokenizer, self.max_len)
            if res is not None and len(res[0]) >= 16:
                return res
        return None

    def __iter__(self):
        while True:
            res = self._next_sample()
            if res is None:
                continue
            input_ids, label_mask = res
            yield torch.tensor(input_ids, dtype=torch.long), \
                  torch.tensor(label_mask, dtype=torch.long)


def collate_fn(items):
    """Pad (input_ids, label_mask) tuples to same length in a batch."""
    # items is list of (ids, mask) tensors of varying lengths
    maxlen = min(MAX_SEQ_LEN, max(len(ids) for ids, _ in items))
    pad = torch.tensor([items[0][0][0].item()])  # placeholder; real pad id below
    # actually we need tokenizer; we handle padding outside by passing pad_id as global.
    # But simpler: use -1 for "pad" decision in label mask and tokenizer.pad_id for ids.
    pad_id = collate_fn.pad_id
    batched_ids = torch.full((len(items), maxlen), pad_id, dtype=torch.long)
    batched_masks = torch.zeros((len(items), maxlen), dtype=torch.long)
    for i, (ids, m) in enumerate(items):
        L = min(len(ids), maxlen)
        batched_ids[i, :L] = ids[:L]
        batched_masks[i, :L] = m[:L]
    return batched_ids, batched_masks


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
def build_model(vocab_size: int, pad_id: int, eos_id: int, bos_id: int):
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
    )
    model = GPT2LMHeadModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"[model] {n:,} trainable parameters")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def cosine_lr(step: int, total_steps: int):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * step / max(total_steps, 1)))


def train_epoch(model, optimizer, loader, epoch_idx, device, seconds=EPOCH_SECONDS):
    """Run one training round (epoch) of up to `seconds` seconds."""
    model.train()
    t0 = time.time()
    total_loss = 0.0
    n_tokens = 0
    n_steps = 0

    for input_ids, label_mask in loader:
        input_ids = input_ids.to(device)
        label_mask = label_mask.to(device)

        # Build labels: copy input_ids shifted, masked where label_mask==0 and
        # on pad positions.
        x = input_ids[:, :-1].contiguous()
        # Labels: next-token prediction; mask positions that are pad or where
        # label_mask == 0 (we use label_mask shifted by 1 so it lines up with
        # the "next token" position).
        next_token_mask = label_mask[:, 1:].contiguous()
        y = input_ids[:, 1:].contiguous().clone()

        # Mask: set y=-100 wherever next_token_mask==0 OR input is pad
        y[next_token_mask == 0] = -100
        y[y == collate_fn.pad_id] = -100  # never train on pad

        # Attention mask: 1 where input is not pad
        attn_mask = (x != collate_fn.pad_id).long()

        outputs = model(input_ids=x, attention_mask=attn_mask, labels=y)
        loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # LR scheduling
        lr = cosine_lr(epoch_idx * 500 + n_steps, 40 * 500)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.step()

        total_loss += loss.item()
        n_tokens += attn_mask.sum().item()
        n_steps += 1

        if n_steps % 20 == 0:
            elapsed = time.time() - t0
            print(f"  step={n_steps:<4d} loss={loss.item():.3f} lr={lr:.2e} "
                  f"tok/s={n_tokens/max(elapsed,1e-6):.0f} elapsed={elapsed:.1f}s")

        if time.time() - t0 > seconds:
            break

    elapsed = time.time() - t0
    print(f"[epoch {epoch_idx}] steps={n_steps} avg_loss={total_loss/max(n_steps,1):.3f} "
          f"elapsed={elapsed:.1f}s tok/s={int(n_tokens/max(elapsed,1e-6))}")


# ---------------------------------------------------------------------------
# Inference: greedy + temperature sampling, with forbidden-token filter
# ---------------------------------------------------------------------------
@torch.no_grad()
def chat_reply(model, tokenizer: BPETokenizer, user_text: str, lang: str = "en",
               max_new: int = 80, temperature: float = 0.8) -> str:
    model.eval()
    device = next(model.parameters()).device

    sys_prompt = {"zh": ID_SYS_ZH, "en": ID_SYS_EN, "de": ID_SYS_DE}.get(lang, ID_SYS_EN)
    prompt_text = f"{SYS_TOKEN}{sys_prompt}\n{USR_TOKEN}{user_text}\n{AIS_TOKEN}"
    prefix_ids = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)

    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    if input_ids.shape[1] > MAX_SEQ_LEN - max_new:
        input_ids = input_ids[:, -(MAX_SEQ_LEN - max_new):]

    # Tokens the model is forbidden to emit (role markers + pad).
    # The model should only emit normal text after <ai>.
    forbidden = {tokenizer.pad_id, tokenizer.bos_id}
    # role marker ids: find them in the tokenizer
    for marker in [SYS_TOKEN, USR_TOKEN, AIS_TOKEN]:
        if marker in tokenizer.token_to_id:
            forbidden.add(tokenizer.token_to_id[marker])

    for _ in range(max_new):
        attn = (input_ids != tokenizer.pad_id).long()
        logits = model(input_ids=input_ids, attention_mask=attn).logits[:, -1, :]

        # Penalize forbidden tokens heavily
        for tid in forbidden:
            if 0 <= tid < logits.shape[-1]:
                logits[0, tid] = -float("inf")

        if temperature <= 0.02:
            next_tok = torch.argmax(logits, dim=-1).unsqueeze(0)
        else:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)

        input_ids = torch.cat([input_ids, next_tok], dim=-1)
        if next_tok.item() == tokenizer.eos_id:
            break

    # Decode only the newly-generated part (skip prefix)
    new_ids = input_ids[0, len(prefix_ids):].tolist()
    # Filter forbidden tokens from decode too
    new_ids = [i for i in new_ids if i not in forbidden and i != tokenizer.eos_id]
    return tokenizer.decode(new_ids, skip_special=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    tokenizer = build_or_load_tokenizer(force_rebuild=False)
    collate_fn.pad_id = tokenizer.pad_id

    print(f"[tokenizer] vocab_size={len(tokenizer.token_to_id)} "
          f"bos={tokenizer.bos_id} eos={tokenizer.eos_id} pad={tokenizer.pad_id}")
    print(f"           <sys>={tokenizer.token_to_id.get(SYS_TOKEN)} "
          f"<usr>={tokenizer.token_to_id.get(USR_TOKEN)} "
          f"<ai>={tokenizer.token_to_id.get(AIS_TOKEN)}")

    # Sanity test
    test_text = "Hello, world! 你好，世界！ Hallo Welt!"
    ids = tokenizer.encode(test_text, add_bos=True, add_eos=True)
    back = tokenizer.decode(ids, skip_special=False)
    print(f"[tokenizer] round-trip test: {len(ids)} tokens")
    print(f"             orig : {test_text}")
    print(f"             decode: {back}")

    # Dataset / loader
    dataset = ChatDataset(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

    # Model
    ckpt_path = WORKDIR / "model_last.pt"
    if ckpt_path.exists():
        print("[model] loading existing checkpoint ...")
        model = build_model(len(tokenizer.token_to_id), tokenizer.pad_id,
                            tokenizer.eos_id, tokenizer.bos_id)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "model" in state:
            model.load_state_dict(state["model"])
            start_epoch = int(state.get("epoch", 0))
        else:
            model.load_state_dict(state)
            start_epoch = 0
    else:
        model = build_model(len(tokenizer.token_to_id), tokenizer.pad_id,
                            tokenizer.eos_id, tokenizer.bos_id)
        start_epoch = 0
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Conversation test prompts
    test_prompts = [
        ("en", "What is your name?"),
        ("zh", "你叫什么名字？"),
        ("de", "Wie heißt du?"),
        ("en", "Give three tips for staying healthy."),
        ("zh", "什么是原子？"),
        ("de", "Was ist künstliche Intelligenz?"),
    ]

    print("\n=== Training starts ===")
    MAX_EPOCHS = 80
    for epoch in range(start_epoch, MAX_EPOCHS):
        train_epoch(model, optimizer, loader, epoch_idx=epoch, device=device)

        torch.save({"model": model.state_dict(), "epoch": epoch + 1}, str(ckpt_path))
        print(f"[save] checkpoint -> {ckpt_path}")

        if (epoch + 1) % 5 == 0:
            print("\n--- Conversation test ---")
            for lang, q in test_prompts:
                reply = chat_reply(model, tokenizer, q, lang=lang, max_new=90, temperature=0.7)
                print(f"  [{lang}] Q: {q}")
                print(f"         A: {reply}")
            print("--------------------------\n")
            sys.stdout.flush()

    print("[done] training finished.")


if __name__ == "__main__":
    main()
