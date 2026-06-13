"""
bpe.py  -- GPT-2 style Byte-Pair Encoding tokenizer (from scratch).

Key improvements:
  1. Force-preserve "NextAI", "Next Studio" and similar identity phrases as atomic tokens
  2. Treat each CJK character as its own pre-tokenization "word" (matches GPT-2 behavior)
  3. Byte-level fallback for every other character (256 base byte-unicode tokens)
"""

import collections
import heapq
import json
import re
import sys
from pathlib import Path


def _build_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) \
       + list(range(ord("¡"), ord("¬") + 1)) \
       + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs)), dict(zip(cs, bs))


BYTES_TO_UNICODE, UNICODE_TO_BYTES = _build_bytes_to_unicode()


# --- Pre-tokenization -------------------------------------------------------

# Identity phrases to preserve as ATOMIC tokens (never split)
IDENTITY_ATOMS = [
    # English
    "NextAI", "Next Studio", "nextai", "next studio",
    # Chinese phrase atoms
    "下一个工作室", "下一代工作室", "下一个AI",
    # German-ish capitalizations
    "NextStudio",
]

# Sort by length descending for regex alternation
IDENTITY_ATOMS_SORTED = sorted(set(IDENTITY_ATOMS), key=lambda x: -len(x))


def _is_cjk(ch):
    """Check if a char is in a CJK unified ideograph / Hangul / Hiragana / Katakana range."""
    cp = ord(ch)
    return (
        0x3400 <= cp <= 0x4DBF    # CJK Ext A
        or 0x4E00 <= cp <= 0x9FFF  # CJK Unified
        or 0x3040 <= cp <= 0x30FF  # Hiragana / Katakana
        or 0xAC00 <= cp <= 0xD7A3  # Hangul
        or 0xF900 <= cp <= 0xFAFF  # Compatibility Ideographs
        or 0x20000 <= cp <= 0x2A6DF  # CJK Ext B
    )


def pre_tokenize(text: str):
    """
    Split text into word-sized units with these rules:
      1. Identity atoms (NextAI, Next Studio) are split out FIRST as whole units.
      2. Every CJK character becomes its own unit.
      3. ASCII punctuation / apostrophe fragments (it's, don't) get conventional splitting.
      4. Whitespace (incl. newlines) is preserved as its own unit.
      5. Runs of letters/digits each form a unit.
    """
    tokens = []
    i = 0
    n = len(text)

    # Build regex to find identity atoms first (case sensitive match)
    # We iterate character by character but check atoms at each position.

    while i < n:
        ch = text[i]

        # 1) Check for identity atom starting at position i
        matched_atom = False
        for atom in IDENTITY_ATOMS_SORTED:
            if text[i:i + len(atom)] == atom:
                tokens.append(atom)
                i += len(atom)
                matched_atom = True
                break
        if matched_atom:
            continue

        # 2) CJK character -> its own word
        if _is_cjk(ch):
            tokens.append(ch)
            i += 1
            continue

        # 3) Whitespace / newline -> preserved unit
        if ch == "\n" or ch == "\r":
            j = i
            while j < n and text[j] in "\r\n":
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
        if ch.isspace():
            j = i
            while j < n and text[j].isspace() and text[j] not in "\r\n":
                j += 1
            tokens.append(text[i:j])
            i = j
            continue

        # 4) English contractions (eat the longest "'s" / "'re" etc.)
        if ch == "'" or ch == "\u2019":
            frags = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
            matched = False
            for f in frags:
                if text[i:i + len(f)].lower() == f.lower():
                    tokens.append(text[i:i + len(f)])
                    i += len(f)
                    matched = True
                    break
            if matched:
                continue

        # 5) Digit run (up to 3)
        if ch.isdigit():
            j = i
            while j < n and text[j].isdigit() and (j - i) < 3:
                j += 1
            tokens.append(text[i:j])
            i = j
            continue

        # 6) ASCII letter run (English / Latin words)
        if ch.isalpha() and ord(ch) < 0x80:
            j = i
            while j < n and text[j].isalpha() and ord(text[j]) < 0x80:
                j += 1
            tokens.append(text[i:j])
            i = j
            continue

        # 7) Other letters (accented chars, Greek, Cyrillic, etc.)
        if ch.isalpha():
            j = i
            while j < n and text[j].isalpha() and not _is_cjk(text[j]):
                j += 1
            tokens.append(text[i:j])
            i = j
            continue

        # 8) Any other non-alphanumeric non-space: one char per token
        tokens.append(ch)
        i += 1

    return tokens


# --- The tokenizer ---------------------------------------------------------

class BPETokenizer:
    def __init__(self, vocab_size: int = 5000,
                 bos_token: str = "<s>",
                 eos_token: str = "</s>",
                 pad_token: str = "<pad>",
                 special_tokens: list = None):
        self.vocab_size = vocab_size
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        extra = list(special_tokens or [])
        self._all_special = [bos_token, eos_token, pad_token] + extra
        seen = set()
        self._all_special = [t for t in self._all_special
                             if not (t in seen or seen.add(t))]

        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = {}
        self._next_id = 0

    def _add_token(self, tok: str):
        if tok in self.token_to_id:
            return False
        self.token_to_id[tok] = self._next_id
        self.id_to_token[self._next_id] = tok
        self._next_id += 1
        return True

    @property
    def bos_id(self):
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self):
        return self.token_to_id[self.eos_token]

    @property
    def pad_id(self):
        return self.token_to_id[self.pad_token]

    def train(self, iterator, max_words_for_bpe: int = 200000):
        word_freq = collections.Counter()
        total = 0
        for text in iterator:
            if not isinstance(text, str) or len(text) == 0:
                continue
            for w in pre_tokenize(text):
                word_freq[w] += 1
                total += 1
                if total >= max_words_for_bpe:
                    break
            if total >= max_words_for_bpe:
                continue

        # Initial vocab: special tokens + 256 byte tokens
        self.token_to_id = {}
        self.id_to_token = {}
        self._next_id = 0
        for t in self._all_special:
            self._add_token(t)
        for b in range(256):
            self._add_token(BYTES_TO_UNICODE[b])

        # Identity atoms → also add as vocab tokens (they won't be BPE-merged
        # but since pre_tokenize returns them as whole words, the byte-level
        # mapping will be used). Actually, we need identity atoms to get
        # encoded as a SINGLE token. We do that by adding them as vocab atoms
        # BEFORE BPE merge, which effectively protects them from further
        # splitting because encode_word will check if the whole word is in
        # token_to_id first.
        for atom in IDENTITY_ATOMS_SORTED:
            self._add_token("".join(BYTES_TO_UNICODE[b] for b in atom.encode("utf-8")))

        # Represent each unique word as list of subtokens
        word_index = []
        for chars, freq in word_freq.items():
            # Byte-unicode mapping
            bu = "".join(BYTES_TO_UNICODE[b] for b in chars.encode("utf-8"))
            # Start as single-char list; but if the whole thing is a known
            # token (identity atom), just store as single string.
            if bu in self.token_to_id:
                word_index.append(([bu], freq))
            else:
                word_index.append((list(bu), freq))

        # Pair counting
        pair_counts = collections.Counter()
        for wi, (tokens, freq) in enumerate(word_index):
            for pi in range(len(tokens) - 1):
                pair = (tokens[pi], tokens[pi + 1])
                pair_counts[pair] += freq

        target_merges = max(0, self.vocab_size - self._next_id)
        print(f"[bpe] unique words={len(word_index)} total_subwords={total} "
              f"target_merges={target_merges}", file=sys.stderr)

        heap = []
        for pair, cnt in pair_counts.items():
            heapq.heappush(heap, (-cnt, pair))

        merges_done = 0
        while merges_done < target_merges and heap:
            neg_cnt, best = heapq.heappop(heap)
            real_cnt = pair_counts.get(best, 0)
            if real_cnt <= 0 or -neg_cnt != real_cnt:
                continue
            a, b = best
            merged = a + b
            self._add_token(merged)
            self.merges[(a, b)] = merges_done
            merges_done += 1

            # Update affected words
            for wi, (tokens, freq) in enumerate(word_index):
                new_toks = []
                j = 0
                changed = False
                while j < len(tokens):
                    if j < len(tokens) - 1 and tokens[j] == a and tokens[j + 1] == b:
                        new_toks.append(merged)
                        j += 2
                        changed = True
                    else:
                        new_toks.append(tokens[j])
                        j += 1
                if changed:
                    # Decrement old pair counts
                    # Simpler: walk both old and new, decrement/increment.
                    old_toks = tokens
                    # Decrement all pairs in old
                    for pi in range(len(old_toks) - 1):
                        op = (old_toks[pi], old_toks[pi + 1])
                        pair_counts[op] -= freq
                        if pair_counts[op] <= 0:
                            pair_counts.pop(op, None)
                    # Increment all pairs in new
                    for pi in range(len(new_toks) - 1):
                        np_ = (new_toks[pi], new_toks[pi + 1])
                        pair_counts[np_] += freq
                        heapq.heappush(heap, (-pair_counts[np_], np_))
                    word_index[wi] = (new_toks, freq)

            if merges_done % 500 == 0:
                print(f"[bpe] merge {merges_done}/{target_merges} "
                      f"vocab={len(self.token_to_id)}", file=sys.stderr)

        print(f"[bpe] final vocab={len(self.token_to_id)} merges={len(self.merges)}",
              file=sys.stderr)

    def _encode_word(self, word: str):
        """Encode a single pre-tokenized word to a list of BPE sub-token strings."""
        byte_unicode = "".join(BYTES_TO_UNICODE[b] for b in word.encode("utf-8"))
        # Fast path: whole identity atom exists as a single token
        if byte_unicode in self.token_to_id:
            return [byte_unicode]
        chars = list(byte_unicode)
        while len(chars) >= 2:
            best_rank = None
            best_i = -1
            for i in range(len(chars) - 1):
                r = self.merges.get((chars[i], chars[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_rank is None:
                break
            chars[best_i:best_i + 2] = [chars[best_i] + chars[best_i + 1]]
        return chars

    def encode(self, text: str, max_length: int = None,
               add_bos: bool = False, add_eos: bool = False) -> list:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for word in pre_tokenize(text):
            for sw in self._encode_word(word):
                ids.append(self.token_to_id[sw])
        if add_eos:
            ids.append(self.eos_id)
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: list, skip_special: bool = True) -> str:
        out_chars = []
        for i in ids:
            tok = self.id_to_token.get(int(i))
            if tok is None:
                continue
            if skip_special and tok in self._all_special:
                continue
            out_chars.append(tok)
        joined = "".join(out_chars)
        byte_vals = []
        for ch in joined:
            b = UNICODE_TO_BYTES.get(ch)
            if b is None:
                byte_vals.extend(ch.encode("utf-8"))
            else:
                byte_vals.append(b)
        try:
            return bytes(byte_vals).decode("utf-8", errors="replace")
        except Exception:
            return bytes(byte_vals).decode("utf-8", errors="replace")

    # persistence
    def save(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        merges_json = {" ".join(k): int(v) for k, v in self.merges.items()}
        data = {
            "vocab_size": self.vocab_size,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "all_special": self._all_special,
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "merges": merges_json,
        }
        with open(save_dir / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, save_dir):
        save_dir = Path(save_dir)
        with open(save_dir / "tokenizer.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(
            vocab_size=data["vocab_size"],
            bos_token=data["bos_token"],
            eos_token=data["eos_token"],
            pad_token=data["pad_token"],
        )
        tok._all_special = data.get("all_special", [data["bos_token"],
                                                     data["eos_token"],
                                                     data["pad_token"]])
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        tok._next_id = max(tok.id_to_token.keys()) + 1
        merges = {}
        for k, v in data["merges"].items():
            parts = k.split(" ")
            if len(parts) >= 2:
                merges[(parts[0], " ".join(parts[1:]))] = int(v)
        tok.merges = merges
        return tok
