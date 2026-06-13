"""
bpe.py  -- A GPT-2 style Byte-Pair Encoding tokenizer, written from scratch.

Algorithm outline:
  1. Raw text -> regex-style pre-tokenization  (split into word-sized units).
  2. Each "word" is mapped character-by-character to unicode bytes so that
     every byte has a visible representation (exactly like GPT-2 / tiktoken).
  3. Iteratively merge the highest-frequency adjacent pair in the vocabulary
     until we reach `vocab_size` tokens.  This uses a priority-queue-style
     count-tracker so that each merge is found in O(1) average time after
     the initial O(words * word_len) pair-count pass.
  4. Encoding re-applies the learned merges; decoding is a straightforward
     byte -> string conversion.
"""

import collections
import json
import heapq
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 1) Byte <-> Unicode mapping
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2) Pre-tokenization (a manual scan that behaves like GPT-2's pattern)
# ---------------------------------------------------------------------------
def pre_tokenize(text: str):
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        if ch in "\r\n":
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

        # English-style contractions
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

        if ch.isdigit():
            j = i
            while j < n and text[j].isdigit() and (j - i) < 3:
                j += 1
            tokens.append(text[i:j])
            i = j
            continue

        if ch.isalpha():
            j = i
            while j < n and text[j].isalpha():
                j += 1
            tokens.append(text[i:j])
            i = j
            continue

        # Catch-all: single non-alnum non-space char
        tokens.append(ch)
        i += 1

    return tokens


# ---------------------------------------------------------------------------
# 3) The tokenizer
# ---------------------------------------------------------------------------
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
        # deduplicate, keep order
        seen = set()
        self._all_special = [t for t in self._all_special
                             if not (t in seen or seen.add(t))]

        # Populated by train() / load()
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = {}   # (a, b) -> merge rank (smaller = earlier merge)
        self._next_id = 0

    # ----------------- vocab helpers -----------------
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

    # ----------------- training -----------------
    def train(self, iterator, max_words_for_bpe: int = 150000):
        """Build BPE vocab.  `iterator` yields text strings."""

        # Step 1: collect unique words (as byte-unicode tuples) + counts
        word_freq = collections.Counter()
        total_words = 0
        for text in iterator:
            if not isinstance(text, str) or len(text) == 0:
                continue
            for w in pre_tokenize(text):
                chars = "".join(BYTES_TO_UNICODE[b] for b in w.encode("utf-8"))
                if not chars:
                    continue
                word_freq[chars] += 1
                total_words += 1
                if total_words >= max_words_for_bpe:
                    break
            if total_words >= max_words_for_bpe:
                # Allow a final drain of the iterator for identity-sentence
                # yields that come after; but in practice, stop here.
                pass

        # Represent each unique word as a MUTABLE list of sub-tokens.
        # Each sub-token starts as a single byte-unicode char.
        # word_index[i] = (list_of_subtokens, frequency)
        word_index = []
        for chars, freq in word_freq.items():
            word_index.append((list(chars), freq))

        # Initial vocab: special tokens + 256 single bytes
        self.token_to_id = {}
        self.id_to_token = {}
        self._next_id = 0
        for t in self._all_special:
            self._add_token(t)
        for b in range(256):
            self._add_token(BYTES_TO_UNICODE[b])
        self.merges = {}

        # Step 2: initial pair counts
        pair_counts = collections.Counter()
        # pair -> list of (word_idx, position_in_word) for fast updates
        pair_positions = collections.defaultdict(set)
        for wi, (tokens, freq) in enumerate(word_index):
            for pi in range(len(tokens) - 1):
                pair = (tokens[pi], tokens[pi + 1])
                pair_counts[pair] += freq
                pair_positions[pair].add((wi, pi))

        target_merges = max(0, self.vocab_size - self._next_id)
        print(f"[bpe] unique words={len(word_index)} total_subwords={total_words} "
              f"target_merges={target_merges}", file=sys.stderr)

        # Step 3: iterative merge with priority-queue-style updates
        rank = 0
        # Use a max heap (negate counts).  Entries may be stale (pair_count
        # was reduced), so we verify after popping and re-push the corrected
        # value if the heap data doesn't match reality.
        heap = []
        for pair, cnt in pair_counts.items():
            heapq.heappush(heap, (-cnt, pair))

        merges_done = 0
        while merges_done < target_merges and heap:
            neg_cnt, best = heapq.heappop(heap)
            real_cnt = pair_counts.get(best, 0)
            if real_cnt <= 0 or -neg_cnt != real_cnt:
                # stale entry; skip (a newer correct entry is already in heap)
                continue
            a, b = best
            merged = a + b

            # Add to vocab
            if not self._add_token(merged):
                # already existed somehow; skip without increasing rank
                # (shouldn't happen in clean flow)
                pass
            self.merges[(a, b)] = rank
            rank += 1
            merges_done += 1

            # For every occurrence of (a, b) in every word: merge and update
            # pair counts.
            # We make a local copy of positions because we mutate pair_positions.
            affected = list(pair_positions.get(best, ()))
            if not affected:
                pair_counts[best] = 0
                continue
            del pair_positions[best]
            pair_counts[best] = 0

            # We need to mutate word_index[wi][0] which is a list
            for wi, pi in affected:
                tokens, freq = word_index[wi]
                # bounds check: token list may have been updated if this
                # (wi, pi) is from a previous update of the same word
                if pi >= len(tokens) - 1:
                    continue
                if tokens[pi] != a or tokens[pi + 1] != b:
                    continue
                # Merge the pair in-place
                tokens[pi:pi + 2] = [merged]
                # Now update counts for neighbours of position pi:
                #  - Left: (tokens[pi-1], a) and (tokens[pi-1], merged)
                #  - Right: (b, tokens[pi+2]) and (merged, tokens[pi+2])
                #    (Note: after in-place merge, tokens[pi+1] is now the
                #    old tokens[pi+2], so there is no need to update the
                #    right neighbour pair specifically.)
                # Actually we must recompute for neighbours properly:
                #
                # Before merge: pairs at positions pi-1, pi, pi+1 were
                #   (tokens[pi-1], a)  (a, b)  (b, tokens[pi+2])
                # After merge:
                #   (tokens[pi-1], merged)  (merged, tokens[pi+2])
                # So decrement the two stale pairs, increment the two new ones.
                for delta in (-1, 0):  # pi-1 and pi are the two new/old edges
                    pos = pi + delta
                    # "pos" is the position BEFORE the merge in the original
                    # list, but we've already done the in-place merge so
                    # tokens[pos] and tokens[pos+1] now reflect the new state.
                    if 0 <= pos < len(tokens) - 1:
                        new_pair = (tokens[pos], tokens[pos + 1])
                        # count it once (we need to compute the BEFORE-state
                        # pair too but we already handled it when we removed
                        # (a, b) globally; this is getting complicated ...).
                        # Simpler approach: when we visit each (wi, pi)
                        # from the affected-set, we already know that position
                        # pi in the BEFORE list was pair (a, b). After the
                        # in-place merge, edges (pi-1, pi) and (pi, pi+1) are
                        # new and edge (pi-1, pi) corresponds to (tokens[pi-1],
                        # merged) which before was (tokens[pi-1], a) and
                        # (tokens[pi-1], b) doesn't quite exist ... too messy.
                        # Instead: DEFER neighbour updates to a second scan.
                        pass

                # Because the above neighbour logic is error-prone when many
                # occurrences of the same pair share words, we instead do a
                # clean local recount for this word. It's O(len(word)) which
                # is fine because words are short (bpe unit sized).
                # Recount strategy: subtract old pair counts for this word,
                # then add back pair counts for the new state, removing the
                # pair_positions for old pairs too.
                # But we can't easily know WHICH pair_positions entries to
                # remove for this word without tracking them, so: just
                # recompute with a second scan for positions.
                #
                # To keep correctness simple, we:
                #  (a) decrement freq for every old pair in this word AND
                #      remove (wi, pos) from pair_positions.
                #  (b) re-build pair_positions for this word and increment
                #      freq for every pair in the new state.
                #
                # NOTE: step (a) requires us to know the PRE-merge pair list.
                # We can reconstruct it because before the merge we had
                # (a, b) at position pi; so before merge the tokens list
                # was: tokens = [.. tokens[pi-1], a, b, tokens[pi+2] ..]
                # The pairs that change are those with indices pi-1, pi, pi+1
                # in the BEFORE list (all other pairs are untouched).
                # Pair at pi was (a, b) - handled globally above (deleted).
                # Pairs at pi-1 and pi+1: need decrement & re-insert into
                # pair_positions.
                #
                # Actually we've already done the in-place merge, so we need
                # to KNOW the previous tokens. The previous tokens at pi-1
                # position is tokens[pi-1] (unchanged). The previous pair at
                # pi-1 was (tokens[pi-1], a) which now is (tokens[pi-1],
                # merged). The previous pair at pi+1 was (b, tokens[pi+2])
                # which now is (merged, tokens[pi+2]). tokens[pi+2] in the
                # BEFORE state is tokens[pi+1] after the in-place merge.
                #
                # So the two affected neighbour pairs are at neighbour
                # positions pi-1 and pi+1 of the ORIGINAL list:
                before_pair_left = (tokens[pi - 1], a) if pi > 0 else None
                before_pair_right = (b, tokens[pi + 1]) if pi + 1 < len(tokens) else None
                new_pair_left = (tokens[pi - 1], merged) if pi > 0 else None
                new_pair_right = (merged, tokens[pi + 1]) if pi + 1 < len(tokens) else None

                for old_p, new_p, orig_pos in [
                    (before_pair_left, new_pair_left, pi - 1),
                    (before_pair_right, new_pair_right, pi + 1),
                ]:
                    if old_p is None:
                        continue
                    pair_counts[old_p] -= freq
                    try:
                        pair_positions[old_p].discard((wi, orig_pos))
                    except Exception:
                        pass
                    if pair_counts[old_p] <= 0:
                        pair_counts.pop(old_p, None)
                    if new_p is not None:
                        pair_counts[new_p] += freq
                        pair_positions[new_p].add((wi, orig_pos))
                        heapq.heappush(heap, (-pair_counts[new_p], new_p))

            if merges_done and merges_done % 1000 == 0:
                print(f"[bpe] merge {merges_done}/{target_merges} "
                      f"vocab={len(self.token_to_id)}", file=sys.stderr)

        print(f"[bpe] final vocab={len(self.token_to_id)} merges={len(self.merges)}",
              file=sys.stderr)

    # ----------------- encoding -----------------
    def _encode_word(self, word: str):
        """Encode a single pre-tokenized word to a list of sub-token strings."""
        chars = [BYTES_TO_UNICODE[b] for b in word.encode("utf-8")]
        if not chars:
            return []
        # Repeatedly apply the merge with the lowest rank
        while len(chars) >= 2:
            # find the adjacent pair with smallest rank
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

    # ----------------- decoding -----------------
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

    # ----------------- persistence -----------------
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
