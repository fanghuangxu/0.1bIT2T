"""Build tokenizer cache for NextAI model evaluation."""
import random
import pickle
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from train_nextai import ByteTokenizer, build_pairs

seed = 1337
random.seed(seed)

def main():
    print("Building tokenizer cache...")
    pairs = build_pairs()

    # Flatten and shuffle texts (same as train_nextai.py)
    all_texts = []
    for a, b in pairs:
        all_texts.append(a)
        all_texts.append(b)
    random.shuffle(all_texts)

    print(f"Training tokenizer on {len(all_texts)} texts...")
    tok = ByteTokenizer(vocab_size=2048)
    tok.learn(all_texts, max_merges=1780)  # Match training: 1780 merges

    cache = {
        "b2i": tok.b2i,
        "i2b": tok.i2b,
        "merges": tok.merges,
        "vocab_size": tok.vocab_size,
    }

    with open("tokenizer_cache.pkl", "wb") as f:
        pickle.dump(cache, f)

    print(f"Tokenizer cache saved: vocab_size={tok.vocab_size}, merges={len(tok.merges)}")


if __name__ == "__main__":
    main()
