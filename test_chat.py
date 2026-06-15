import torch
import sys
sys.path.insert(0, '/workspace')

from train_lstm_nextai import NextAILSTM, ByteTokenizer

PAD, BOS, EOS, UNK = 0, 1, 2, 3

# 加载 checkpoint
ckpt = torch.load("/workspace/nextai-full.pt", map_location='cpu')
CFG = ckpt["cfg"]
tok_data = ckpt["tokenizer"]

# 构建分词器
tok = ByteTokenizer(
    b2i=tok_data["b2i"],
    i2b=tok_data["i2b"],
    merges=tok_data["merges"],
    vocab_size=tok_data["vocab_size"],
)
print("分词器: vocab={}, merges={}".format(tok.vocab_size, len(tok.merges)))
print("CFG: hidden_size={}, n_layers={}, max_len={}".format(CFG["hidden_size"], CFG["n_layers"], CFG["max_len"]))

# 加载模型
model = NextAILSTM(CFG)
model.load_state_dict(ckpt['model'])
model.eval()
print("模型参数:", sum(p.numel() for p in model.parameters()))

def chat(text, max_new=80):
    src_ids_full = tok.encode(text, max_len=CFG["max_len"])
    # src_ids_full = [BOS, ...tokens..., EOS]
    # generate 需要的是 src_ids (可以直接传)
    
    print(f"\n> {text}")
    print("  ", end="", flush=True)
    
    generated_ids = model.generate(src_ids_full, max_new=max_new)
    text_out = tok.decode(generated_ids)
    print(text_out)
    return text_out

print("\n=== 开始测试 ===")
test_cases = [
    "你是谁？",
    "你的名字是什么？",
    "What is your name?",
    "Who are you?",
    "translate to English: 你好",
    "translate to English: 我爱你",
    "translate to English: 谢谢你",
    "翻译成中文：Hello",
    "翻译成中文：I love you",
    "翻译成中文：Good morning",
    "Q: 什么是人工智能？",
    "Q: 法国的首都是什么？",
    "Q: 天空是什么颜色？",
    "Wie heißt du?",
    "你好，最近怎么样？",
]

for q in test_cases:
    chat(q)

print("\n=== 测试完成 ===")
