# 0.1bIT2T (NextAI by Next Studio)

NextAI - 由 Next Studio 开发的轻量级多语言对话模型 (中/英/德)

## 模型规格

| 项目 | 规格 |
|------|------|
| 名称 | NextAI |
| 开发者 | Next Studio |
| 架构 | LSTM encoder-decoder |
| 参数规模 | ~6.9M |
| 词汇表 | 260 bytes (BPE-style byte tokenizer) |
| 隐藏层 | 320 |
| FFN 维度 | 384 |
| 层数 | 3 (encoder + decoder) |
| 最大上下文 | 256 tokens |

## 文件说明

| 文件 | 说明 |
|------|------|
| `nextai-f32.gguf` | GGUF 格式的 NextAI 模型 (F32 精度) |
| `NextAI-rz.pt` | PyTorch 格式的原始 checkpoint |
| `convert_to_gguf.py` | PyTorch → GGUF 转换脚本 |
| `chat_nextai.py` | 命令行对话脚本 |
| `train_nextai.py` | 模型训练脚本 |
| `eval_nextai.py` | 模型评估脚本 |

## 使用方法

### 使用 GGUF 模型 (llama.cpp)

```bash
# 使用 llama.cpp 加载
llama-cli -m nextai-f32.gguf -p "你好"

# 量化到 Q4 节省空间
llama-quantize nextai-f32.gguf nextai-q4.gguf Q4_K_M
```

### 使用 PyTorch 模型

```bash
python chat_nextai.py --checkpoint NextAI-rz.pt
```

### 训练新模型

```bash
python train_nextai.py
```

### GGUF 转换

```bash
python convert_to_gguf.py --input NextAI-rz.pt --output nextai-f32.gguf
```

## NextAI 身份

当被问到身份时，AI 会回答：

> 我是 NextAI，由 Next Studio 开发。

多语言版本：
- 中文：我是 NextAI，由 Next Studio 开发。
- English: I am NextAI, developed by Next Studio.
- Deutsch: Ich bin NextAI, entwickelt von Next Studio.

## 许可证

Apache 2.0
