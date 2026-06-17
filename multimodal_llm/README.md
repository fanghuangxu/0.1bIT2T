# 多语言对话AI系统

基于 Qwen2.5-1.5B 的多语言对话系统，支持中文、英文、德文，可部署在2GB显存或CPU上运行。

## 核心特性

- **多语言支持**: 中文、英文、德文
- **Markdown输出**: 支持代码块、列表、标题等格式
- **EOS终止符**: 模型自行决定停止输出时机
- **RL奖励机制**: 长度奖励 + 内容正确性奖励
- **多轮对话**: 上下文记忆，最长20轮对话
- **角色扮演**: 预设6种角色，支持自定义角色
- **低资源运行**: 4bit量化支持，2GB显存即可推理

## 模型选择

推荐使用 **Qwen2.5-1.5B-Instruct**:
- 量化后仅需 ~1.1GB 显存
- 支持29种语言
- MMLU: 60.9
- 可在CPU上运行

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```python
from multimodal_llm import create_chat_ai

# 创建AI实例 (会自动使用Qwen2.5-1.5B)
ai = create_chat_ai()

# 对话
result = ai.chat("你好，请用中英德三种语言打招呼")
print(result["text"])
```

## 使用预训练模型

```python
ai = create_chat_ai(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    device="auto",
    use_quantization=True
)
```

## 角色扮演

```python
# 列出可用角色
roles = ai.list_roles()

# 切换角色
ai.switch_role("translator")

# 对话
result = ai.chat("请翻译: Hello, how are you?")
```

## RL奖励机制

系统自动计算奖励，包括:
- **长度奖励**: 鼓励适当长度的完整回答
- **正确性奖励**: 连贯性、事实性、安全性
- **EOS奖励**: 自然停止vs强制停止

```python
# 获取奖励详情
result = ai.chat("你好", return_rewards=True)
print(result["rewards"])
```

## 数据集配置

推荐数据集组合 (符合2GB显存约束):

| 阶段 | 数据集 | 用途 |
|------|--------|------|
| 预训练 | FineWeb-Edu (40%) + SkyPile (25%) | 英文教育 + 中文 |
| SFT | alpaca-zh (35%) + UltraFeedback (30%) | 指令微调 |
| DPO | HH-RLHF (50%) + PKU-Alignment (50%) | 偏好学习 |

## 训练

```bash
# DPO训练
python -m multimodal_llm.src.training \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --train_data ./data/dpo_train.jsonl \
    --output_dir ./output \
    --epochs 3
```

## 硬件要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| 显存 | 2GB | 4GB |
| 内存 | 8GB | 16GB |
| CPU | 支持 | 支持 |

## 项目结构

```
multimodal_llm/
├── configs/          # 配置文件
│   └── config.py
├── datasets/         # 数据集配置
│   └── dataset_config.py
├── src/              # 源代码
│   ├── dialogue/     # 对话管理
│   ├── rewards/      # RL奖励
│   ├── roles/        # 角色系统
│   ├── model_interface.py
│   └── training.py
└── __init__.py
```

## License

Apache 2.0
