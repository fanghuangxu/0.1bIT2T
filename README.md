# Mini Dialog AI

一个轻量级的对话AI模型，支持图像理解和文本生成功能。

## 功能特性

- **文本到文本**: 使用 BART 模型进行文本生成
- **图像到文本**: 使用 BLIP 模型进行图像描述和图像问答
- **对话历史**: 支持对话历史管理
- **对话摘要**: 支持对话内容摘要

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 使用 CLI

```bash
python -m mini_dialog_ai.cli
```

### 使用 Python API

```python
from mini_dialog_ai import DialogAI

ai = DialogAI(device="cpu")

# 文本对话
response = ai.chat("Hello!")
print(response)

# 图像描述
response = ai.chat("", image_path="image.jpg")
print(response)

# 图像问答
response = ai.chat("What is in this image?", image_path="image.jpg")
print(response)

# 对话摘要
summary = ai.summarize_conversation()
print(summary)
```

## 项目结构

```
.
├── mini_dialog_ai/
│   ├── __init__.py
│   ├── config.py          # 配置文件
│   ├── image_model.py     # 图像理解模型
│   ├── text_model.py      # 文本生成模型
│   ├── model.py           # 统一对话接口
│   └── cli.py             # 命令行接口
├── examples/
│   └── basic_usage.py     # 使用示例
├── tests/
│   └── test_model.py      # 测试用例
├── requirements.txt       # 依赖列表
├── setup.py              # 安装配置
└── README.md             # 项目说明
```

## 依赖

- torch >= 2.0.0
- torchvision >= 0.15.0
- transformers >= 4.30.0
- pillow >= 10.0.0
- numpy >= 1.24.0
- accelerate >= 0.20.0
- sentencepiece >= 0.1.99

## 模型说明

- **图像模型**: Salesforce/blip-image-captioning-base
- **文本模型**: facebook/bart-large-cnn

## 许可证

MIT License