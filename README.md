# Mini Dialog AI

基于Qwen MoE架构的轻量级对话AI模型，支持文本对话和图像理解。

## 功能特性

- **文本对话**: 使用Qwen2-MoE模型进行对话生成
- **图像理解**: 使用BLIP模型进行图像描述和问答
- **低端设备支持**: 支持4-bit量化，适配低配置设备
- **对话历史**: 支持对话历史管理
- **自动数据集加载**: 从HF-mirror自动加载公开对话数据集

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 训练模型

```bash
# 使用默认配置训练
python train.py

# 使用量化训练（适合低端设备）
python train.py --quantization

# 指定参数
python train.py --model_name Qwen/Qwen2-MoE-2.7B-Instruct \
                --dataset_name HuggingFaceH4/ultrachat_200k \
                --batch_size 2 \
                --quantization
```

### 运行推理

```bash
python inference.py
```

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
```

## 支持的数据集

### 文本对话数据集
- lmsys/vicuna-chat-v1.5
- tatsu-lab/alpaca
- HuggingFaceH4/ultrachat_200k
- MAGAer13/ShareGPT_Vicuna_unfiltered
- timdettmers/openassistantistant

### 图文对话数据集
- liuhaotian/LLaVA-Instruct-150K
- microsoft/COCO-Captions
- linqingyang/cc_sbu_align
- HuggingFaceM4/COCO

## 项目结构

```
.
├── mini_dialog_ai/
│   ├── __init__.py          # 模块入口
│   ├── config.py           # 配置文件
│   ├── image_model.py      # 图像理解模型
│   ├── text_model.py       # 文本生成模型
│   ├── model.py            # 统一对话接口
│   ├── cli.py              # 命令行接口
│   ├── trainer.py          # 训练器模块
│   ├── dataset.py          # 本地数据集处理
│   └── hf_dataset.py       # HF数据集加载
├── examples/
│   └── basic_usage.py      # 使用示例
├── tests/
│   └── test_model.py       # 测试用例
├── train.py                # 训练脚本
├── inference.py            # 推理脚本
├── requirements.txt        # 依赖列表
├── setup.py               # 安装配置
└── README.md              # 项目说明
```

## 依赖

- torch >= 2.0.0
- torchvision >= 0.15.0
- transformers >= 4.30.0
- datasets >= 2.14.0
- bitsandbytes >= 0.41.0
- peft >= 0.6.0
- trl >= 0.7.0
- pillow >= 10.0.0
- numpy >= 1.24.0
- accelerate >= 0.20.0
- sentencepiece >= 0.1.99

## 模型说明

- **主模型**: Qwen/Qwen2-MoE-2.7B-Instruct (MoE架构)
- **图像模型**: Salesforce/blip-image-captioning-small
- **量化**: 4-bit NF4量化

## 配置说明

配置文件 `mini_dialog_ai/config.py`:

```python
class ModelConfig:
    MODEL_NAME = "Qwen/Qwen2-MoE-2.7B-Instruct"
    IMAGE_MODEL_NAME = "Salesforce/blip-image-captioning-small"
    DEVICE = "cpu"
    MAX_LENGTH = 512
    TEMPERATURE = 0.7
    TOP_P = 0.9
    BATCH_SIZE = 2
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    GRADIENT_ACCUMULATION_STEPS = 4
    QUANTIZATION_BITS = 4
    USE_MOE = True
```

## 许可证

MIT License