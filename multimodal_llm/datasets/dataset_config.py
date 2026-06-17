"""
数据集配置文件
指定多语言训练数据集 (中文、英文、德文)
适配2GB显存约束
"""

DATASETS = {
    # ========== 预训练数据集 ==========
    "pretrain": {
        "description": "用于基础语言模型预训练",
        "languages": ["zh", "en", "de"],
        
        "sources": {
            # FineWeb-Edu - 高质量教育内容，模型学习效果最好
            "fineweb_edu": {
                "url": "https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/main/FineWeb-edu",
                "size_gb": 140,
                "quality": "high",
                "languages": ["en"],
                "use_case": "教育内容、解释性文本",
                "split": "train",
            },
            
            # DCLM - 通用高质量语料
            "dclm": {
                "url": "https://huggingface.co/datasets/mlfoundations/dclm-corpus",
                "size_gb": 800,
                "quality": "medium-high",
                "languages": ["en", "de"],
                "use_case": "通用文本",
                "split": "train",
            },
            
            # The Stack - 代码数据
            "the_stack": {
                "url": "https://huggingface.co/datasets/bigcode/the-stack",
                "size_gb": 800,
                "quality": "medium-high",
                "languages": ["en", "de", "zh"],
                "use_case": "代码、函数文档",
                "note": "包含代码注释中的多语言",
                "split": "data",
            },
            
            # SkyPile - 中文预训练数据
            "skypile": {
                "url": "https://huggingface.co/datasets/SimShang/SkyPile-1.5B",
                "size_gb": 100,
                "quality": "medium",
                "languages": ["zh"],
                "use_case": "中文文本",
                "note": "用于增强中文能力",
                "split": "train",
            },
            
            # Wikipedia - 多语言百科
            "wikipedia": {
                "url": "https://huggingface.co/datasets/wikimedia/wikipedia",
                "size_gb": 20,
                "quality": "high",
                "languages": ["zh", "en", "de"],
                "use_case": "高质量事实性文本",
                "split": "train",
            },
        },
    },
    
    # ========== 指令微调数据集 ==========
    "sft": {
        "description": "用于监督微调，提升指令跟随能力",
        
        "sources": {
            # UltraFeedback - 高质量反馈数据
            "ultrafeedback": {
                "url": "https://huggingface.co/datasets/openbmb/UltraFeedback",
                "size_gb": 2,
                "quality": "high",
                "languages": ["en", "zh"],
                "use_case": "指令跟随、对话",
                "split": "train",
            },
            
            # RolePlay-GCG - 角色扮演数据
            "roleplay": {
                "url": "https://huggingface.co/datasets/thome/RolePlay-GCG",
                "size_gb": 1,
                "quality": "medium-high",
                "languages": ["zh", "en"],
                "use_case": "角色扮演、创意写作",
                "split": "train",
            },
            
            # OpenOrca - 多语言推理数据
            "openorca": {
                "url": "https://huggingface.co/datasets/BigSalmon/chunked-adapter-ultrachat_zh",
                "size_gb": 10,
                "quality": "medium-high",
                "languages": ["en", "zh"],
                "use_case": "推理、解释",
                "split": "default",
            },
            
            # alpaca-zh - 中文指令数据
            "alpaca_zh": {
                "url": "https://huggingface.co/datasets/yahma/alpaca-gpt4-data-zh",
                "size_gb": 0.5,
                "quality": "medium",
                "languages": ["zh"],
                "use_case": "中文指令",
                "split": "train",
            },
            
            # oast - 德语指令数据
            "oast": {
                "url": "https://huggingface.co/datasets/Pankaj01Fuloria/oast",
                "size_gb": 0.1,
                "quality": "medium",
                "languages": ["de"],
                "use_case": "德语指令",
                "split": "train",
            },
        },
    },
    
    # ========== RLHF/DPO 偏好数据集 ==========
    "preference": {
        "description": "用于RLHF/DPO训练，构建偏好模型",
        
        "sources": {
            # HH-RLHF - 人类偏好数据
            "hh_rlhf": {
                "url": "https://huggingface.co/datasets/anthropic/hh-rlhf",
                "size_gb": 1,
                "quality": "very_high",
                "languages": ["en"],
                "use_case": "偏好学习、安全性",
                "split": "train",
            },
            
            # PKU-Alignment - 中文偏好数据
            "pku_align": {
                "url": "https://huggingface.co/datasets/PKU-Alignment/PKU-Alignment-SFT",
                "size_gb": 1,
                "quality": "high",
                "languages": ["zh", "en"],
                "use_case": "中文偏好、安全性",
                "split": "sft",
            },
            
            # UltraFeedback-Binary - 二进制偏好
            "ultrafeedback_binary": {
                "url": "https://huggingface.co/datasets/Anthropic/hh-rlhf",
                "size_gb": 0.5,
                "quality": "high",
                "languages": ["en"],
                "use_case": "偏好排序",
                "split": "train",
            },
        },
    },
    
    # ========== 多语言专项数据集 ==========
    "multilingual": {
        "description": "多语言能力专项训练",
        
        "sources": {
            # MultiHPLT - 多语言平行语料
            "multiphplt": {
                "url": "https://huggingface.co/datasets/kunishou/hplt",
                "size_gb": 50,
                "quality": "high",
                "languages": ["zh", "en", "de", "fr", "es"],
                "use_case": "多语言翻译、语义理解",
                "note": "包含高质量平行语料",
                "split": "train",
            },
            
            # ParaCrawl - 机器翻译平行语料
            "paracrawl": {
                "url": "https://huggingface.co/datasets/kunishou/hplt",
                "size_gb": 100,
                "quality": "medium",
                "languages": ["en", "de", "zh"],
                "use_case": "翻译",
                "split": "train",
            },
            
            # FLORES-200 - 评估翻译
            "flores200": {
                "url": "https://huggingface.co/datasets/facebook/flores",
                "size_gb": 0.1,
                "quality": "very_high",
                "languages": ["zh", "en", "de", "many"],
                "use_case": "翻译评估、多语言验证",
                "note": "主要用于评估，非训练",
                "split": "dev",
            },
            
            # XNLI - 跨语言自然语言推理
            "xnli": {
                "url": "https://huggingface.co/datasets/facebook/xnli",
                "size_gb": 0.01,
                "quality": "high",
                "languages": ["zh", "en", "de"],
                "use_case": "跨语言推理",
                "split": "test",
            },
        },
    },
    
    # ========== 对话专项数据集 ==========
    "conversation": {
        "description": "对话能力专项训练",
        
        "sources": {
            # ChatML - 对话格式数据
            "chatml": {
                "url": "https://huggingface.co/datasets/PythonPianist/chatml_exemplar",
                "size_gb": 0.5,
                "quality": "high",
                "languages": ["en"],
                "use_case": "对话格式学习",
                "split": "train",
            },
            
            # ShareGPT - 真实对话数据
            "sharegpt": {
                "url": "https://huggingface.co/datasets/Ray777n/clean_sharegpt",
                "size_gb": 2,
                "quality": "medium",
                "languages": ["en", "zh"],
                "use_case": "自然对话",
                "note": "需要清洗",
                "split": "train",
            },
            
            # Camel - 多代理对话
            "camel": {
                "url": "https://huggingface.co/datasets/camel-ai/science",
                "size_gb": 1,
                "quality": "high",
                "languages": ["en"],
                "use_case": "多代理推理对话",
                "split": "train",
            },
        },
    },
}


# 推荐的训练数据组合 (符合2GB显存约束)
RECOMMENDED_TRAINING_MIX = {
    "description": "针对1.5B-3B模型的推荐训练数据组合",
    
    # 预训练阶段
    "pretrain": {
        "total_tokens": "100B",  # 建议训练100B tokens
        "mix": {
            "fineweb_edu": 0.40,      # 40% 英文教育内容
            "skypile": 0.25,         # 25% 中文内容
            "the_stack": 0.15,       # 15% 代码
            "wikipedia": 0.10,       # 10% 多语言百科
            "multiphplt": 0.10,      # 10% 多语言平行
        },
        "estimated_size_gb": 50,  # 压缩后约50GB
    },
    
    # SFT阶段
    "sft": {
        "total_samples": "100K",  # 100K样本
        "mix": {
            "alpaca_zh": 0.35,       # 35% 中文指令
            "ultrafeedback": 0.30,   # 30% 英文反馈
            "roleplay": 0.20,       # 20% 角色扮演
            "openorca": 0.15,       # 15% 推理
        },
        "estimated_size_gb": 5,
    },
    
    # DPO阶段
    "dpo": {
        "total_samples": "50K",   # 50K偏好对
        "mix": {
            "hh_rlhf": 0.50,        # 50% 英文偏好
            "pku_align": 0.50,      # 50% 中文偏好
        },
        "estimated_size_gb": 1,
    },
}


# 数据下载和处理脚本模板
DATA_DOWNLOAD_SCRIPT = '''#!/bin/bash
# 数据集下载脚本

set -e

DATA_DIR="./data"
mkdir -p $DATA_DIR

# 预训练数据
echo "下载预训练数据..."

# FineWeb-Edu
echo "下载 FineWeb-Edu..."
huggingface-cli download HuggingFaceTB/smollm-corpus FineWeb-edu --repo-type dataset --local-dir $DATA_DIR/fineweb_edu

# SkyPile (中文)
echo "下载 SkyPile..."
huggingface-cli download SimShang/SkyPile-1.5B --repo-type dataset --local-dir $DATA_DIR/skypile

# Wikipedia 多语言
echo "下载 Wikipedia..."
huggingface-cli download wikimedia/wikipedia --repo-type dataset --local-dir $DATA_DIR/wikipedia

# SFT数据
echo "下载 SFT 数据..."
huggingface-cli download yahma/alpaca-gpt4-data-zh --repo-type dataset --local-dir $DATA_DIR/alpaca_zh
huggingface-cli download openbmb/UltraFeedback --repo-type dataset --local-dir $DATA_DIR/ultrafeedback

# DPO数据
echo "下载 DPO 数据..."
huggingface-cli download anthropic/hh-rlhf --repo-type dataset --local-dir $DATA_DIR/hh_rlhf
huggingface-cli download PKU-Alignment/PKU-Alignment-SFT --repo-type dataset --local-dir $DATA_DIR/pku_align

echo "下载完成!"
'''

# 数据处理脚本
DATA_PROCESSING_SCRIPT = '''
import json
from datasets import load_dataset
from transformers import AutoTokenizer

def process_sample(sample, tokenizer, max_length=2048):
    """处理单个样本"""
    text = sample.get("text", "")
    
    # Tokenize
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    }

def prepare_multilingual_dataset(
    dataset_name,
    output_file,
    tokenizer,
    languages=["zh", "en", "de"],
):
    """准备多语言数据集"""
    
    # 加载数据集
    if dataset_name == "alpaca_zh":
        dataset = load_dataset("yahma/alpaca-gpt4-data-zh", split="train")
    elif dataset_name == "ultrafeedback":
        dataset = load_dataset("openbmb/UltraFeedback", split="train")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 过滤语言
    # ... (根据数据集结构进行过滤)
    
    # 处理并保存
    processed = []
    for sample in dataset:
        processed.append(process_sample(sample, tokenizer))
        
        if len(processed) % 10000 == 0:
            print(f"处理了 {len(processed)} 样本")
    
    # 保存为JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\\n")
    
    print(f"保存了 {len(processed)} 样本到 {output_file}")

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    
    prepare_multilingual_dataset(
        "alpaca_zh",
        "./data/processed/alpaca_zh.jsonl",
        tokenizer,
    )
'''
