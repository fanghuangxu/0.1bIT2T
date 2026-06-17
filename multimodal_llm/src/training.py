"""
训练脚本
支持DPO (Direct Preference Optimization) 训练
适配2GB显存约束
"""

import os
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

from .configs.config import (
    MODEL_CONFIG,
    REWARD_CONFIG,
    DIALOGUE_CONFIG,
    ROLE_CONFIG,
    TRAINING_CONFIG,
    HARDWARE_CONFIG,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_quantization: bool = True
    quantization_bits: int = 4
    
    # LoRA配置
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # 训练配置
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 500
    max_grad_norm: float = 1.0
    
    # 数据配置
    train_data_path: str = "./data/train"
    eval_data_path: str = "./data/eval"
    max_length: int = 1024
    
    # 设备配置
    device: str = "auto"
    fp16: bool = True
    bf16: bool = False


class PreferenceDataset(Dataset):
    """
    偏好数据集
    
    格式: [{"prompt": str, "chosen": str, "rejected": str}, ...]
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据"""
        import json
        
        data = []
        if os.path.isfile(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        elif os.path.isdir(data_path):
            for filename in os.listdir(data_path):
                if filename.endswith(".json") or filename.endswith(".jsonl"):
                    filepath = os.path.join(data_path, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        for line in f:
                            data.append(json.loads(line))
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # Tokenize
        chosen_enc = self.tokenizer(
            prompt + chosen,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        rejected_enc = self.tokenizer(
            prompt + rejected,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(),
            "prompt_len": len(self.tokenizer.encode(prompt)),
        }


def create_model_and_tokenizer(config: TrainingConfig):
    """创建模型和分词器"""
    logger.info(f"加载模型: {config.model_name}")
    
    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 量化配置
    bnb_config = None
    if config.use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("启用4bit量化 (QLoRA)")
    
    # 模型
    model_kwargs = {
        "trust_remote_code": True,
        "quantization_config": bnb_config,
    }
    
    if config.fp16:
        model_kwargs["torch_dtype"] = torch.float16
    elif config.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    if config.device == "auto":
        model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    # 应用LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def compute_dpo_loss(
    model,
    chosen_input_ids,
    chosen_attention_mask,
    rejected_input_ids,
    rejected_attention_mask,
    prompt_len,
):
    """
    计算DPO损失
    
    DPO损失:
    L = - log(sigmoid(log(chosen_prob / rejected_prob)))
    """
    # 获取logits
    chosen_logits = model(
        input_ids=chosen_input_ids,
        attention_mask=chosen_attention_mask,
    ).logits
    
    rejected_logits = model(
        input_ids=rejected_input_ids,
        attention_mask=rejected_attention_mask,
    ).logits
    
    # 计算对数概率
    # (简化实现，实际需要更复杂的处理)
    chosen_logps = torch.gather(
        chosen_logits,
        dim=-1,
        index=chosen_input_ids.unsqueeze(-1),
    ).squeeze(-1)
    
    rejected_logps = torch.gather(
        rejected_logits,
        dim=-1,
        index=rejected_input_ids.unsqueeze(-1),
    ).squeeze(-1)
    
    # 计算差值
    diff = (chosen_logps - rejected_logps).mean()
    
    # DPO损失
    loss = -torch.log(torch.sigmoid(diff))
    
    return loss


def train(
    config: TrainingConfig,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
):
    """执行DPO训练"""
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 创建模型和分词器
    model, tokenizer = create_model_and_tokenizer(config)
    
    # 数据集
    train_dataset = PreferenceDataset(
        train_data_path,
        tokenizer,
        config.max_length,
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    
    eval_dataset = None
    if eval_data_path and os.path.exists(eval_data_path):
        eval_dataset = PreferenceDataset(
            eval_data_path,
            tokenizer,
            config.max_length,
        )
        logger.info(f"评估集大小: {len(eval_dataset)}")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        bf16=config.bf16,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",  # 优化器内存优化
        lr_scheduler_type="cosine",
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_loss=compute_dpo_loss,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    logger.info("开始DPO训练...")
    trainer.train()
    
    # 保存模型
    logger.info(f"保存模型到 {config.output_dir}")
    trainer.save_model(os.path.join(config.output_dir, "final_model"))
    
    return model


def main():
    parser = argparse.ArgumentParser(description="DPO训练脚本")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_data", type=str, default="./data/dpo_train.jsonl")
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_quantization", action="store_true", default=True)
    parser.add_argument("--no_quantization", dest="use_quantization", action="store_false")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_name=args.model_name,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_quantization=args.use_quantization,
    )
    
    train(config, args.train_data, args.eval_data)


if __name__ == "__main__":
    main()
