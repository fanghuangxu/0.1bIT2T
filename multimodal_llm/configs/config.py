"""
多语言对话AI系统配置
支持中文、英文、德文，适配2GB显存推理
"""

MODEL_CONFIG = {
    # 基础模型选择 - Qwen2.5-1.5B 量化后约1.1GB，支持29种语言
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "model_type": "causal_lm",
    
    # 量化配置 - 4bit量化可在2GB显存运行
    "quantization": {
        "enabled": True,
        "bits": 4,
        "method": "gptq",  # 或 "awq", "gguf"
    },
    
    # 推理设备
    "device": "auto",  # 自动选择cuda/cpu
    
    # 上下文配置
    "max_context_length": 2048,
    "max_new_tokens": 512,
    
    # 生成配置
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repeat_penalty": 1.1,
}

# 数据集配置 - 使用开源多语言数据集
DATASET_CONFIG = {
    # 预训练数据 (用于模型能力提升)
    "pretrain": {
        "sources": [
            # FineWeb-Edu - 高质量教育内容
            "https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/main/FineWeb-edu",
            # DCLM - 通用语言模型训练数据
            "https://huggingface.co/datasets/mlfoundations/dclm-corpus",
            # The Stack - 代码数据
            "https://huggingface.co/datasets/bigcode/the-stack",
        ],
        "languages": ["zh", "en", "de"],  # 中文、英文、德文
    },
    
    # 指令微调数据 (用于对话和角色扮演)
    "sft": {
        "sources": [
            # 多语言指令数据
            "https://huggingface.co/datasets/openbmb/UltraFeedback",
            "https://huggingface.co/datasets/thome/RolePlay-GCG",  # 角色扮演数据
        ],
    },
    
    # RLHF奖励模型训练数据
    "reward": {
        "preference_data": "https://huggingface.co/datasets/anthropic/hh-rlhf",
        "multilingual": "https://huggingface.co/datasets/BigSalmon/chunked-adapter-ultrachat_zh",  # 中文对话
    },
}

# 奖励机制配置
REWARD_CONFIG = {
    # 长度奖励 - 鼓励适当长度的完整回答
    "length": {
        "enabled": True,
        "min_length": 50,         # 最小有效长度
        "optimal_length": 300,    # 最优长度
        "max_length": 1024,       # 最大长度惩罚
        "weight": 0.1,            # 奖励权重
    },
    
    # 内容正确性奖励
    "correctness": {
        "enabled": True,
        "coherence_weight": 0.3,   # 连贯性权重
        "factuality_weight": 0.4,  # 事实性权重
        "safety_weight": 0.3,      # 安全性权重
    },
    
    # EOS控制奖励 - 鼓励模型自然停止
    "eos": {
        "enabled": True,
        "natural_stop_bonus": 0.5,      # 自然停止奖励
        "forced_stop_penalty": -0.3,     # 强制停止惩罚
        "max_length_stop_bonus": 0.2,   # 达到最大长度自然停止
    },
}

# 对话系统配置
DIALOGUE_CONFIG = {
    "max_turns": 20,              # 最大对话轮次
    "system_prompt": "你是一个人工智能助手，名为小问。你能够用中文、英文和德文进行流畅对话。你乐于助人、知识渊博、友善亲切。",
    "context_window": 2048,       # 上下文窗口大小
    "memory_compression": True,   # 记忆压缩
}

# 角色扮演配置
ROLE_CONFIG = {
    "enabled": True,
    "preset_roles": [
        {
            "id": "assistant",
            "name": "AI助手",
            "description": "通用AI助手角色",
            "prompt": "你是一个乐于助人的AI助手。"
        },
        {
            "id": "teacher",
            "name": "老师",
            "description": "教育者角色，可以教授各种学科",
            "prompt": "你是一位经验丰富的老师，擅长用浅显易懂的方式讲解知识。"
        },
        {
            "id": "translator",
            "name": "翻译官",
            "description": "专业翻译，支持中英德互译",
            "prompt": "你是一位专业翻译，精通中文、英文和德文。"
        },
    ],
}

# 训练配置
TRAINING_CONFIG = {
    "method": "dpo",  # DPO (Direct Preference Optimization)
    "batch_size": 4,
    "learning_rate": 1e-6,
    "epochs": 3,
    "gradient_accumulation": 4,
    
    # LoRA微调配置 - 大幅降低显存需求
    "lora": {
        "enabled": True,
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    },
    
    # 量化训练配置
    "quantization_training": {
        "enabled": True,
        "qlora": True,  # QLoRA - 4bit量化+LoRA
    },
}

# 硬件要求
HARDWARE_CONFIG = {
    "min_vram_gb": 2,           # 最小显存需求
    "recommended_vram_gb": 4,  # 推荐显存
    "cpu_support": True,        # 支持CPU推理
    "optimizations": [
        "flash_attention",
        "gradient_checkpointing",
        "cpu_offload",
    ],
}
