"""
主模型接口
整合奖励机制、对话管理、角色扮演系统
支持Markdown输出和EOS终止符控制
"""

import os
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import re

# 模型加载器 (兼容多种后端)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import gguf
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

from .rewards.reward import RewardCalculator, create_reward_calculator
from .dialogue.dialogue_manager import (
    DialogueManager, 
    MessageRole, 
    create_dialogue_manager
)
from .roles.role_system import (
    RoleManager, 
    RolePlayEngine,
    create_role_manager,
    create_role_play_engine
)


@dataclass
class GenerationResult:
    """生成结果"""
    text: str
    stopped_at_eos: bool
    stopped_naturally: bool
    rewards: Dict[str, float]
    metadata: Dict


class ModelInterface:
    """
    统一的模型接口
    
    功能:
    - 多后端支持 (Transformers, GGUF)
    - 4bit/8bit量化
    - Markdown格式输出
    - EOS终止符控制
    - 多轮对话
    - 角色扮演
    - RL奖励计算
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config.get("MODEL_CONFIG", {})
        self.reward_config = config.get("REWARD_CONFIG", {})
        self.dialogue_config = config.get("DIALOGUE_CONFIG", {})
        self.role_config = config.get("ROLE_CONFIG", {})
        
        # 设备配置
        self.device = self._get_device()
        
        # 模型和分词器
        self.model = None
        self.tokenizer = None
        
        # 子系统初始化
        self.reward_calculator = create_reward_calculator(self.reward_config)
        self.dialogue_manager = create_dialogue_manager(self.dialogue_config)
        self.role_manager = create_role_manager(self.role_config)
        self.role_engine = create_role_play_engine(
            self.dialogue_manager, 
            self.role_manager
        )
        
        # 生成配置
        self.eos_token_id = None
        self.pad_token_id = None
        
    def _get_device(self) -> str:
        """获取可用设备"""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _load_quantization_config(self):
        """加载量化配置"""
        if not TRANSFORMERS_AVAILABLE:
            return None
            
        quant_config = self.model_config.get("quantization", {})
        if not quant_config.get("enabled", False):
            return None
            
        bits = quant_config.get("bits", 4)
        
        if bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        return None
    
    def load_model(
        self, 
        model_path: Optional[str] = None,
        use_quantization: bool = True
    ):
        """
        加载模型
        
        Args:
            model_path: 模型路径或HuggingFace模型ID
            use_quantization: 是否使用量化
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers库未安装")
        
        model_name = model_path or self.model_config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
        
        print(f"正在加载模型: {model_name}")
        print(f"设备: {self.device}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.pad_token_id = self.tokenizer.pad_token_id
        
        self.eos_token_id = self.tokenizer.eos_token_id
        
        # 量化配置
        quantization_config = None
        if use_quantization:
            quantization_config = self._load_quantization_config()
        
        # 加载模型
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            print("启用4bit量化 (QLoRA)")
        
        if self.device == "cuda" and not quantization_config:
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 打印模型大小信息
        self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        if self.model is None:
            return
            
        # 计算参数数量
        param_count = sum(p.numel() for p in self.model.parameters())
        
        # 估算模型大小
        param_size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        param_size_gb = param_size_bytes / (1024 ** 3)
        
        print(f"\n模型信息:")
        print(f"  参数数量: {param_count / 1e9:.2f}B")
        print(f"  模型大小 (FP16): {param_size_gb:.2f} GB")
        print(f"  设备: {self.device}")
    
    def generate(
        self,
        prompt: Union[str, List[Dict]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        stop_strings: Optional[List[str]] = None,
        returnRewards: bool = True,
    ) -> GenerationResult:
        """
        生成文本
        
        Args:
            prompt: 输入提示词 (字符串或消息列表)
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus采样参数
            top_k: top-k采样参数
            repeat_penalty: 重复惩罚
            stop_strings: 停止字符串列表
            returnRewards: 是否计算奖励
            
        Returns:
            GenerationResult: 生成结果
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载")
        
        # 默认参数
        max_new_tokens = max_new_tokens or self.model_config.get("max_new_tokens", 512)
        temperature = temperature or self.model_config.get("temperature", 0.7)
        top_p = top_p or self.model_config.get("top_p", 0.9)
        top_k = top_k or self.model_config.get("top_k", 50)
        repeat_penalty = repeat_penalty or self.model_config.get("repeat_penalty", 1.1)
        
        # 处理输入
        if isinstance(prompt, str):
            input_text = prompt
        elif isinstance(prompt, list):
            # 消息列表格式
            input_text = self._format_messages(prompt)
        else:
            input_text = str(prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成参数
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": temperature > 0,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
        }
        
        # 如果设置了重复惩罚
        if repeat_penalty != 1.0:
            generation_kwargs["repetition_penalty"] = repeat_penalty
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        
        # 解码
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        # 检查停止条件
        stopped_at_eos = outputs[0][-1].item() == self.eos_token_id
        stopped_naturally = True
        
        # 检查自定义停止字符串
        if stop_strings:
            for stop_str in stop_strings:
                if stop_str in generated_text:
                    generated_text = generated_text.split(stop_str)[0]
                    stopped_naturally = False
                    break
        
        # 计算奖励
        rewards = {}
        if returnRewards:
            rewards = self.reward_calculator.calculate_total_reward(
                text=generated_text,
                prompt=input_text,
                stopped_at_eos=stopped_at_eos,
                stopped_naturally=stopped_naturally,
            )
        
        return GenerationResult(
            text=generated_text,
            stopped_at_eos=stopped_at_eos,
            stopped_naturally=stopped_naturally,
            rewards=rewards,
            metadata={
                "model": self.model_config.get("model_name"),
                "device": self.device,
                "generated_length": len(generated_text),
            },
        )
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """格式化消息列表为提示词"""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"<|{role}|>\n{content}")
        return "\n\n".join(formatted)
    
    def chat(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        role_id: Optional[str] = None,
        max_history_turns: Optional[int] = None,
        **generation_kwargs
    ) -> GenerationResult:
        """
        对话接口 (多轮对话)
        
        Args:
            user_input: 用户输入
            system_prompt: 系统提示词 (可选)
            role_id: 角色ID (可选)
            max_history_turns: 最大历史轮次
            **generation_kwargs: 生成参数
            
        Returns:
            GenerationResult: 生成结果
        """
        # 添加用户消息
        self.dialogue_manager.add_user_message(user_input)
        
        # 如果指定了角色
        if role_id and not self.role_engine.is_role_playing:
            self.role_engine.start_role_play(role_id)
        
        # 构建提示词
        prompt = self.dialogue_manager.build_prompt(
            include_system=system_prompt is not None,
            max_history_turns=max_history_turns
        )
        
        # 如果有自定义系统提示词
        if system_prompt:
            prompt = f"<|system|>\n{system_prompt}\n\n{prompt}"
        
        # 生成回复
        result = self.generate(prompt, **generation_kwargs)
        
        # 添加助手回复到历史
        self.dialogue_manager.add_assistant_message(result.text)
        
        # 检查是否需要压缩记忆
        if self.dialogue_manager.should_compress():
            self.dialogue_manager.compress_memory()
        
        return result
    
    def format_markdown(self, text: str) -> str:
        """
        格式化文本为Markdown
        
        支持:
        - 代码块
        - 列表
        - 标题
        - 链接
        - 图片
        """
        # 这个方法主要是在输出时调用
        # 模型生成的内容如果是Markdown，会被正确渲染
        return text
    
    def reset_conversation(self):
        """重置对话"""
        self.dialogue_manager.reset()
        if self.role_engine.is_role_playing:
            self.role_engine.end_role_play()
    
    def get_conversation_history(self) -> List[Dict]:
        """获取对话历史"""
        messages = self.dialogue_manager.current_state.messages
        return [msg.to_dict() for msg in messages]
    
    def get_active_roles(self) -> List[str]:
        """获取当前活跃角色"""
        return [role.id for role in self.role_manager.active_roles]
    
    def list_available_roles(self) -> List[Dict]:
        """列出可用角色"""
        return self.role_manager.list_roles()
    
    def switch_role(self, role_id: str) -> bool:
        """切换角色"""
        return self.role_engine.switch_role(role_id)
    
    def end_role_play(self):
        """结束角色扮演"""
        self.role_engine.end_role_play()
    
    def get_reward_breakdown(self, text: str, prompt: str) -> Dict[str, float]:
        """获取奖励分解"""
        rewards = self.reward_calculator.calculate_total_reward(
            text=text,
            prompt=prompt,
            stopped_at_eos=True,
            stopped_naturally=True,
        )
        return rewards


class CPUModelInterface(ModelInterface):
    """
    CPU优化模型接口
    
    额外的CPU优化:
    - 更好的内存管理
    - 量化优化
    - 批处理优化
    """
    
    def _get_device(self) -> str:
        """强制使用CPU"""
        return "cpu"
    
    def load_model(self, model_path: Optional[str] = None):
        """加载CPU优化模型"""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers库未安装")
        
        model_name = model_path or self.model_config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
        
        print(f"正在加载模型 (CPU模式): {model_name}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,  # CPU上use_fast=False可能更快
        )
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.pad_token_id = self.tokenizer.pad_token_id
        
        self.eos_token_id = self.tokenizer.eos_token_id
        
        # CPU加载 (FP32以节省内存)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        
        self.model.eval()
        self._print_model_info()


class QuantizedModelInterface(ModelInterface):
    """
    量化模型接口 (GGUF格式)
    
    用于加载预量化的GGUF模型
    """
    
    def load_model(self, model_path: str):
        """加载GGUF格式的量化模型"""
        if not GGUF_AVAILABLE:
            raise RuntimeError("需要安装gguf库来加载量化模型")
        
        print(f"正在加载GGUF模型: {model_path}")
        
        # TODO: 实现GGUF模型加载
        # 这需要根据具体的GGUF实现来编写
        raise NotImplementedError("GGUF加载待实现")


# 工厂函数
def create_model_interface(
    config: Dict,
    mode: str = "auto"
) -> ModelInterface:
    """
    创建模型接口
    
    Args:
        config: 配置字典
        mode: 运行模式 ("auto", "cpu", "quantized")
        
    Returns:
        ModelInterface: 模型接口实例
    """
    if mode == "cpu":
        return CPUModelInterface(config)
    elif mode == "quantized":
        return QuantizedModelInterface(config)
    else:
        return ModelInterface(config)


# 便捷函数
def quick_chat(
    model_path: Optional[str] = None,
    message: str = "你好",
    system_prompt: Optional[str] = None,
    device: str = "auto"
) -> str:
    """
    快速对话 (单轮)
    
    这是一个便捷函数，用于快速测试
    """
    # 加载默认配置
    from .configs.config import MODEL_CONFIG, REWARD_CONFIG, DIALOGUE_CONFIG, ROLE_CONFIG
    
    config = {
        "MODEL_CONFIG": MODEL_CONFIG,
        "REWARD_CONFIG": REWARD_CONFIG,
        "DIALOGUE_CONFIG": DIALOGUE_CONFIG,
        "ROLE_CONFIG": ROLE_CONFIG,
    }
    
    # 创建接口
    if device == "cpu":
        interface = CPUModelInterface(config)
    else:
        interface = ModelInterface(config)
    
    # 加载模型
    interface.load_model(model_path)
    
    # 对话
    if system_prompt:
        result = interface.chat(message, system_prompt=system_prompt)
    else:
        result = interface.chat(message)
    
    return result.text
