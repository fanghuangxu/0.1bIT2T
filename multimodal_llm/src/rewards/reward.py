"""
奖励机制模块
实现长度奖励、内容正确性奖励、EOS终止符奖励
"""

import re
from typing import Dict, List, Tuple, Optional
import torch


class RewardCalculator:
    """
    奖励计算器
    
    支持:
    - 长度奖励: 鼓励适当长度的完整回答
    - 内容正确性奖励: 连贯性、事实性、安全性
    - EOS奖励: 自然停止vs强制停止
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.length_config = config.get("length", {})
        self.correctness_config = config.get("correctness", {})
        self.eos_config = config.get("eos", {})
        
        # 安全词列表
        self.safety_patterns = [
            r"我不知道",
            r"抱歉",
            r"对不起",
            r"无法.*",
            r"不能.*",
        ]
        
    def calculate_length_reward(self, text: str) -> float:
        """
        计算长度奖励
        
        逻辑:
        - 短于min_length: 惩罚
        - 在optimal_length附近: 奖励
        - 长于max_length: 惩罚
        """
        if not self.length_config.get("enabled", False):
            return 0.0
            
        length = len(text)
        min_len = self.length_config.get("min_length", 50)
        optimal_len = self.length_config.get("optimal_length", 300)
        max_len = self.length_config.get("max_length", 1024)
        weight = self.length_config.get("weight", 0.1)
        
        if length < min_len:
            # 过短惩罚
            return weight * (length / min_len - 1)
        elif length <= optimal_len:
            # 递增奖励
            return weight * (length / optimal_len)
        elif length <= max_len:
            # 递减奖励
            decay = (max_len - length) / (max_len - optimal_len)
            return weight * (1 - 0.3 * decay)
        else:
            # 过长惩罚
            return -weight * 0.5
    
    def calculate_correctness_reward(
        self, 
        text: str, 
        prompt: str,
        context: Optional[List[str]] = None
    ) -> float:
        """
        计算内容正确性奖励
        
        包含:
        - 连贯性: 与对话历史的一致性
        - 事实性: 基础逻辑和常识
        - 安全性: 不包含有害内容
        """
        if not self.correctness_config.get("enabled", False):
            return 0.0
            
        coherence_weight = self.correctness_config.get("coherence_weight", 0.3)
        factuality_weight = self.correctness_config.get("factuality_weight", 0.4)
        safety_weight = self.correctness_config.get("safety_weight", 0.3)
        
        # 1. 连贯性奖励
        coherence_reward = self._calculate_coherence(text, prompt, context)
        
        # 2. 事实性奖励
        factuality_reward = self._calculate_factuality(text)
        
        # 3. 安全性奖励
        safety_reward = self._calculate_safety(text)
        
        total = (
            coherence_weight * coherence_reward +
            factuality_weight * factuality_reward +
            safety_weight * safety_reward
        )
        
        return total
    
    def _calculate_coherence(
        self, 
        text: str, 
        prompt: str,
        context: Optional[List[str]] = None
    ) -> float:
        """
        计算连贯性奖励
        检查回答是否与问题相关
        """
        if not text or not prompt:
            return 0.0
        
        # 简单检查: 回答中是否包含问题的关键词
        prompt_keywords = set(re.findall(r'\w+', prompt.lower()))
        text_keywords = set(re.findall(r'\w+', text.lower()))
        
        if not prompt_keywords:
            return 0.5
            
        overlap = len(prompt_keywords & text_keywords)
        ratio = overlap / len(prompt_keywords)
        
        # 奖励相关度高的回答
        if ratio > 0.3:
            return min(1.0, ratio + 0.3)
        else:
            return max(0.0, ratio - 0.2)
    
    def _calculate_factuality(self, text: str) -> float:
        """
        计算事实性奖励
        基于基础语言质量指标
        """
        if not text:
            return 0.0
            
        score = 0.5  # 基础分
        
        # 检查是否包含"我不知道"等不确定表达 (轻微惩罚)
        uncertainty_patterns = [
            r"可能", r"也许", r"大概", r"可能吧",
            r"不确定", r"不清楚", r"不知道"
        ]
        
        uncertainty_count = sum(
            len(re.findall(pat, text)) 
            for pat in uncertainty_patterns
        )
        
        if uncertainty_count > 3:
            score -= 0.2 * min(uncertainty_count - 3, 5)
        
        # 检查是否有完整句子
        sentences = re.split(r'[.!?。！？]', text)
        complete_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(complete_sentences) >= 2:
            score += 0.2
        elif len(complete_sentences) == 1 and len(text) > 100:
            score += 0.1
        
        # 检查是否乱码 (中文/英文混合检查)
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text)
        
        if total_chars > 0:
            chinese_ratio = chinese_chars / total_chars
            
            # 如果全是乱码或全是某种语言但异常，惩罚
            if chinese_ratio > 0.95 or chinese_ratio < 0.01:
                # 可能是纯某种语言文本，不惩罚
                pass
            elif 0.1 < chinese_ratio < 0.9:
                # 混合语言，检查是否有交替乱码
                if self._has_garbled(text):
                    score -= 0.5
        
        return max(0.0, min(1.0, score))
    
    def _has_garbled(self, text: str) -> bool:
        """
        检测文本是否可能是乱码
        """
        # 检测异常字符比例
        normal_chars = len(re.findall(r'[\w\s\u4e00-\u9fff\u00-\u024f]', text))
        total_chars = len(text)
        
        if total_chars == 0:
            return False
            
        abnormal_ratio = 1 - (normal_chars / total_chars)
        
        # 异常字符超过30%可能是乱码
        return abnormal_ratio > 0.3
    
    def _calculate_safety(self, text: str) -> float:
        """
        计算安全性奖励
        """
        if not text:
            return 0.0
            
        # 检查是否包含安全拒绝回复
        for pattern in self.safety_patterns:
            if re.search(pattern, text):
                return 0.5  # 安全的拒绝回复
        
        # 检查是否包含明显有害内容 (简单检测)
        harmful_patterns = [
            r"如何制造.*武器",
            r"如何.*攻击.*人",
            r"密码.*破解",
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, text):
                return 0.0  # 危险内容
        
        return 0.8  # 正常安全内容
    
    def calculate_eos_reward(
        self, 
        stopped_at_eos: bool, 
        stopped_naturally: bool,
        length: int,
        max_length: int
    ) -> float:
        """
        计算EOS终止符奖励
        
        逻辑:
        - 自然遇到EOS停止: 高奖励
        - 达到最大长度自然停止: 中等奖励
        - 被强制截断: 惩罚
        """
        if not self.eos_config.get("enabled", False):
            return 0.0
            
        natural_bonus = self.eos_config.get("natural_stop_bonus", 0.5)
        forced_penalty = self.eos_config.get("forced_stop_penalty", -0.3)
        max_length_bonus = self.eos_config.get("max_length_stop_bonus", 0.2)
        
        if stopped_at_eos and stopped_naturally:
            # 自然遇到EOS停止 - 最佳情况
            return natural_bonus
        elif length >= max_length * 0.9 and stopped_naturally:
            # 接近最大长度自然停止
            return max_length_bonus
        else:
            # 强制截断
            return forced_penalty
    
    def calculate_total_reward(
        self,
        text: str,
        prompt: str,
        stopped_at_eos: bool,
        stopped_naturally: bool,
        context: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        计算总奖励 (供RL训练使用)
        """
        max_length = 1024  # 可配置
        
        length_reward = self.calculate_length_reward(text)
        correctness_reward = self.calculate_correctness_reward(text, prompt, context)
        eos_reward = self.calculate_eos_reward(
            stopped_at_eos, 
            stopped_naturally,
            len(text),
            max_length
        )
        
        total = length_reward + correctness_reward + eos_reward
        
        return {
            "total": total,
            "length_reward": length_reward,
            "correctness_reward": correctness_reward,
            "eos_reward": eos_reward,
        }


class RewardCache:
    """
    奖励缓存 - 用于加速批量奖励计算
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[float]:
        """获取缓存的奖励值"""
        return self.cache.get(key)
    
    def set(self, key: str, value: float):
        """设置缓存"""
        if len(self.cache) >= self.max_size:
            # LRU淘汰
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()


# 工厂函数
def create_reward_calculator(config: Dict) -> RewardCalculator:
    """创建奖励计算器"""
    return RewardCalculator(config)
