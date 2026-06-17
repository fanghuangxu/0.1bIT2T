"""
多轮对话管理器
支持上下文记忆、记忆压缩
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import copy


class MessageRole(Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    OBSERVER = "observer"  # 用于观察模式的角色


@dataclass
class Message:
    """对话消息"""
    role: MessageRole
    content: str
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        return cls(
            role=MessageRole(data.get("role", "user")),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DialogueState:
    """对话状态"""
    messages: List[Message] = field(default_factory=list)
    current_role: Optional[str] = None
    role_context: Dict = field(default_factory=dict)
    turn_count: int = 0
    
    def add_message(self, role: MessageRole, content: str, metadata: Dict = None):
        """添加消息"""
        self.messages.append(Message(
            role=role,
            content=content,
            metadata=metadata or {}
        ))
        if role == MessageRole.USER:
            self.turn_count += 1
    
    def get_history(self, max_turns: Optional[int] = None) -> List[Message]:
        """获取对话历史"""
        if max_turns is None:
            return copy.deepcopy(self.messages)
        
        # 只返回最近N轮对话
        user_turns = 0
        start_idx = 0
        for i, msg in enumerate(self.messages):
            if msg.role == MessageRole.USER:
                user_turns += 1
                if user_turns > max_turns:
                    start_idx = i
                    break
        
        return copy.deepcopy(self.messages[start_idx:])


class DialogueManager:
    """
    多轮对话管理器
    
    功能:
    - 维护对话上下文
    - 支持多轮记忆
    - 记忆压缩
    - 角色上下文管理
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_turns = config.get("max_turns", 20)
        self.context_window = config.get("context_window", 2048)
        self.memory_compression = config.get("memory_compression", True)
        
        # 当前对话状态
        self.current_state = DialogueState()
        
        # 系统提示词
        self.system_prompt = config.get("system_prompt", "")
        
        # 角色扮演上下文
        self.role_context: Dict[str, str] = {}
        
    def add_user_message(self, content: str, metadata: Dict = None):
        """添加用户消息"""
        self.current_state.add_message(
            MessageRole.USER, 
            content, 
            metadata
        )
        
    def add_assistant_message(self, content: str, metadata: Dict = None):
        """添加助手消息"""
        self.current_state.add_message(
            MessageRole.ASSISTANT, 
            content, 
            metadata
        )
    
    def add_system_message(self, content: str, metadata: Dict = None):
        """添加系统消息"""
        self.current_state.add_message(
            MessageRole.SYSTEM, 
            content, 
            metadata
        )
    
    def set_role_context(self, role_id: str, role_prompt: str):
        """设置角色扮演上下文"""
        self.role_context[role_id] = role_prompt
        
    def clear_role_context(self):
        """清除角色上下文"""
        self.role_context = {}
    
    def get_history(self, max_turns: Optional[int] = None) -> List[Message]:
        """获取对话历史"""
        return self.current_state.get_history(max_turns)
    
    def build_prompt(
        self, 
        include_system: bool = True,
        max_history_turns: Optional[int] = None
    ) -> str:
        """
        构建用于推理的完整提示词
        
        格式 (支持Markdown):
        <|system|>
        系统提示词
        <|user|>
        用户消息
        <|assistant|>
        助手回复
        ...
        """
        parts = []
        
        # 系统提示词
        if include_system:
            system_content = self.system_prompt
            
            # 添加角色扮演上下文
            if self.role_context:
                role_desc = "\n".join(
                    f"[{rid}]: {rp}" 
                    for rid, rp in self.role_context.items()
                )
                system_content += f"\n\n角色扮演上下文:\n{role_desc}"
            
            parts.append(f"<|system|>\n{system_content}")
        
        # 对话历史
        history = self.current_state.get_history(max_history_turns)
        
        for msg in history:
            if msg.role == MessageRole.SYSTEM and not include_system:
                continue
                
            role_tag = f"<|{msg.role.value}|>"
            parts.append(f"{role_tag}\n{msg.content}")
        
        return "\n\n".join(parts)
    
    def build_messages_for_api(
        self,
        max_history_turns: Optional[int] = None
    ) -> List[Dict]:
        """
        构建API格式的消息列表
        用于OpenAI兼容API
        """
        messages = []
        
        # 系统消息
        system_content = self.system_prompt
        if self.role_context:
            role_desc = "\n".join(
                f"[{rid}]: {rp}" 
                for rid, rp in self.role_context.items()
            )
            system_content += f"\n\n角色扮演上下文:\n{role_desc}"
        
        if system_content:
            messages.append({
                "role": "system",
                "content": system_content
            })
        
        # 对话历史
        history = self.current_state.get_history(max_history_turns)
        
        for msg in history:
            if msg.role == MessageRole.SYSTEM:
                continue
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        return messages
    
    def compress_memory(self, target_turns: int = 10):
        """
        压缩记忆
        
        策略:
        1. 保留系统提示词
        2. 保留最近N轮对话
        3. 中间的长回复可以缩短
        """
        if not self.memory_compression:
            return
        
        history = self.current_state.get_history()
        
        # 重新构建对话历史
        self.current_state.messages = []
        
        for msg in history:
            if msg.role == MessageRole.SYSTEM:
                continue  # 重新添加
            
            # 压缩过长的助手回复
            if msg.role == MessageRole.ASSISTANT and len(msg.content) > 500:
                # 保留前100和后100字符，中间省略
                compressed = (
                    msg.content[:100] + 
                    f"\n...[已压缩，原始长度: {len(msg.content)}字符]...\n" +
                    msg.content[-100:]
                )
                self.current_state.messages.append(Message(
                    role=msg.role,
                    content=compressed,
                    metadata={**msg.metadata, "compressed": True}
                ))
            else:
                self.current_state.messages.append(msg)
        
        # 保留最新的目标轮数
        user_turns = 0
        final_messages = []
        for msg in reversed(self.current_state.messages):
            if msg.role == MessageRole.USER:
                user_turns += 1
            final_messages.append(msg)
            if user_turns >= target_turns:
                break
        
        self.current_state.messages = list(reversed(final_messages))
    
    def get_context_length(self) -> int:
        """估算当前上下文长度"""
        prompt = self.build_prompt(include_system=True)
        return len(prompt) // 4  # 粗略估算token数
    
    def should_compress(self) -> bool:
        """检查是否需要压缩"""
        return self.get_context_length() > self.context_window * 0.8
    
    def reset(self):
        """重置对话"""
        self.current_state = DialogueState()
        self.role_context = {}
    
    def get_conversation_summary(self) -> Dict:
        """获取对话摘要"""
        return {
            "turn_count": self.current_state.turn_count,
            "total_messages": len(self.current_state.messages),
            "current_context_length": self.get_context_length(),
            "active_roles": list(self.role_context.keys()),
            "last_user_message": (
                self.current_state.messages[-1].content 
                if self.current_state.messages and 
                   self.current_state.messages[-1].role == MessageRole.USER
                else None
            ),
        }


class MultiAgentDialogueManager(DialogueManager):
    """
    多代理对话管理器
    
    支持:
    - 多个角色同时参与对话
    - 观察者模式
    - 群聊
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.agents: Dict[str, DialogueState] = {}
        self.active_agents: List[str] = []
    
    def register_agent(self, agent_id: str):
        """注册代理"""
        if agent_id not in self.agents:
            self.agents[agent_id] = DialogueState()
            self.active_agents.append(agent_id)
    
    def switch_active_agent(self, agent_id: str):
        """切换当前活跃代理"""
        if agent_id in self.agents:
            self.current_state = self.agents[agent_id]
    
    def broadcast_message(self, from_agent: str, content: str):
        """广播消息给所有代理"""
        for agent_id, state in self.agents.items():
            if agent_id != from_agent:
                state.add_message(
                    MessageRole.OBSERVER,
                    f"[{from_agent}]: {content}"
                )


# 工厂函数
def create_dialogue_manager(config: Dict) -> DialogueManager:
    """创建对话管理器"""
    return DialogueManager(config)
