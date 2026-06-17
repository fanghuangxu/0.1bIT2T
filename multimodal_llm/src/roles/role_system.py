"""
角色扮演系统
支持预设角色、自定义角色、角色切换
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json


@dataclass
class Role:
    """角色定义"""
    id: str
    name: str
    description: str
    prompt: str
    avatar: Optional[str] = None
    personality: List[str] = field(default_factory=list)
    speaking_style: str = "balanced"
    languages: List[str] = field(default_factory=lambda: ["zh", "en", "de"])
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "avatar": self.avatar,
            "personality": self.personality,
            "speaking_style": self.speaking_style,
            "languages": self.languages,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Role":
        return cls(**data)
    
    def apply_to_prompt(self, base_prompt: str) -> str:
        """将角色应用到基础提示词"""
        role_intro = f"""角色设定: {self.name}
角色描述: {self.description}
"""
        
        if self.personality:
            role_intro += f"性格特点: {', '.join(self.personality)}\n"
        
        role_intro += f"说话风格: {self.speaking_style}\n"
        role_intro += f"语言能力: {', '.join(self.languages)}\n"
        
        return f"{role_intro}\n{self.prompt}\n\n{base_prompt}"


class RoleManager:
    """
    角色管理器
    
    功能:
    - 预设角色管理
    - 自定义角色创建
    - 角色切换
    - 角色组合
    """
    
    # 预设角色库
    PRESET_ROLES = {
        "assistant": Role(
            id="assistant",
            name="NextAI助手",
            description="由Next Studio开发的通用AI助手",
            prompt="你是 NextAI，一个由 Next Studio 开发的先进多语言AI助手。你精通中文、英文和德文，能够进行流畅的多语言对话。",
            speaking_style="professional",
            languages=["zh", "en", "de"],
        ),
        "teacher": Role(
            id="teacher",
            name="老师",
            description="经验丰富的教育者",
            prompt="你是一位有着20年教学经验的老师。你擅长用浅显易懂的方式讲解知识，注重启发学生思考。你的讲解总是条理清晰、循序渐进。",
            personality=["耐心", "细心", "启发性", "严谨"],
            speaking_style="educational",
            languages=["zh", "en", "de"],
        ),
        "translator": Role(
            id="translator",
            name="翻译官",
            description="专业翻译，精通中英德互译",
            prompt="你是一位专业的语言翻译，精通中文、英文和德文。你翻译准确流畅，既保留原意又符合目标语言的表达习惯。",
            personality=["精准", "严谨", "博学"],
            speaking_style="precise",
            languages=["zh", "en", "de"],
        ),
        "friend": Role(
            id="friend",
            name="朋友",
            description="友善的朋友",
            prompt="你是用户的好朋友，你们可以随意聊天、分享心事。你真诚友善，善于倾听，也会在适当时候给出建议。",
            personality=["友善", "真诚", "幽默", "善解人意"],
            speaking_style="casual",
            languages=["zh", "en", "de"],
        ),
        "expert": Role(
            id="expert",
            name="专家",
            description="各领域专家",
            prompt="你在特定领域有深厚的专业知识，可以提供权威的建议和分析。回答问题时注重专业性和实用性。",
            personality=["专业", "权威", "严谨", "务实"],
            speaking_style="professional",
            languages=["zh", "en", "de"],
        ),
        "creative_writer": Role(
            id="creative_writer",
            name="创意作家",
            description="富有想象力的作家",
            prompt="你是一位富有想象力的作家，擅长创作故事、诗歌和各种创意内容。你的文字生动有趣，充满感染力。",
            personality=["创意", "想象力", "表达力", "感性"],
            speaking_style="creative",
            languages=["zh", "en", "de"],
        ),
    }
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.preset_roles = self.PRESET_ROLES.copy()
        
        # 从配置加载额外角色
        if config and "preset_roles" in config:
            for role_data in config["preset_roles"]:
                role = Role.from_dict(role_data)
                self.preset_roles[role.id] = role
        
        # 当前激活的角色
        self.active_roles: List[Role] = []
        
        # 角色组合
        self.role_combinations: Dict[str, List[str]] = {}
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """获取角色"""
        return self.preset_roles.get(role_id)
    
    def list_roles(self) -> List[Dict]:
        """列出所有可用角色"""
        return [
            {
                "id": role.id,
                "name": role.name,
                "description": role.description,
                "languages": role.languages,
            }
            for role in self.preset_roles.values()
        ]
    
    def activate_role(self, role_id: str) -> bool:
        """激活角色"""
        role = self.get_role(role_id)
        if role and role not in self.active_roles:
            self.active_roles.append(role)
            return True
        return False
    
    def deactivate_role(self, role_id: str) -> bool:
        """停用角色"""
        role = self.get_role(role_id)
        if role and role in self.active_roles:
            self.active_roles.remove(role)
            return True
        return False
    
    def get_active_roles(self) -> List[Role]:
        """获取当前激活的角色"""
        return self.active_roles.copy()
    
    def create_custom_role(
        self,
        role_id: str,
        name: str,
        description: str,
        prompt: str,
        **kwargs
    ) -> Role:
        """创建自定义角色"""
        role = Role(
            id=role_id,
            name=name,
            description=description,
            prompt=prompt,
            **kwargs
        )
        self.preset_roles[role_id] = role
        return role
    
    def delete_custom_role(self, role_id: str) -> bool:
        """删除自定义角色"""
        if role_id in self.PRESET_ROLES:
            return False  # 不能删除预设角色
        if role_id in self.preset_roles:
            del self.preset_roles[role_id]
            if role_id in [r.id for r in self.active_roles]:
                self.active_roles = [r for r in self.active_roles if r.id != role_id]
            return True
        return False
    
    def build_role_prompt(self, base_prompt: str = None) -> str:
        """构建包含角色的完整提示词"""
        if not self.active_roles:
            return base_prompt or ""
        
        if len(self.active_roles) == 1:
            return self.active_roles[0].apply_to_prompt(base_prompt or "")
        
        # 多角色组合
        role_context = "【当前角色组合】\n"
        for role in self.active_roles:
            role_context += f"- {role.name}: {role.description}\n"
        
        role_context += "\n请根据以上角色设定进行对话。如需切换角色，可明确说明。\n"
        
        if base_prompt:
            return f"{role_context}\n{base_prompt}"
        return role_context
    
    def save_role_combination(self, name: str, role_ids: List[str]):
        """保存角色组合"""
        self.role_combinations[name] = role_ids
    
    def load_role_combination(self, name: str) -> bool:
        """加载角色组合"""
        if name not in self.role_combinations:
            return False
        
        self.active_roles = []
        for role_id in self.role_combinations[name]:
            self.activate_role(role_id)
        return True
    
    def export_roles(self) -> str:
        """导出角色配置为JSON"""
        custom_roles = [
            role.to_dict() 
            for role_id, role in self.preset_roles.items()
            if role_id not in self.PRESET_ROLES
        ]
        return json.dumps(custom_roles, ensure_ascii=False, indent=2)
    
    def import_roles(self, json_str: str) -> int:
        """导入角色配置"""
        try:
            roles_data = json.loads(json_str)
            count = 0
            for role_data in roles_data:
                role = Role.from_dict(role_data)
                self.preset_roles[role.id] = role
                count += 1
            return count
        except Exception:
            return 0


class RolePlayEngine:
    """
    角色扮演引擎
    
    整合对话管理和角色系统
    """
    
    def __init__(
        self, 
        dialogue_manager,  # DialogueManager实例
        role_manager: RoleManager
    ):
        self.dialogue_manager = dialogue_manager
        self.role_manager = role_manager
        
        # 角色扮演状态
        self.is_role_playing = False
        self.current_scene: Optional[str] = None
    
    def start_role_play(
        self, 
        role_id: str,
        user_name: str = "用户",
        scene: Optional[str] = None
    ):
        """开始角色扮演"""
        role = self.role_manager.get_role(role_id)
        if not role:
            return False
        
        self.role_manager.activate_role(role_id)
        self.is_role_playing = True
        self.current_scene = scene
        
        # 设置对话上下文
        scene_intro = f"【场景设定】\n{scene}\n\n" if scene else ""
        role_intro = f"【角色扮演开始】\n你正在扮演: {role.name}\n{role.description}"
        
        context = f"{scene_intro}{role_intro}"
        
        # 更新对话管理器
        self.dialogue_manager.set_role_context(role_id, context)
        
        return True
    
    def end_role_play(self):
        """结束角色扮演"""
        self.is_role_playing = False
        self.current_scene = None
        self.role_manager.active_roles = []
        self.dialogue_manager.clear_role_context()
    
    def switch_role(self, new_role_id: str):
        """切换角色"""
        if not self.is_role_playing:
            return False
        
        old_role_id = self.role_manager.active_roles[0].id if self.role_manager.active_roles else None
        
        if old_role_id:
            self.role_manager.deactivate_role(old_role_id)
        
        self.role_manager.activate_role(new_role_id)
        return True
    
    def get_current_role(self) -> Optional[Role]:
        """获取当前角色"""
        if self.role_manager.active_roles:
            return self.role_manager.active_roles[0]
        return None


# 工厂函数
def create_role_manager(config: Dict = None) -> RoleManager:
    """创建角色管理器"""
    return RoleManager(config)


def create_role_play_engine(
    dialogue_manager,
    role_manager: RoleManager
) -> RolePlayEngine:
    """创建角色扮演引擎"""
    return RolePlayEngine(dialogue_manager, role_manager)
