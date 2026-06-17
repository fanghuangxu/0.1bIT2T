"""
多语言对话AI系统
支持Markdown输出、EOS终止符、RL奖励、多轮对话、角色扮演

基于 Qwen2.5-1.5B 模型，支持中文、英文、德文
可在2GB显存或CPU上运行
"""

import os
import sys

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .configs.config import (
    MODEL_CONFIG,
    REWARD_CONFIG,
    DIALOGUE_CONFIG,
    ROLE_CONFIG,
    HARDWARE_CONFIG,
)
from .src.model_interface import (
    ModelInterface,
    CPUModelInterface,
    create_model_interface,
)
from .src.rewards.reward import RewardCalculator, create_reward_calculator
from .src.dialogue.dialogue_manager import (
    DialogueManager,
    MessageRole,
    create_dialogue_manager,
)
from .src.roles.role_system import (
    RoleManager,
    RolePlayEngine,
    create_role_manager,
    create_role_play_engine,
)

__version__ = "1.0.0"
__author__ = "Multimodal AI Team"


class MultimodalChatAI:
    """
    多语言对话AI系统主类
    
    功能:
    - 多语言对话 (中文、英文、德文)
    - Markdown格式输出
    - 多轮对话记忆
    - 角色扮演
    - RL奖励机制
    - 支持2GB显存或CPU运行
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = "auto",
        use_quantization: bool = True,
    ):
        """
        初始化多语言对话AI
        
        Args:
            model_path: 模型路径或HuggingFace模型ID
            device: 设备类型 ("auto", "cuda", "cpu")
            use_quantization: 是否使用量化
        """
        self.config = {
            "MODEL_CONFIG": MODEL_CONFIG,
            "REWARD_CONFIG": REWARD_CONFIG,
            "DIALOGUE_CONFIG": DIALOGUE_CONFIG,
            "ROLE_CONFIG": ROLE_CONFIG,
        }
        
        # 选择接口类型
        if device == "cpu":
            self.interface = CPUModelInterface(self.config)
        else:
            self.interface = create_model_interface(self.config)
        
        self.device = device
        self.model_loaded = False
        
        # 如果提供了模型路径，加载模型
        if model_path:
            self.load_model(model_path, use_quantization)
    
    def load_model(
        self,
        model_path: str = None,
        use_quantization: bool = True,
    ):
        """加载模型"""
        print(f"正在加载模型 (设备: {self.device}, 量化: {use_quantization})...")
        self.interface.load_model(model_path, use_quantization)
        self.model_loaded = True
        print("模型加载完成!")
    
    def chat(
        self,
        message: str,
        system_prompt: str = None,
        role_id: str = None,
        return_rewards: bool = False,
        **generation_kwargs
    ) -> dict:
        """
        对话接口
        
        Args:
            message: 用户消息
            system_prompt: 系统提示词
            role_id: 角色ID (可选)
            return_rewards: 是否返回奖励信息
            **generation_kwargs: 生成参数
            
        Returns:
            dict: {
                "text": 回复文本,
                "rewards": 奖励信息 (可选),
                "stopped_at_eos": 是否在EOS处停止,
                "metadata": 元数据
            }
        """
        if not self.model_loaded:
            return {"error": "模型未加载"}
        
        # 执行对话
        result = self.interface.chat(
            user_input=message,
            system_prompt=system_prompt,
            role_id=role_id,
            **generation_kwargs
        )
        
        response = {
            "text": result.text,
            "stopped_at_eos": result.stopped_at_eos,
            "stopped_naturally": result.stopped_naturally,
            "metadata": result.metadata,
        }
        
        if return_rewards:
            response["rewards"] = result.rewards
        
        return response
    
    def generate(
        self,
        prompt: str,
        return_rewards: bool = False,
        **generation_kwargs
    ) -> dict:
        """
        文本生成接口
        
        Args:
            prompt: 输入提示词
            return_rewards: 是否返回奖励
            **generation_kwargs: 生成参数
            
        Returns:
            dict: 生成结果
        """
        if not self.model_loaded:
            return {"error": "模型未加载"}
        
        result = self.interface.generate(
            prompt=prompt,
            returnRewards=return_rewards,
            **generation_kwargs
        )
        
        response = {
            "text": result.text,
            "stopped_at_eos": result.stopped_at_eos,
            "stopped_naturally": result.stopped_naturally,
        }
        
        if return_rewards:
            response["rewards"] = result.rewards
        
        return response
    
    def list_roles(self) -> list:
        """列出可用角色"""
        return self.interface.list_available_roles()
    
    def switch_role(self, role_id: str) -> bool:
        """切换角色"""
        return self.interface.switch_role(role_id)
    
    def end_role_play(self):
        """结束角色扮演"""
        self.interface.end_role_play()
    
    def reset(self):
        """重置对话"""
        self.interface.reset_conversation()
    
    def get_history(self) -> list:
        """获取对话历史"""
        return self.interface.get_conversation_history()
    
    def get_reward_breakdown(self, text: str, prompt: str = "") -> dict:
        """获取奖励分解"""
        return self.interface.get_reward_breakdown(text, prompt)
    
    def format_markdown(self, text: str) -> str:
        """格式化Markdown"""
        return self.interface.format_markdown(text)


# 便捷函数
def create_chat_ai(
    model_path: str = None,
    device: str = "auto",
    use_quantization: bool = True,
) -> MultimodalChatAI:
    """
    创建多语言对话AI实例
    
    Args:
        model_path: 模型路径
        device: 设备类型
        use_quantization: 是否量化
        
    Returns:
        MultimodalChatAI: AI实例
    """
    return MultimodalChatAI(
        model_path=model_path,
        device=device,
        use_quantization=use_quantization,
    )


# CLI界面
def run_cli():
    """运行命令行界面"""
    import readline  # 支持历史记录
    
    print("=" * 60)
    print("多语言对话AI系统 v1.0")
    print("支持: 中文、英文、德文")
    print("命令: /role <id> - 切换角色")
    print("      /reset - 重置对话")
    print("      /history - 查看历史")
    print("      /quit - 退出")
    print("=" * 60)
    
    # 加载模型
    model_path = input("\n请输入模型路径 (直接回车使用Qwen2.5-1.5B): ").strip()
    if not model_path:
        model_path = None
    
    device = input("使用设备 (auto/cuda/cpu, 直接回车使用auto): ").strip() or "auto"
    
    ai = create_chat_ai(model_path=model_path, device=device)
    
    print("\n开始对话吧!")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input.startswith("/"):
                cmd = user_input[1:].split()[0].lower()
                args = user_input[1:].split()[1:]
                
                if cmd == "quit":
                    print("再见!")
                    break
                elif cmd == "reset":
                    ai.reset()
                    print("对话已重置")
                    continue
                elif cmd == "history":
                    history = ai.get_history()
                    for msg in history[-10:]:
                        print(f"[{msg['role']}]: {msg['content'][:50]}...")
                    continue
                elif cmd == "role":
                    if args:
                        roles = ai.list_roles()
                        role_ids = [r["id"] for r in roles]
                        if args[0] in role_ids:
                            ai.switch_role(args[0])
                            print(f"已切换到角色: {args[0]}")
                        else:
                            print(f"可用角色: {', '.join(role_ids)}")
                    else:
                        roles = ai.list_roles()
                        for r in roles:
                            print(f"  {r['id']}: {r['name']} - {r['description']}")
                    continue
                elif cmd == "help":
                    print("/role <id> - 切换角色")
                    print("/reset - 重置对话")
                    print("/history - 查看历史")
                    print("/quit - 退出")
                    continue
            
            # 对话
            result = ai.chat(user_input, return_rewards=True)
            
            if "error" in result:
                print(f"错误: {result['error']}")
                continue
            
            # 输出
            print(f"\nAI: {result['text']}")
            
            # 显示奖励信息 (可选)
            if result.get("rewards"):
                rewards = result["rewards"]
                print(f"\n[奖励: 总={rewards['total']:.3f}, "
                      f"长度={rewards['length_reward']:.3f}, "
                      f"正确性={rewards['correctness_reward']:.3f}, "
                      f"EOS={rewards['eos_reward']:.3f}]")
        
        except KeyboardInterrupt:
            print("\n\n使用 /quit 退出")
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    run_cli()
