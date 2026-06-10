import os
import sys
sys.path.insert(0, '/workspace')

from mini_dialog_ai.config import ModelConfig
from mini_dialog_ai.dataset import DataProcessor
from mini_dialog_ai.trainer import QwenMoEModel, Trainer

def main():
    print("=== 初始化训练环境 ===")
    
    print("\n1. 生成示例数据集...")
    data_path = "data/train_data.json"
    DataProcessor.generate_sample_data(data_path, num_samples=50)
    
    print("\n2. 加载模型和tokenizer...")
    qwen_model = QwenMoEModel(device=ModelConfig.DEVICE)
    
    print("\n3. 创建数据处理器...")
    processor = DataProcessor(qwen_model.tokenizer)
    train_dataloader = processor.create_dataloader(
        data_path,
        batch_size=ModelConfig.BATCH_SIZE,
        max_length=ModelConfig.MAX_LENGTH
    )
    
    print("\n4. 开始训练...")
    trainer = Trainer(qwen_model.model, qwen_model.tokenizer, device=ModelConfig.DEVICE)
    trainer.train(train_dataloader, num_epochs=ModelConfig.NUM_EPOCHS)
    
    print("\n5. 保存模型...")
    output_dir = "models/trained_model"
    trainer.save_model(output_dir)
    
    print("\n=== 训练完成 ===")
    
    print("\n=== 测试对话 ===")
    test_prompt = "你好！"
    response = qwen_model.generate_response(test_prompt)
    print(f"用户: {test_prompt}")
    print(f"AI: {response}")

if __name__ == "__main__":
    main()