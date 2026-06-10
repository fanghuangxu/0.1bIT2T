import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BlipProcessor,
    BlipForConditionalGeneration,
    get_scheduler,
    BitsAndBytesConfig
)
from .config import ModelConfig
from .dataset import DataProcessor

class QwenMoEModel:
    def __init__(self, device=None):
        self.device = device or ModelConfig.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(
            ModelConfig.MODEL_NAME,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            ModelConfig.MODEL_NAME,
            quantization_config=self.quantization_config if ModelConfig.QUANTIZATION_BITS == 4 else None,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.image_processor = BlipProcessor.from_pretrained(ModelConfig.IMAGE_MODEL_NAME)
        self.image_model = BlipForConditionalGeneration.from_pretrained(
            ModelConfig.IMAGE_MODEL_NAME,
            quantization_config=self.quantization_config if ModelConfig.QUANTIZATION_BITS == 4 else None,
            device_map="auto"
        )

    def generate_response(self, prompt, image_path=None):
        if image_path:
            image = Image.open(image_path).convert("RGB")
            image_inputs = self.image_processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.image_model.get_image_features(**image_inputs)
            image_text = self.image_processor.decode(
                self.image_model.generate(**image_inputs)[0],
                skip_special_tokens=True
            )
            prompt = f"图片描述: {image_text}\n问题: {prompt}"
        
        messages = [
            {"role": "system", "content": "你是一个友好的助手。"},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=ModelConfig.MAX_LENGTH,
                temperature=ModelConfig.TEMPERATURE,
                top_p=ModelConfig.TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(text, "").strip()

class Trainer:
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ModelConfig.DEVICE
        self.optimizer = AdamW(model.parameters(), lr=ModelConfig.LEARNING_RATE)
    
    def train(self, train_dataloader, num_epochs=3):
        self.model.train()
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                if batch is None:
                    continue
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss / ModelConfig.GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                
                if (batch_idx + 1) % ModelConfig.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.optimizer.step()
                    lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item() * ModelConfig.GRADIENT_ACCUMULATION_STEPS
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
    
    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to: {output_dir}")

from PIL import Image