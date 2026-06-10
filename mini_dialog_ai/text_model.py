from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .config import ModelConfig

class TextModel:
    def __init__(self, device=None):
        self.device = device or ModelConfig.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(ModelConfig.TEXT_MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(ModelConfig.TEXT_MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def generate_response(self, input_text):
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=ModelConfig.MAX_LENGTH,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=ModelConfig.MAX_LENGTH,
                temperature=ModelConfig.TEMPERATURE,
                top_p=ModelConfig.TOP_P,
                do_sample=True,
                num_beams=4,
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=ModelConfig.MAX_LENGTH,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=int(ModelConfig.MAX_LENGTH * 0.3),
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                num_beams=2,
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)