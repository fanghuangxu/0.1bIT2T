from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from .config import ModelConfig

class ImageModel:
    def __init__(self, device=None):
        self.device = device or ModelConfig.DEVICE
        self.processor = BlipProcessor.from_pretrained(ModelConfig.IMAGE_MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(ModelConfig.IMAGE_MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def describe_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=ModelConfig.MAX_LENGTH,
                temperature=ModelConfig.TEMPERATURE,
                top_p=ModelConfig.TOP_P,
                do_sample=True
            )
        
        return self.processor.decode(out[0], skip_special_tokens=True)

    def answer_about_image(self, image_path, question):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=ModelConfig.MAX_LENGTH,
                temperature=ModelConfig.TEMPERATURE,
                top_p=ModelConfig.TOP_P,
                do_sample=True
            )
        
        return self.processor.decode(out[0], skip_special_tokens=True)