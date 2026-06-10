from .image_model import ImageModel
from .text_model import TextModel
from .config import ModelConfig

class DialogAI:
    def __init__(self, device=None):
        self.device = device or ModelConfig.DEVICE
        self.image_model = ImageModel(device=self.device)
        self.text_model = TextModel(device=self.device)
        self.conversation_history = []

    def chat(self, message, image_path=None):
        response = ""
        
        if image_path:
            if message.strip():
                response = self.image_model.answer_about_image(image_path, message)
            else:
                response = self.image_model.describe_image(image_path)
        else:
            response = self.text_model.generate_response(message)
        
        self.conversation_history.append({
            "user": message,
            "image": image_path,
            "assistant": response
        })
        
        return response

    def summarize_conversation(self):
        full_text = "\n".join([f"User: {turn['user']}" for turn in self.conversation_history])
        return self.text_model.summarize(full_text)

    def clear_history(self):
        self.conversation_history = []

    def get_history(self):
        return self.conversation_history