import pytest
from mini_dialog_ai import DialogAI, ImageModel, TextModel

class TestImageModel:
    def test_describe_image(self):
        model = ImageModel(device="cpu")
        assert model is not None
    
    def test_answer_about_image(self):
        model = ImageModel(device="cpu")
        assert model is not None

class TestTextModel:
    def test_generate_response(self):
        model = TextModel(device="cpu")
        response = model.generate_response("Hello")
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_summarize(self):
        model = TextModel(device="cpu")
        text = "This is a long text that needs to be summarized. It contains multiple sentences and should be shortened."
        summary = model.summarize(text)
        assert isinstance(summary, str)
        assert len(summary) > 0

class TestDialogAI:
    def test_chat_text(self):
        ai = DialogAI(device="cpu")
        response = ai.chat("Hello")
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_conversation_history(self):
        ai = DialogAI(device="cpu")
        ai.chat("Hello")
        history = ai.get_history()
        assert len(history) == 1
    
    def test_clear_history(self):
        ai = DialogAI(device="cpu")
        ai.chat("Hello")
        ai.clear_history()
        history = ai.get_history()
        assert len(history) == 0
    
    def test_summarize_conversation(self):
        ai = DialogAI(device="cpu")
        ai.chat("Hello")
        ai.chat("How are you?")
        summary = ai.summarize_conversation()
        assert isinstance(summary, str)
        assert len(summary) > 0