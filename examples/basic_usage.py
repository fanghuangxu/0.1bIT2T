from mini_dialog_ai import DialogAI

def text_to_text_example():
    ai = DialogAI(device="cpu")
    
    response = ai.chat("Hello! How are you?")
    print("Text Response:", response)
    
    response = ai.chat("What is artificial intelligence?")
    print("Text Response:", response)

def image_to_text_example():
    ai = DialogAI(device="cpu")
    
    response = ai.chat("", image_path="example_image.jpg")
    print("Image Description:", response)
    
    response = ai.chat("What is in this image?", image_path="example_image.jpg")
    print("Image Q&A Response:", response)

if __name__ == "__main__":
    print("=== Text to Text Example ===")
    text_to_text_example()
    
    print("\n=== Image to Text Example ===")
    image_to_text_example()
    
    print("\n=== Conversation Summary ===")
    summary = ai.summarize_conversation()
    print("Summary:", summary)