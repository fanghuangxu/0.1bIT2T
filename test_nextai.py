import torch
from nextai.model import NextAI
from nextai.dataset import SimpleTokenizer

def load_model():
    checkpoint = torch.load("./models/nextai/model.pt", map_location='cpu')
    
    tokenizer = SimpleTokenizer()
    tokenizer.vocab = checkpoint['vocab']
    tokenizer.reverse_vocab = checkpoint['reverse_vocab']
    
    vocab_size = len(tokenizer.vocab)
    model = NextAI(vocab_size=vocab_size, num_experts=4, embed_dim=128, max_seq_len=64)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer

def test_chat():
    print("Loading NextAI model...")
    model, tokenizer = load_model()
    print("NextAI loaded successfully!\n")
    
    test_prompts = [
        "hello",
        "how are you",
        "what is your name",
        "tell me a joke",
        "thank you",
        "goodbye",
        "what is ai",
        "who created you"
    ]
    
    print("=== NextAI Chat Test ===\n")
    
    for prompt in test_prompts:
        print(f"You: {prompt}")
        
        input_text = f"<user> {prompt}"
        input_ids = tokenizer.encode(input_text, max_length=64).unsqueeze(0)
        
        generated_tokens = model.generate(input_ids, max_length=16)
        response = tokenizer.decode(generated_tokens)
        
        print(f"NextAI: {response}")
        print()

def test_image_input():
    print("\n=== Image Input Test ===\n")
    model, tokenizer = load_model()
    
    dummy_image_features = torch.randn(1, 2048)
    
    prompt = "describe this image"
    print(f"You (with image): {prompt}")
    
    input_text = f"<user> {prompt}"
    input_ids = tokenizer.encode(input_text, max_length=64).unsqueeze(0)
    
    generated_tokens = model.generate(input_ids, image_features=dummy_image_features, max_length=16)
    response = tokenizer.decode(generated_tokens)
    
    print(f"NextAI: {response}")

if __name__ == "__main__":
    test_chat()
    test_image_input()