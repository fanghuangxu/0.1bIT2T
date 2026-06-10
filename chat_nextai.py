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

def chat():
    print("Loading NextAI model...")
    model, tokenizer = load_model()
    print("NextAI loaded successfully!\n")
    
    print("Welcome to NextAI Chat!")
    print("Type 'exit' to quit.\n")
    
    while True:
        prompt = input("You: ")
        
        if prompt.lower() == 'exit':
            print("NextAI: Goodbye!")
            break
        
        input_text = f"<user> {prompt}"
        input_ids = tokenizer.encode(input_text, max_length=64).unsqueeze(0)
        
        generated_tokens = model.generate(input_ids, max_length=16)
        response = tokenizer.decode(generated_tokens)
        
        print(f"NextAI: {response}\n")

if __name__ == "__main__":
    chat()