import os
import torch
import torch.nn as nn
from nextai.model import NextAI
from nextai.dataset import SimpleTokenizer, NextAIDataset, DataProcessor

def train():
    print("Initializing NextAI MoE model...")
    
    tokenizer = SimpleTokenizer()
    
    dataset = NextAIDataset(tokenizer, max_length=64)
    all_texts = [sample['input'] + ' ' + sample['target'] for sample in dataset.data]
    tokenizer.fit_on_texts(all_texts)
    
    vocab_size = len(tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    model = NextAI(vocab_size=vocab_size, num_experts=4, embed_dim=128, max_seq_len=64)
    device = torch.device("cpu")
    model.to(device)
    
    print(f"Model loaded on {device}")
    print(f"Number of experts: 4")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    dataloader = DataProcessor.create_dataloader(tokenizer, batch_size=4, max_length=64)
    print(f"Dataset size: {len(dataset)}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    num_epochs = 500
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids)
            
            loss = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        
        if avg_loss < 0.5:
            print(f"Early stopping at epoch {epoch+1} with loss {avg_loss:.4f}")
            break
    
    os.makedirs("./models/nextai", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': tokenizer.vocab,
        'reverse_vocab': tokenizer.reverse_vocab
    }, "./models/nextai/model.pt")
    print("\nModel saved to ./models/nextai/")
    
    test_inference(model, tokenizer, device)

def test_inference(model, tokenizer, device):
    print("\nTesting inference...")
    model.eval()
    
    test_prompts = [
        "hello",
        "how are you",
        "what is your name",
        "tell me a joke",
        "thank you",
        "goodbye"
    ]
    
    for prompt in test_prompts:
        print(f"\nUser: {prompt}")
        
        input_text = f"<user> {prompt}"
        input_ids = tokenizer.encode(input_text, max_length=64).unsqueeze(0).to(device)
        
        generated_tokens = model.generate(input_ids, max_length=16)
        response = tokenizer.decode(generated_tokens)
        
        print(f"NextAI: {response}")

if __name__ == "__main__":
    train()