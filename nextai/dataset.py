import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<user>': 3, '<assistant>': 4}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.next_id = 5
    
    def fit_on_texts(self, texts):
        for text in texts:
            for word in text.lower().split():
                if word not in self.vocab:
                    self.vocab[word] = self.next_id
                    self.reverse_vocab[self.next_id] = word
                    self.next_id += 1
    
    def encode(self, text, max_length=64):
        tokens = []
        for word in text.lower().split():
            tokens.append(self.vocab.get(word, 0))
            if len(tokens) >= max_length:
                break
        while len(tokens) < max_length:
            tokens.append(0)
        return torch.tensor(tokens)
    
    def decode(self, tokens):
        words = []
        for token in tokens:
            if token == 0:
                continue
            if token == 2:
                break
            words.append(self.reverse_vocab.get(token, '<unk>'))
        return ' '.join(words)

class NextAIDataset(Dataset):
    def __init__(self, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.generate_sample_data()
    
    def generate_sample_data(self):
        samples = []
        
        conversations = [
            ("hello", "hi there!"),
            ("how are you", "i'm fine, thank you!"),
            ("what is your name", "my name is nextai"),
            ("tell me a joke", "why did the computer go to the doctor? it had a virus!"),
            ("goodbye", "bye! have a nice day"),
            ("what is ai", "ai stands for artificial intelligence"),
            ("how old are you", "i am just a computer program"),
            ("what can you do", "i can have conversations and answer questions"),
            ("thank you", "you're welcome!"),
            ("tell me about yourself", "i am nextai, an ai assistant"),
            ("who created you", "i was created by a team of developers"),
            ("what is the weather today", "i don't have access to real-time weather"),
            ("explain machine learning", "machine learning is a type of ai that learns from data"),
            ("what is deep learning", "deep learning uses neural networks with many layers"),
            ("how does ai work", "ai uses algorithms to learn patterns from data"),
        ]
        
        for user_msg, assistant_msg in conversations:
            samples.append({
                'input': f"<user> {user_msg}",
                'target': f"<assistant> {assistant_msg}"
            })
        
        return samples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        input_ids = self.tokenizer.encode(sample['input'], max_length=self.max_length)
        target_ids = self.tokenizer.encode(sample['target'], max_length=self.max_length)
        
        return {
            'input_ids': input_ids,
            'labels': target_ids
        }

class DataProcessor:
    @staticmethod
    def create_dataloader(tokenizer, batch_size=2, max_length=64):
        dataset = NextAIDataset(tokenizer, max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }