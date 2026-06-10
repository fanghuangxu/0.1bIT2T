import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class HFDialogDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, max_length=512, split='train'):
        self.dataset = load_dataset(dataset_name, split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        if 'messages' in sample:
            messages = sample['messages']
        elif 'conversations' in sample:
            messages = sample['conversations']
        elif 'text' in sample:
            messages = [{'role': 'user', 'content': sample['text']}]
        else:
            return None
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class HFDataProcessor:
    @staticmethod
    def get_available_datasets():
        return {
            'text': [
                'lmsys/vicuna-chat-v1.5',
                'tatsu-lab/alpaca',
                'HuggingFaceH4/ultrachat_200k',
                'MAGAer13/ShareGPT_Vicuna_unfiltered',
                'timdettmers/openassistantistant'
            ],
            'image_text': [
                'liuhaotian/LLaVA-Instruct-150K',
                'microsoft/COCO-Captions',
                'linqingyang/cc_sbu_align',
                'HuggingFaceM4/COCO',
            ]
        }
    
    @staticmethod
    def load_text_dataset(dataset_name, tokenizer, max_length=512, split='train[:10%]'):
        dataset = HFDialogDataset(dataset_name, tokenizer, max_length, split)
        return dataset
    
    @staticmethod
    def load_image_text_dataset(dataset_name, tokenizer, image_processor, max_length=512, split='train[:10%]'):
        dataset = load_dataset(dataset_name, split=split)
        return dataset
    
    @staticmethod
    def create_dataloader(dataset, batch_size=4, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=HFDataProcessor.collate_fn)
    
    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }