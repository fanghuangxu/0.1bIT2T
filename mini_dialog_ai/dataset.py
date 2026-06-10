import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class DialogDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, image_processor=None):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_processor = image_processor

    def load_data(self, data_path):
        data = []
        if os.path.isdir(data_path):
            for file in os.listdir(data_path):
                if file.endswith('.json'):
                    with open(os.path.join(data_path, file), 'r', encoding='utf-8') as f:
                        data.extend(json.load(f))
        elif data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        messages = sample.get('messages', [])
        
        if not messages:
            return None
        
        text = ''
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role:
                text += f"{role}: {content}\n"
            else:
                text += f"{content}\n"
        
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
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        if 'image' in sample and self.image_processor:
            image_path = sample['image']
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                image_features = self.image_processor(image, return_tensors='pt')
                result['pixel_values'] = image_features['pixel_values'].flatten()
        
        return result

class DataProcessor:
    def __init__(self, tokenizer, image_processor=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def create_dataloader(self, data_path, batch_size=4, max_length=512, shuffle=True):
        dataset = DialogDataset(data_path, self.tokenizer, max_length, self.image_processor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.stack(labels)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        if 'pixel_values' in batch[0]:
            pixel_values = [item['pixel_values'] for item in batch]
            result['pixel_values'] = torch.stack(pixel_values)
        
        return result

    @staticmethod
    def generate_sample_data(output_path, num_samples=100):
        samples = []
        for i in range(num_samples):
            sample_type = random.choice(['text', 'text', 'image'])
            
            if sample_type == 'image':
                messages = [
                    {'role': 'user', 'content': '描述这张图片'},
                    {'role': 'assistant', 'content': '这是一张展示城市风景的图片，画面中有高楼大厦和繁忙的街道。'}
                ]
                sample = {
                    'id': f'img_{i}',
                    'image': f'images/sample_{i}.jpg',
                    'messages': messages
                }
            else:
                messages = []
                num_turns = random.randint(2, 5)
                for j in range(num_turns):
                    if j % 2 == 0:
                        content = random.choice([
                            '你好！',
                            '今天天气怎么样？',
                            '什么是人工智能？',
                            '讲个笑话吧',
                            '推荐一本好书',
                            '解释一下量子计算',
                            '明天会下雨吗？'
                        ])
                        messages.append({'role': 'user', 'content': content})
                    else:
                        content = random.choice([
                            '你好！很高兴为你服务。',
                            '今天天气晴朗，温度适宜。',
                            '人工智能是计算机科学的一个分支...',
                            '为什么程序员总是分不清万圣节和圣诞节？因为Oct 31等于Dec 25！',
                            '我推荐《人类简史》这本书。',
                            '量子计算利用量子力学原理进行计算...',
                            '根据天气预报，明天可能会下雨。'
                        ])
                        messages.append({'role': 'assistant', 'content': content})
                
                sample = {
                    'id': f'text_{i}',
                    'messages': messages
                }
            
            samples.append(sample)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"Sample data generated at: {output_path}")

import torch