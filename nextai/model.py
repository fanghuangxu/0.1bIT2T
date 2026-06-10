import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        gates = self.softmax(self.fc(x))
        return gates

class MoEModel(nn.Module):
    def __init__(self, num_experts=4, input_dim=256, hidden_dim=256, output_dim=256):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])
        self.gating_network = GatingNetwork(input_dim, num_experts)
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1) if x.dim() > 1 else 1
        
        if x.dim() == 3:
            x_flat = x.reshape(-1, x.size(-1))
            gates = self.gating_network(x_flat)
            expert_outputs = []
            for expert in self.experts:
                expert_outputs.append(expert(x_flat))
            expert_outputs = torch.stack(expert_outputs, dim=-1)
            gates = gates.unsqueeze(-2)
            output = torch.matmul(expert_outputs, gates.transpose(-1, -2)).squeeze(-1)
            return output.reshape(batch_size, seq_len, -1), gates.reshape(batch_size, seq_len, -1)
        else:
            gates = self.gating_network(x)
            expert_outputs = []
            for expert in self.experts:
                expert_outputs.append(expert(x))
            expert_outputs = torch.stack(expert_outputs, dim=-1)
            gates = gates.unsqueeze(-2)
            output = torch.matmul(expert_outputs, gates.transpose(-1, -2)).squeeze(-1)
            return output, gates

class NextAI(nn.Module):
    def __init__(self, vocab_size=10000, num_experts=4, embed_dim=128, max_seq_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        
        self.encoder_moe = MoEModel(
            num_experts=num_experts,
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim
        )
        
        self.decoder_moe = MoEModel(
            num_experts=num_experts,
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim
        )
        
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
        self.image_projection = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.special_tokens = {
            'pad': 0,
            'bos': 1,
            'eos': 2,
            'user': 3,
            'assistant': 4
        }
    
    def forward(self, input_ids, image_features=None):
        batch_size, seq_len = input_ids.size()
        
        if seq_len > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len:]
            seq_len = self.max_seq_len
        
        x = self.embedding(input_ids) + self.pos_encoding[:, :seq_len, :]
        
        encoder_output, _ = self.encoder_moe(x)
        
        if image_features is not None:
            image_embedding = self.image_projection(image_features).unsqueeze(1)
            encoder_output = encoder_output + image_embedding
        
        decoder_output, _ = self.decoder_moe(encoder_output)
        
        logits = self.output_layer(decoder_output)
        
        return logits
    
    def generate(self, input_ids, image_features=None, max_length=16):
        self.eval()
        with torch.no_grad():
            generated = []
            
            for _ in range(max_length):
                logits = self.forward(input_ids, image_features)
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                
                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == self.special_tokens['eos']:
                    break
            
            return generated