import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score
import logging
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO, datefmt='%I:%M:%S')

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


# ---------------------------
# Feed Forward
# ---------------------------
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ---------------------------
# Transformer Block
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------
# BERT Transformer
# ---------------------------
class BERTTransformer(nn.Module):
    def __init__(self, vocab_size=30522, max_seq_len=128, embed_dim=512, 
                 num_heads=8, num_layers=8, num_classes=1, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Regression head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        
    def forward(self, input_ids, mask=None):
        B, N = input_ids.shape
        
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :N, :]
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


def create_bert_50m(num_classes=1):
    return BERTTransformer(
        vocab_size=30522,
        max_seq_len=128,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        num_classes=num_classes,
        dropout=0.05
    )


# ---------------------------
class YelpReviewDataset(Dataset):
    def __init__(self, split='train', max_length=128, subset_size=100000):
        raw_data = load_dataset("yelp_review_full", split=split)
        self.data = raw_data.select(range(min(len(raw_data), subset_size)))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        label = torch.tensor(item['label'] / 4.0, dtype=torch.float32)
        return input_ids, label


# ---------------------------
# Training and Evaluation
# ---------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for input_ids, y in loader:
            input_ids, y = input_ids.to(device), y.to(device)
            out = model(input_ids).squeeze(-1)
            loss = criterion(out, y)
            total_loss += loss.item()
            preds.extend(out.cpu().numpy())
            labels.extend(y.cpu().numpy())
    r2 = r2_score(labels, preds)
    return total_loss / len(loader), r2


def main():
    save_dir = 'yelp_bert_100M_regression_models'
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = 32
    num_epochs = 3
    learning_rate = 1e-4
    patience = 2
    num_models = 100
    
    logging.info(f"Device: {device}")
    logging.info("Training BERT regression on Yelp Review 100M\n")
    
    # Load datasets
    train_dataset = YelpReviewDataset('train')
    val_dataset = YelpReviewDataset('test', subset_size=8000)
    test_dataset = YelpReviewDataset('test', subset_size=20000)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logging.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}\n")
    
    for model_idx in tqdm(range(num_models), desc="Training models"):
        torch.manual_seed(model_idx)
        np.random.seed(model_idx)
        
        model = create_bert_50m(num_classes=1).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        if model_idx == 0:
            total_params = sum(p.numel() for p in model.parameters())
            logging.info(f"Total parameters:{total_params}")
        
        best_r2 = -float("inf")
        epochs_no_improve = 0
        best_model_path = os.path.join(save_dir, f"bert_{model_idx}_best.pt")
        
        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_r2 = evaluate(model, val_loader)
            logging.info(f"Epoch {epoch} | train_loss:{train_loss:.4f} | val_loss:{val_loss:.4f} | val_r2:{val_r2:.4f}")
            
            if val_r2 > best_r2:
                best_r2 = val_r2
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break
        
        model.load_state_dict(torch.load(best_model_path))
        test_loss, test_r2 = evaluate(model, test_loader)
        logging.info(f"Final Test | loss:{test_loss:.4f} | R2:{test_r2:.4f}\n")


if __name__ == "__main__":
    main()
