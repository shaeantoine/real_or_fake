import torch
import pandas as pd
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Internal classes
from siamesenetwork import SiameseNetwork
from textpairdataset import TextPairDataset

# Load in training data
train_data = "data/train_df.csv"
train_df = pd.read_csv(train_data)

# Initialize Model/ Architecture
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = TextPairDataset(train_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = SiameseNetwork()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.BCEWithLogitsLoss()

# Training Loop 
for epoch in range(3):
    model.train()
    total_loss = 0
    preds, labels = [], []

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids_1 = batch["input_ids_1"].to(device)
        attention_mask_1 = batch["attention_mask_1"].to(device)
        input_ids_2 = batch["input_ids_2"].to(device)
        attention_mask_2 = batch["attention_mask_2"].to(device)
        targets = batch["label"].float().to(device)

        logits = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds += (torch.sigmoid(logits) > 0.5).cpu().numpy().tolist()
        labels += targets.cpu().numpy().tolist()

    acc = accuracy_score(labels, preds)
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

# Save Model
model_path = "models/trained_model.zip"
torch.save(model.state_dict(), model_path)