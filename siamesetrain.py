import os 
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

# ====================================
# Training Loop
# ====================================
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

train_dataset = TextPairDataset(train_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = SiameseNetwork()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.BCEWithLogitsLoss()

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



# ====================================
# Evaluation Script
# ====================================
import os
import pandas as pd
import torch
from transformers import AutoTokenizer

# === Configuration ===
test_dir = "data/test/"
output_csv = "predictions.csv"
model_name = "roberta-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Tokenizer and Model ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()
model.to(device)

# === Load Test File Pairs ===
test_data = []
for i, (dirpath, _, filenames) in enumerate(sorted(os.walk(test_dir))):
    if len(filenames) >= 2:
        try:
            f1_path = os.path.join(dirpath, filenames[0])
            f2_path = os.path.join(dirpath, filenames[1])

            with open(f1_path, 'r', encoding='utf-8') as f1:
                text1 = f1.read().strip()
            with open(f2_path, 'r', encoding='utf-8') as f2:
                text2 = f2.read().strip()

            test_data.append((i, text1, text2))
        except Exception as e:
            print(f"Error reading {dirpath}: {e}")

# === Predict Real Document ===
predictions = []

model.eval()
with torch.no_grad():
    for idx, text1, text2 in test_data:
        enc1 = tokenizer(text1, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        enc2 = tokenizer(text2, return_tensors="pt", truncation=True, padding='max_length', max_length=512)

        # Properly squeeze inputs for batch size 1
        input_ids_1 = enc1['input_ids'].squeeze(0).to(device)
        attention_mask_1 = enc1['attention_mask'].squeeze(0).to(device)
        input_ids_2 = enc2['input_ids'].squeeze(0).to(device)
        attention_mask_2 = enc2['attention_mask'].squeeze(0).to(device)

        logits = model(
            input_ids_1.unsqueeze(0),  # add batch dim
            attention_mask_1.unsqueeze(0),
            input_ids_2.unsqueeze(0),
            attention_mask_2.unsqueeze(0)
        )

        prob = torch.sigmoid(logits).item()
        prediction = "2" if prob > 0.5 else "1"
        predictions.append({"id": idx - 1, "real_text_id": prediction})

# === Save Predictions ===
df_pred = pd.DataFrame(predictions)
df_pred.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")
