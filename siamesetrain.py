import torch
import pandas as pd
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Internal classes
from siamesenetwork import SiameseNetwork
from textpairdataset import TextPairDataset

# Load in training data
train_data = "data/train_df.csv"
train_df = pd.read_csv(train_data)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=1)

print("Number of file_1 being real in training set:", sum(train_df["real_file_label"] == 1))
print("Number of file_2 being real in training set:", sum(train_df["real_file_label"] == 2))
print("Number of file_1 being real in validation set:", sum(val_df["real_file_label"] == 1))
print("Number of file_2 being real in validation set:", sum(val_df["real_file_label"] == 2))

# Initialize Model/ Architecture
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = TextPairDataset(train_df, tokenizer)
val_dataset = TextPairDataset(val_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

model = SiameseNetwork()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Freezing the model's encoder
for param in model.encoder.parameters():
    param.requires_grad = False 

optimizer = AdamW(model.parameters(), lr=1e-4)
class_prop = sum(train_df["real_file_label"] == 1)/sum(train_df["real_file_label"] == 2)
pos_weight = torch.tensor([class_prop], dtype=torch.float).to(device)
#loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss_fn = nn.BCEWithLogitsLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True)


# Training Loop 
best_val_acc = 0
patience = 10
epochs_no_improve = 0
num_epochs = 10

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    model.train()
    train_loss = 0
    correct = 0
    total = 0
    all_preds=[]
    all_labels=[]

    for batch in train_loader:
        input_ids_1 = batch["input_ids_1"].to(device)
        attention_mask_1 = batch["attention_mask_1"].to(device)
        input_ids_2 = batch["input_ids_2"].to(device)
        attention_mask_2 = batch["attention_mask_2"].to(device)
        labels = batch["label"].float().to(device)

        optimizer.zero_grad()
        logits = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

        all_preds += preds.cpu().numpy().tolist()
        all_labels += labels.cpu().numpy().tolist()

    train_acc = correct / total
    avg_train_loss = train_loss / len(train_loader)
    train_f1 = f1_score(all_labels, all_preds)

    print(f"Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids_1 = batch["input_ids_1"].to(device)
            attention_mask_1 = batch["attention_mask_1"].to(device)
            input_ids_2 = batch["input_ids_2"].to(device)
            attention_mask_2 = batch["attention_mask_2"].to(device)
            labels = batch["label"].float().to(device)

            outputs = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).long()
            val_correct += (preds == labels.long()).sum().item()
            val_total += labels.size(0)

            val_preds += preds.cpu().numpy().tolist()
            val_labels += labels.cpu().numpy().tolist()

    val_acc = val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(val_labels, val_preds)
    
    scheduler.step(avg_val_loss)

    print(f"Val   Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
    print(f"\nPredicted labels of the validation set:\n{val_preds}\n")
    print(f"Actual labels of the validation set: \n{val_labels}\n")

    # Saving the model, checking for no improvement after patience
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "models/siamese_network.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break