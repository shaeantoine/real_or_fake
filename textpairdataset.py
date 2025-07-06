import torch
from torch.utils.data import Dataset

class TextPairDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text1, text2 = str(row['file_1']), str(row['file_2'])
        # Convert [1,2] system to [0,1] for optimizers
        label = 0.0 if row['real_file_label'] == 1 else 1.0

        enc1 = self.tokenizer(text1, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        enc2 = self.tokenizer(text2, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

        return {
            "input_ids_1": enc1["input_ids"].squeeze(),
            "attention_mask_1": enc1["attention_mask"].squeeze(),
            "input_ids_2": enc2["input_ids"].squeeze(),
            "attention_mask_2": enc2["attention_mask"].squeeze(),
            "label": torch.tensor(label)
        }