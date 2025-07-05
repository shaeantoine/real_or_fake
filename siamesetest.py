import torch
import pandas as pd
from transformers import AutoTokenizer

# Internal classes
from siamesenetwork import SiameseNetwork

# Load Test Data
test_file = "data/test_df.csv"
test_df = pd.read_csv(test_file)

# Load Model
model = SiameseNetwork()
device = "cpu"
model_path = "models/trained_model.zip"
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Initialize Model/ Architecture
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Evaluation Loop
predictions = []
with torch.no_grad():
    for idx, row in test_df.iterrows():
        text1 = str(row["file_1"])
        text2 = str(row["file_2"])
        enc1 = tokenizer(text1, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        enc2 = tokenizer(text2, return_tensors="pt", truncation=True, padding='max_length', max_length=512)

        # Properly squeeze inputs for batch size 1
        input_ids_1 = enc1['input_ids'].squeeze(0).to(device)
        attention_mask_1 = enc1['attention_mask'].squeeze(0).to(device)
        input_ids_2 = enc2['input_ids'].squeeze(0).to(device)
        attention_mask_2 = enc2['attention_mask'].squeeze(0).to(device)

        logits = model(
            input_ids_1.unsqueeze(0),
            attention_mask_1.unsqueeze(0),
            input_ids_2.unsqueeze(0),
            attention_mask_2.unsqueeze(0)
        )

        prob = torch.sigmoid(logits).item()
        prediction = "2" if prob > 0.5 else "1"
        predictions.append({"id": idx - 1, "real_text_id": prediction})

# Save Predictions to CSV
output_csv = "predictions.csv"
df_pred = pd.DataFrame(predictions)
df_pred.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")
