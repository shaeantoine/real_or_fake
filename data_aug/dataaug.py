import pandas as pd 
from transformers import pipeline

pipe_to_de = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
pipe_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")

def back_translate(text): 
    if pd.isna(text): 
        text = ""
        none += 1
    german_translation = pipe_to_de(text, max_length=3700)[0]["translation_text"]
    english_translation = pipe_to_en(german_translation, max_length=3700)[0]["translation_text"]
    

    return english_translation

train_file = "data/train_df.csv"
train_df = pd.read_csv(train_file)

none_counter = 0
new_data = []
for index, row in train_df.iterrows(): 
    print(f"Appending three datapoints for row with index {index}")

    # Simply append the existing rows
    print(f"Copying exising rows for row {index}")
    text1 = row["file_1"]
    text2 = row["file_2"]
    label = row["real_file_label"]
    new_data.append((text1, text2, label))

    # Backtranslate file 1
    print(f"Backtranslating file 1 for row {index}")
    text1 = back_translate(row["file_1"])
    text2 = row["file_2"]
    label = row["real_file_label"]
    new_data.append((text1, text2, label))

    # Backtranslate file 2
    print(f"Backtranslating file 2 for row {index}")
    text1 = row["file_1"]
    text2 = back_translate(row["file_2"])
    label = row["real_file_label"]
    new_data.append((text1, text2, label))

print(none_counter)
df = pd.DataFrame(new_data, columns=['file_1', 'file_2', "real_file_label"], index=False) 
df.to_csv("data/train_df_aug.csv")