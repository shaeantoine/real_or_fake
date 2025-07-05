import os 
import pandas as pd

train_dir = "data/train/"
train_labels = "data/train.csv"

# Read in training data labels
df = pd.read_csv(train_labels)
df_labels = df["real_text_id"]

# Iterate through training data directories and add text files + label to array
data = []
i = 0
for dirpath, dirnames, filenames in sorted(os.walk(train_dir)):
    try: 
        with open(os.path.join(dirpath, filenames[0]), 'r', encoding='utf-8') as f1:
            text1 = f1.read().strip()
        with open(os.path.join(dirpath, filenames[1]), 'r', encoding='utf-8') as f2:
            text2 = f2.read().strip()

        # Appending text from file 1 and file 2 as well as real data label
        data.append((text1, text2, df_labels[i]))
    except: 
        continue 

    i += 1

# Converting data[] to a Pandas DF
train_df = pd.DataFrame(data, columns=['file_1', 'file_2', "real_file_label"])

# Storing new dataset as CSV
train_df.to_csv("train_df.csv") 