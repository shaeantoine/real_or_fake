import os 
import pandas as pd

def extract_data(data_dir): 
    if "train" in data_dir:
        # Read in training data labels
        df_labels = pd.read_csv(train_labels)["real_text_id"]

    # Iterate through training data directories and add text files + label to array
    data = []
    i = 0
    for dirpath, _, filenames in sorted(os.walk(train_dir)):
        try: 
            with open(os.path.join(dirpath, filenames[0]), 'r', encoding='utf-8') as f1:
                text1 = f1.read().strip()
            with open(os.path.join(dirpath, filenames[1]), 'r', encoding='utf-8') as f2:
                text2 = f2.read().strip()

            # Appending text from file 1 and file 2 as well as real data label
            if "train" in data_dir:
                data.append((text1, text2, df_labels[i]))
            else:
                data.append((text1, text2))

        except: 
            continue 

        i += 1
    
    # Converting data[] to a Pandas DF
    # Storing new dataset as CSV
    if "train" in data_dir:
        df = pd.DataFrame(data, columns=['file_1', 'file_2', "real_file_label"]) 
        df.to_csv("data/train_df.csv")
    else:
        df = pd.DataFrame(data, columns=['file_1', 'file_2']) 
        df.to_csv("data/test_df.csv")


# Extracting 
train_dir = "data/train/"
train_labels = "data/train.csv"
test_dir = "data/test/"

extract_data(train_dir)
extract_data(test_dir)