import os
import json
import urllib
import ssl
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utility_functions import format_input


URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
DATA_FILE_PATH ="instruction-data.json"

import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def download_and_load_file(file_path, url):
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url, context=ssl_context) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

def load_data(tokenizer,batch_size=8, train_frac=0.85, test_frac=0.1,num_workers=0):
    # Download & preprocess
  
    data = download_and_load_file(DATA_FILE_PATH, URL)
    train_portion = int(len(data) * train_frac)  # 85% for training
    test_portion = int(len(data) * test_frac)    # 10% for testing
    val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    # Loaders

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=num_workers)
   
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=num_workers)

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_data.max_length, val_data







