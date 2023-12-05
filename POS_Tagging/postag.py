import torch
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import numpy as np

# Função para converter tags em índices
def tag_2_id():
    return {'ADJ': 0, 'ADV': 1, 'ADV-KS': 2, 'ART': 3, 'CUR': 4, 'IN': 5, 'KC': 6, 'KS': 7, 'N': 8,
            'NPROP': 9, 'NUM': 10, 'PCP': 11, 'PDEN': 12, 'PREP': 13, 'PREP+ADV': 14, 'PREP+ART': 15,
            'PREP+PRO-KS': 16, 'PREP+PROADJ': 17, 'PREP+PROPESS': 18, 'PREP+PROSUB': 19, 'PRO-KS': 20,
            'PROADJ': 21, 'PROPESS': 22, 'PROSUB': 23, 'PU': 24, 'V': 25}

# Classe para o dataset
class PoSDataset(Dataset):
    def __init__(self, filename, tokenizer, max_len, label_map):
        self.sentences = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = label_map

        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                tokens, tags = [], []
                for word in line.strip().split():
                    token, tag = word.rsplit('_', 1)
                    tokens.append(token)
                    tags.append(self.label_map[tag])
                self.sentences.append(tokens)
                self.labels.append(tags)

        self.labels = [pad_labels(label, max_len) for label in self.labels]
        self.encodings = self.tokenizer(self.sentences, is_split_into_words=True, 
                                        return_offsets_mapping=True, padding='max_length', 
                                        truncation=True, max_length=max_len)
        self.encodings.pop("offset_mapping")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def pad_labels(labels, max_len, pad_label_idx=-100):
    return labels + [pad_label_idx] * (max_len - len(labels))