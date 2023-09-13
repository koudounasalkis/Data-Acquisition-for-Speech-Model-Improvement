import torch
import librosa
import numpy as np
import copy


""" Dataset Class """
class Dataset(torch.utils.data.Dataset):
    def __init__(self, examples, feature_extractor, max_duration):
        self.examples = examples['path']
        self.labels = examples['label']
        self.feature_extractor = feature_extractor
        self.max_duration = max_duration

    def __getitem__(self, idx):
        inputs = self.feature_extractor(
            librosa.load(self.examples[idx], sr=16_000)[0].squeeze(),
            sampling_rate=self.feature_extractor.sampling_rate, 
            return_tensors="pt",
            max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
            truncation=True,
            padding='max_length'
            )
        item = {'input_values': inputs['input_values'].squeeze(0)}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.examples)