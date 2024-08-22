from torch.utils.data import Dataset, Sampler
import torch
from transformers import GPT2TokenizerFast


class CustomDataset(Dataset):
    def __init__(self, text_file, config, tokenizer=None):
        with open(text_file, 'r', encoding='utf-8') as f:
            self.text_data = f.read()
        self.config = config
        self.tokenizer = tokenizer if tokenizer else GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenized_data = self.tokenizer.encode(self.text_data, return_tensors='pt')[0]

    def __getitem__(self, index):
        curr_win = self.tokenized_data[index:index + self.config.block_size + 1]
        input_text = curr_win[:self.config.block_size]
        target_text = curr_win[1:self.config.block_size + 1]
        return input_text, target_text

    def __len__(self):
        return len(self.tokenized_data) - self.config.block_size


class CustomSampler(Sampler):
    def __init__(self, data, block_size, replacement=False, num_samples=None):
        self.data = data
        self.block_size = block_size
        self.replacement = replacement
        self._num_samples = num_samples

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data)
        return self._num_samples

    def __iter__(self):
        n = len(self.data) - self.block_size
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples


