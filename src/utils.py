import numpy as np
import torch

from data.make_dataset import reader


class Lang:
    def __init__(self):
        self.voc = set()
        self.rare_AAs = {'X', 'U', 'B', 'O', 'Z'}
        self.default_tokens = {'<pad>': 0, '<unk>': 1}

    def build_vocab(self, data):
        # Build the vocabulary

        for sequence in data:
            self.voc.update(sequence)

        unique_AAs = sorted(self.voc - self.rare_AAs)

        # Build the mapping
        word2id = {w: i for i, w in enumerate(unique_AAs, start=2)}
        word2id['<pad>'] = self.default_tokens['<pad>']
        word2id['<unk>'] = self.default_tokens['<unk>']

        return word2id


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, word2id, fam2label, max_len, data_path=None, split=None):
        self.word2id = word2id
        self.fam2label = fam2label
        self.max_len = max_len

        if data_path and not split:  # For external tests
            self.data, self.label = reader("NA", data_path, external_test=True)
        if data_path and split:  # For when we are training
            self.data, self.label = reader(split, data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.preprocess(self.data.iloc[index])
        label = self.fam2label.get(self.label.iloc[index], self.fam2label['<unk>'])

        return {'sequence': seq, 'target': label}

    def encode_single_sample(self, x, y=None):
        x = self.preprocess(x)
        if y:
            y = self.fam2label.get(y, self.fam2label['<unk>'])
        return x, y

    def preprocess(self, text):
        seq = []

        # Encode into IDs
        for word in text[:self.max_len]:
            seq.append(self.word2id.get(word, self.word2id['<unk>']))

        # Pad to maximal length
        if len(seq) < self.max_len:
            seq += [self.word2id['<pad>'] for _ in range(self.max_len - len(seq))]

        # Convert list into tensor
        seq = torch.from_numpy(np.array(seq))

        # One-hot encode    
        one_hot_seq = torch.nn.functional.one_hot(seq, num_classes=len(self.word2id), )

        # Permute channel (one-hot) dim first
        one_hot_seq = one_hot_seq.permute(1, 0)

        return one_hot_seq
