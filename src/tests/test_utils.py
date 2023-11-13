import pickle

import torch

from src.data.make_dataset import reader
from src.utils import SequenceDataset, Lang


def test_SequenceDataset():

    with open("lightning_logs/lang_params.pickle", 'rb') as handle:
        lang_params = pickle.load(handle)

    # Construct language encoder and encoder input
    lang_encoder = SequenceDataset(
        lang_params["word2id"],
        lang_params["fam2label"],
        lang_params["max_seq_len"],
        None,
        None
    )
    x_encoded = lang_encoder.encode_single_sample("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert x_encoded[0].shape == torch.rand((22, 120)).shape
    assert x_encoded[0].dtype == torch.int64

    x_encoded, y = lang_encoder.encode_single_sample("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "NA")
    assert x_encoded.shape == torch.rand((22, 120)).shape
    assert x_encoded.dtype == torch.int64
    assert isinstance(y, int)

    lang_encoder = SequenceDataset(
        lang_params["word2id"],
        lang_params["fam2label"],
        lang_params["max_seq_len"],
        "data/random_split/test",
        None
    )

    assert lang_encoder.data is not None
    assert lang_encoder.label is not None
    assert len(lang_encoder.data) == len(lang_encoder.label)

    lang_encoder = SequenceDataset(
        lang_params["word2id"],
        lang_params["fam2label"],
        lang_params["max_seq_len"],
        "data/random_split",
        "test"
    )

    assert lang_encoder.data is not None
    assert lang_encoder.label is not None
    assert len(lang_encoder.data) == len(lang_encoder.label)

def test_build_vocab():
    lang = Lang()
    train_data, train_targets = reader("train", "data/random_split")
    word2id = lang.build_vocab(train_data)

    assert isinstance(word2id, dict)
    assert len(word2id.keys()) > 2
    assert isinstance(word2id['<pad>'], int)
    assert isinstance(word2id['<unk>'], int)
