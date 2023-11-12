import numpy as np
import torch
from src.model import ProtCNN, ResidualBlock
from src.utils import SequenceDataset, Lang
from data.make_dataset import reader, build_labels
import pickle

def test_forward():
    num_classes = np.random.randint(1, 100)
    model = ProtCNN(num_classes)

    x = torch.rand((1, 22, 120))
    y = model.forward(x)
    assert y.shape == torch.rand((1, num_classes)).shape


def test_configure_optimizers():
    num_classes = np.random.randint(1, 100)
    model = ProtCNN(num_classes)
    config = model.configure_optimizers()
    assert config is not None
    assert isinstance(config['optimizer'], torch.optim.Optimizer)
    assert isinstance(config['lr_scheduler'], torch.optim.lr_scheduler.MultiStepLR)

    model = ProtCNN(num_classes, optimizer='adam')
    config = model.configure_optimizers()
    assert config is not None
    assert isinstance(config['optimizer'], torch.optim.Optimizer)
    assert isinstance(config['lr_scheduler'], torch.optim.lr_scheduler.MultiStepLR)


def test_residual_block():
    num_classes = np.random.randint(1, 100)
    model = ResidualBlock(120, 120)
    x = torch.rand((1, 120, 120))
    y = model.forward(x)
    assert y.shape == torch.rand((1, 120, 120)).shape


def test_training_step():

    batch = {
        'sequence': torch.rand((1, 22, 120)),
        'target': torch.randint(0, 17930, (1, ))
    }
    model = ProtCNN(17930)
    op = model.training_step(batch, 0)

    assert op.item() is not None
    assert isinstance(op.item(), float)
    assert not np.isnan(op.item())


def test_validation_step():
    batch = {
        'sequence': torch.rand((1, 22, 120)),
        'target': torch.randint(0, 17930, (1,))
    }
    model = ProtCNN(17930)
    op = model.validation_step(batch, 0)

    assert op.item() is not None
    assert isinstance(op.item(), float)
    assert not np.isnan(op.item())

    pass


def test_Lang():
    lang = Lang()
    train_data, train_targets = reader("train", "data/random_split")
    word2id = lang.build_vocab(train_data)

    assert isinstance(word2id, dict)
    assert len(word2id) > 2
    assert isinstance(word2id['<pad>'], int)
    assert isinstance(word2id['<unk>'], int)

def test_SequenceDataset():
    # Get language params for encoding input.
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


def test_build_labels():
    _, targets = reader("test", "data/random_split")
    fam2label = build_labels(targets)
    assert isinstance(fam2label, dict)
    assert len(fam2label) > 2
    assert isinstance(fam2label['<unk>'], int)


# model.py
test_forward()
test_configure_optimizers()
test_residual_block()
test_training_step()
test_validation_step()

# utils.py
test_Lang()
test_SequenceDataset()

# make_dataset.py
test_build_labels()