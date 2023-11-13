import numpy as np
import torch

from src.model import ProtCNN, ResidualBlock


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
    assert isinstance(config, dict)
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
        'target': torch.randint(0, 17930, (1,))
    }
    model = ProtCNN(17930)
    op = model.training_step(batch, 0)
    validator = op.item()
    assert validator is not None
    assert isinstance(validator, float)
    assert not np.isnan(validator)


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
