import numpy as np
from src.model import ProtCNN


def test_forward():
    x = np.array([1, 2, 3])
    model = ProtCNN()
    y = model.forward(x)
    assert len(y) == x.shape[0]