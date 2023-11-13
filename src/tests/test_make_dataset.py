from src.data.make_dataset import reader, build_labels


def test_build_labels():
    _, targets = reader("test", "data/random_split")
    fam2label = build_labels(targets)
    assert isinstance(fam2label, dict)
    assert len(fam2label.keys()) > 2
    assert isinstance(fam2label['<unk>'], int)
