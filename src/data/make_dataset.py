import os

import pandas as pd


def reader(partition, data_path, external_test=False):
    """
    Read the data from the files.
    Args:
        partition: train, dev, test
        data_path: path to the data
        external_test: used for the external test set.

    Returns: sequences, labels

    """
    data = []
    folder_path = os.path.join(data_path, partition) if not external_test else data_path

    for file_name in os.listdir(folder_path):
        with open(os.path.join(folder_path, file_name)) as file:
            data.append(pd.read_csv(file, index_col=None, usecols=["sequence", "family_accession"]))

    all_data = pd.concat(data)

    return all_data["sequence"], all_data["family_accession"]


def build_labels(targets):
    """
    Build integer correspondences for each label type in dataset.
    Args:
        targets: targets from the dataset

    Returns:
        fam2label: a dictionary mapping each family to a unique integer.

    """
    unique_targets = targets.unique()
    fam2label = {target: i for i, target in enumerate(unique_targets, start=1)}
    fam2label['<unk>'] = 0

    print(f"There are {len(fam2label)} types of labels.")

    return fam2label
