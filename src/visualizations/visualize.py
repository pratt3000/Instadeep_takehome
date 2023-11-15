import argparse
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.make_dataset import reader


def get_argparse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')

    # Add the arguments
    parser.add_argument('--data_dir', type=str, required=False, default="data/random_split",
                        help='Directory containing the data')
    parser.add_argument('--save_path', type=str, required=False, default="reports/data_visualizations",
                        help='Directory to save the data')
    parser.add_argument('--partition', type=str, default='all', help='partition to visualize')

    # Parse the arguments
    args = parser.parse_args()

    return args


def generate_label_distribution_graph(labels, partition):
    f, ax = plt.subplots(figsize=(8, 5))
    sorted_targets = labels.groupby(labels).size().sort_values(ascending=False)
    sns.histplot(sorted_targets.values, kde=True, log_scale=True, ax=ax)
    plt.title(f"Distribution of family sizes for the '{partition}' split")
    plt.xlabel("Family size (log scale)")
    plt.ylabel("# Families")

    # Save the figure
    plt.savefig(f"{args.save_path}/{partition}_label_distribution_graph.png", dpi=300, bbox_inches='tight')


def generate_seq_len_distribution_graph(data, partition):
    # Plot the distribution of sequences' lengths
    f, ax = plt.subplots(figsize=(8, 5))

    sequence_lengths = data.str.len()
    median = sequence_lengths.median()
    mean = sequence_lengths.mean()

    sns.histplot(sequence_lengths.values, kde=True, log_scale=True, bins=60, ax=ax)

    ax.axvline(mean, color='r', linestyle='-', label=f"Mean = {mean:.1f}")
    ax.axvline(median, color='g', linestyle='-', label=f"Median = {median:.1f}")

    plt.title("Distribution of sequence lengths")
    plt.xlabel("Sequence' length (log scale)")
    plt.ylabel("# Sequences")
    plt.legend(loc="best")

    # Save the figure
    plt.savefig(f"{args.save_path}/{partition}_label_len_distribution_graph.png", dpi=300, bbox_inches='tight')


def generate_aminoacid_freq_distribution_graph(data, partition):
    def get_amino_acid_frequencies(data):
        aa_counter = Counter()

        for sequence in data:
            aa_counter.update(sequence)

        return pd.DataFrame({'AA': list(aa_counter.keys()), 'Frequency': list(aa_counter.values())})

    f, ax = plt.subplots(figsize=(8, 5))

    amino_acid_counter = get_amino_acid_frequencies(data)

    sns.barplot(x='AA', y='Frequency', data=amino_acid_counter.sort_values(by=['Frequency'], ascending=False), ax=ax)

    plt.title("Distribution of AAs' frequencies in the 'train' split")
    plt.xlabel("Amino acid codes")
    plt.ylabel("Frequency (log scale)")
    plt.yscale("log")

    # Save the figure
    plt.savefig(f"{args.save_path}/{partition}_aminoacid_freq_distribution_graph.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    print("This function may take a while to run...")

    # Get the arguments
    args = get_argparse_arguments()

    # Reads data from data files.
    train_data, train_targets = reader("train", args.data_dir)
    valid_data, valid_targets = reader("dev", args.data_dir)
    test_data, test_targets = reader("test", args.data_dir)

    if args.partition == "all":
        input_data = test_data._append(train_data)._append(valid_data)
        label_data = test_targets._append(train_targets)._append(valid_targets)
    elif args.partition == "train":
        input_data = train_data
        label_data = train_targets
    elif args.partition == "valid":
        input_data = valid_data
        label_data = valid_targets
    elif args.partition == "test":
        input_data = test_data
        label_data = test_targets
    else:
        print("Invalid partition: Pl choose between all, train, valid, test.")
        input_data = None
        label_data = None

    if input_data is not None and label_data is not None:

        generate_label_distribution_graph(label_data, args.partition)
        generate_seq_len_distribution_graph(input_data, args.partition)
        generate_aminoacid_freq_distribution_graph(train_data, args.partition)
        print(f"Images saved to {args.data_dir}")
