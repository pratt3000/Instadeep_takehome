import torch
import argparse

from data.make_dataset import reader, build_labels
from src.utils import Lang

def get_argparse_arguments():

    # Create the parser
    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')

    # Add the arguments
    parser.add_argument('--data_dir', type=str, default="data/random_split", help='Directory containing the data')
    parser.add_argument('--model_dir', type=str, default="model_weights", help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for training')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer for training')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 penalty)')
    parser.add_argument('--seed', type=int, default=420, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=100, help='Interval for logging')
    parser.add_argument('--eval_interval', type=int, default=500, help='Interval for evaluation')
    parser.add_argument('--save_interval', type=int, default=500, help='Interval for saving the model')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--cpu', action='store_true', help='Use CPU for training')
    parser.add_argument('--early_stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--resume', type=str, help='Resume training from this checkpoint')
    parser.add_argument('--config', type=str, help='Configuration file')

    # Parse the arguments
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    # Runs argparse
    args = get_argparse_arguments()

    # Sets the device
    if args.gpu and (torch.cuda.is_available() or torch.backends.mps.is_available()):
        device = "cuda" if torch.cuda.is_available() else "mps"
    else:
        if args.gpu:
            print("Warning: --gpu is set but no GPU is found on this machine. Using CPU instead.")
        device = "cpu"

    # Reads data from data files.
    train_data, train_targets = reader("train", args.data_dir)
    valid_data, valid_targets = reader("dev", args.data_dir)
    test_data, test_targets = reader("test", args.data_dir)

    # Build integer correspondences for each label type in dataset.
    fam2label = build_labels(train_targets) # fam2label is a dictionary mapping each family to a unique integer.
    
    # Build and get language.
    lang = Lang()
    word2id = lang.build_vocab(train_data)
    print(device, word2id)





    




