import argparse
import pickle

import torch
import torchmetrics
from torch.utils import data
from tqdm import tqdm

from src.model import ProtCNN
from src.utils import SequenceDataset


def get_argparse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')

    # Add the arguments
    parser.add_argument('--test_set_dir', type=str, default="data/random_split/test", help='Path to the test dataset')
    parser.add_argument('--model_checkpoint', type=str,
                        default="lightning_logs/version_10/checkpoints/epoch=2-step=12738.ckpt",
                        help='Directory for saved checkpoints')
    parser.add_argument('--lang_params', type=str, default="lightning_logs/lang_params.pickle",
                        help='Language params file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers for test dataloader')
    # Parse the arguments
    return parser.parse_args()


def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on the given dataloader.

    Parameters:
    - model: The PyTorch model to be evaluated.
    - dataloader: The DataLoader containing the evaluation data.
    - device: The device on which to perform the evaluation (e.g., "cuda" or "cpu").

    Returns:
    - accuracy: The accuracy of the model on the evaluation data.
    """
    model.eval()  # Set the model to evaluation mode
    accuracy_batches = []

    train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):  # Optional: tqdm for a progress bar
            # Move data to the specified device
            input_data = batch['sequence'].to(device)
            targets = batch['target'].to(device)

            # Forward pass
            preds = model(input_data)

            # Compute predictions
            preds = torch.argmax(preds, dim=1)

            # Update counts
            cur_acc = train_acc(preds, targets)
            accuracy_batches.append(cur_acc)

    acc = sum(accuracy_batches) / len(accuracy_batches)

    return acc


if __name__ == "__main__":

    # argument parser
    args = get_argparse_arguments()

    # Sets the device
    if args.gpu and (torch.cuda.is_available() or torch.backends.mps.is_available()):
        device = "cuda" if torch.cuda.is_available() else "mps"
    else:
        if args.gpu:
            print("Warning: --gpu is set but no GPU is found on this machine. Using CPU instead.")
        device = "cpu"

    # Load model weight [Easier way]
    model = ProtCNN.load_from_checkpoint(args.model_checkpoint).to(device)

    # Get language params for encoding input.
    with open(args.lang_params, 'rb') as handle:
        lang_params = pickle.load(handle)

    # Construct language encoder and encoder input
    lang_encoder = SequenceDataset(lang_params["word2id"], lang_params["fam2label"], lang_params["max_seq_len"], None,
                                   None)

    # Construct dataloader
    test_dataset = SequenceDataset(lang_params["word2id"], lang_params["fam2label"], lang_params["max_seq_len"],
                                   args.test_set_dir, split=None)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Calculate accuracy
    num_classes = len(lang_params["fam2label"])
    accuracy = evaluate_model(model, test_dataloader, device)
    print(f"Accuracy on the evaluation set: {accuracy * 100:.2f}%")
