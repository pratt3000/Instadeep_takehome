# Inside your main script (after training)
from src.model import ProtCNN
import torch
import argparse


def get_argparse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')

    # Add the arguments
    parser.add_argument('--input_seq', type=str, default="NA", help='Input protein sequence')
    parser.add_argument('--checkpoint_dir', type=str, default="model_weights", help='Directory for saved checkpoints')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--num_gpus', type=int, default=0, help='number of gpus to use')
    parser.add_argument('--cpu', action='store_true', help='Use CPU for training')
    parser.add_argument('--max_seq_len', type=int, default=120,
                        help='Max sequence length of protein (avoid changing this)')

    # Parse the arguments
    return parser.parse_args()


if __name__ == "__main__":

    # # Load model weights
    # checkpoint_path = 'lightning_logs/version_9/checkpoints/epoch=2-step=12738.ckpt'
    # model = ProtCNN(17930)
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint["state_dict"])

    # Load model weight [Easier way]
    model = ProtCNN.load_from_checkpoint("lightning_logs/version_10/checkpoints/epoch=2-step=12738.ckpt")
    print(model)
