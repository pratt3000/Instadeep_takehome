from src.model import ProtCNN
import argparse
import pickle
from src.utils import SequenceDataset
import torch

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

    # Load model weight [Easier way]
    model = ProtCNN.load_from_checkpoint("lightning_logs/version_10/checkpoints/epoch=2-step=12738.ckpt")

    # Get language params for encoding input.
    lang_params_file = "lightning_logs/lang_params.pickle"
    with open(lang_params_file, 'rb') as handle:
        lang_params = pickle.load(handle)

    # Construct language encoder and encoder input
    lang_encoder = SequenceDataset(lang_params["word2id"], lang_params["fam2label"], lang_params["max_seq_len"], None, None)
    x_encoded = lang_encoder.encode_single_sample("INPUT", y=None)
    x_encoded = x_encoded[0].reshape((1, 22, 120)).to("mps") # formatting

    pred = model(x_encoded)


