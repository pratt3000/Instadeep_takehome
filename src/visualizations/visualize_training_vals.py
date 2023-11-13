import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.logger import logger


def get_argparse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')

    # Add the arguments
    parser.add_argument('--metrics_file', type=str, required=False, default="lightning_logs/version_10/metrics.csv",
                        help='Directory containing the metrics file')
    parser.add_argument('--save_path', type=str, required=False, default="reports/training_visualizations",
                        help='Directory to save the data')

    # Parse the arguments
    args = parser.parse_args()

    return args


def generate_graph_acc(df, save_path, x_axis):
    # Set the style of seaborn
    sns.set_style("darkgrid")

    # Plotting train accuracy step
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x=x_axis, y="train_acc_step", label=f"Train Accuracy {x_axis}")

    # Plotting valid acc step
    sns.lineplot(data=df, x=x_axis, y="valid_acc", label=f"valid Accuracy {x_axis}")

    # Adding title and labels
    plt.title(f'Training Accuracy and Loss per {x_axis}')
    plt.xlabel(x_axis)
    plt.ylabel('Value')

    # Show the legend
    plt.legend()

    # Display the plot
    plt.savefig(f"{save_path}/training_params_graph_{x_axis}_acc.png", dpi=300, bbox_inches='tight')


def generate_graph_loss(df, save_path, x_axis):
    # Set the style of seaborn
    sns.set_style("darkgrid")

    # Plotting train accuracy step
    plt.figure(figsize=(10, 5))

    # Plotting train loss step
    sns.lineplot(data=df, x=x_axis, y="train_loss_step", label=f"Train Loss {x_axis}")

    # Adding title and labels
    plt.title(f'Loss per {x_axis}')
    plt.xlabel(x_axis)
    plt.ylabel('Value')

    # Show the legend
    plt.legend()

    # Display the plot
    plt.savefig(f"{save_path}/training_params_graph_{x_axis}_loss.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    logger.info("This function may take a while to run...")

    # Get the arguments
    args = get_argparse_arguments()

    # Load the metrics file
    metrics_df = pd.read_csv(args.metrics_file)

    # Plot the training acc and validation acc
    generate_graph_acc(metrics_df, args.save_path, 'step')
    generate_graph_acc(metrics_df, args.save_path, 'epoch')

    # Plot the training loss
    generate_graph_loss(metrics_df, args.save_path, 'epoch')
    generate_graph_loss(metrics_df, args.save_path, 'step')
