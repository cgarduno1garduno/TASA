"""
Evaluation script for TASA sleep staging model.

Loads a trained model checkpoint and evaluates on test set,
generating confusion matrix, per-class metrics, and visualizations.

Usage:
    python evaluate.py --checkpoint ./results/best_model.pt --data-dir ./processed_data
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, cohen_kappa_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tasa import TASAModel, get_device
from data.dataset import SleepDataset, get_subject_splits


STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> dict:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained TASA model
        dataloader: Data loader for evaluation
        device: Compute device

    Returns:
        Dict containing predictions, labels, and transition probabilities
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_transition_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            signals = batch["signal"].to(device)
            stage_labels = batch["stage_label"]

            outputs = model(signals)
            predictions = outputs["stage_logits"].argmax(dim=1).cpu().numpy()
            transition_probs = torch.sigmoid(
                outputs["transition_logits"]
            ).squeeze(-1).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(stage_labels.numpy())
            all_transition_probs.extend(transition_probs)

    return {
        "predictions": np.array(all_predictions),
        "labels": np.array(all_labels),
        "transition_probs": np.array(all_transition_probs)
    }


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
    normalize: bool = True
):
    """
    Generate and save confusion matrix visualization.

    Args:
        labels: Ground truth labels
        predictions: Model predictions
        output_path: Path to save figure
        normalize: Whether to normalize the matrix
    """
    cm = confusion_matrix(labels, predictions)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=STAGE_NAMES,
        yticklabels=STAGE_NAMES,
        vmin=0,
        vmax=1 if normalize else None
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title("Sleep Stage Classification - Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")


def plot_transition_heatmap(
    labels: np.ndarray,
    transition_probs: np.ndarray,
    output_path: Path,
    max_epochs: int = 500
):
    """
    Generate hypnogram with transition probability heatmap.

    Args:
        labels: Ground truth sleep stage labels
        transition_probs: Model's transition probability predictions
        output_path: Path to save figure
        max_epochs: Maximum epochs to display
    """
    # Limit to max_epochs for visualization
    n_epochs = min(len(labels), max_epochs)
    labels = labels[:n_epochs]
    transition_probs = transition_probs[:n_epochs]
    epochs = np.arange(n_epochs)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.1}
    )

    # Panel 1: Hypnogram
    ax1 = axes[0]
    ax1.step(epochs, labels, where="mid", color="navy", linewidth=1.2)
    ax1.fill_between(epochs, labels, step="mid", alpha=0.3, color="navy")
    ax1.set_ylim(-0.5, 4.5)
    ax1.set_yticks([0, 1, 2, 3, 4])
    ax1.set_yticklabels(STAGE_NAMES)
    ax1.invert_yaxis()
    ax1.set_ylabel("Sleep Stage", fontsize=12, fontweight="bold")
    ax1.set_title("Ground Truth Hypnogram", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Panel 2: Transition probability heatmap
    ax2 = axes[1]
    prob_matrix = transition_probs.reshape(1, -1)
    im = ax2.imshow(
        prob_matrix,
        aspect="auto",
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        extent=[0, n_epochs, 0, 1]
    )
    ax2.set_yticks([])
    ax2.set_ylabel("Trans.\nProb.", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Epochs (30s)", fontsize=12, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, orientation="vertical", pad=0.02, aspect=15)
    cbar.set_label("Probability", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Transition heatmap saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TASA sleep staging model"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to preprocessed data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as checkpoint)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (must match training)"
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.checkpoint.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = get_device()
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})

    # Initialize model with saved hyperparameters
    model = TASAModel(
        input_channels=3,
        d_model=train_args.get("d_model", 64),
        n_layers=train_args.get("n_layers", 4),
        n_heads=train_args.get("n_heads", 4),
        window_size=train_args.get("window_size", 64),
        num_classes=5
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model parameters: {model.count_parameters():,}")

    # Load test data
    metadata_file = args.data_dir / "metadata.csv"
    _, _, test_subjects = get_subject_splits(metadata_file, seed=args.seed)
    print(f"Test subjects: {len(test_subjects)}")

    test_dataset = SleepDataset(args.data_dir, subject_ids=test_subjects)
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Evaluate
    results = evaluate_model(model, test_loader, device)

    # Calculate metrics
    accuracy = (results["predictions"] == results["labels"]).mean()
    kappa = cohen_kappa_score(results["labels"], results["predictions"])

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"\nPer-class metrics:")
    print(classification_report(
        results["labels"],
        results["predictions"],
        target_names=STAGE_NAMES,
        digits=4
    ))

    # Generate visualizations
    plot_confusion_matrix(
        results["labels"],
        results["predictions"],
        args.output_dir / "confusion_matrix.png"
    )

    plot_transition_heatmap(
        results["labels"],
        results["transition_probs"],
        args.output_dir / "transition_heatmap.png"
    )

    # Save results
    eval_results = {
        "accuracy": float(accuracy),
        "kappa": float(kappa),
        "checkpoint": str(args.checkpoint),
        "n_test_samples": len(test_dataset)
    }

    with open(args.output_dir / "evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # Save predictions
    np.savez(
        args.output_dir / "test_predictions.npz",
        predictions=results["predictions"],
        labels=results["labels"],
        transition_probs=results["transition_probs"]
    )

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
