"""
Training script for TASA sleep staging model.

Trains the multi-task TASA model with:
- Sleep stage classification (5-class)
- Transition detection (binary)
- Homoscedastic uncertainty weighting
- Optional alpha priority bias for transition task

Usage:
    python train.py --data-dir ./processed_data --output-dir ./results
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tasa import TASAModel, UncertaintyLossWrapper, get_device
from data.dataset import SleepDataset, get_subject_splits


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_wrapper: UncertaintyLossWrapper,
    stage_criterion: nn.Module,
    transition_criterion: nn.Module,
    device: torch.device,
    alpha: float = 5.0
) -> dict:
    """
    Train for one epoch.

    Args:
        model: TASA model
        dataloader: Training data loader
        optimizer: Optimizer
        loss_wrapper: Uncertainty loss wrapper
        stage_criterion: CrossEntropyLoss for staging
        transition_criterion: BCEWithLogitsLoss for transitions
        device: Compute device
        alpha: Priority weight for transition loss (default: 5.0)

    Returns:
        Dict of training metrics
    """
    model.train()

    total_loss = 0.0
    total_stage_loss = 0.0
    total_transition_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        signals = batch["signal"].to(device)
        stage_labels = batch["stage_label"].to(device)
        transition_labels = batch["transition_label"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(signals)
        stage_logits = outputs["stage_logits"]
        transition_logits = outputs["transition_logits"]

        # Compute individual losses
        stage_loss = stage_criterion(stage_logits, stage_labels)
        transition_loss = transition_criterion(
            transition_logits.squeeze(-1), transition_labels
        )

        # Apply alpha weighting to transition loss
        transition_loss_weighted = alpha * transition_loss

        # Combined loss with uncertainty weighting
        combined_loss = loss_wrapper([stage_loss, transition_loss_weighted])

        # Backward pass
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += combined_loss.item()
        total_stage_loss += stage_loss.item()
        total_transition_loss += transition_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "stage_loss": total_stage_loss / n_batches,
        "transition_loss": total_transition_loss / n_batches
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    stage_criterion: nn.Module,
    transition_criterion: nn.Module,
    device: torch.device
) -> dict:
    """
    Validate the model.

    Args:
        model: TASA model
        dataloader: Validation data loader
        stage_criterion: CrossEntropyLoss for staging
        transition_criterion: BCEWithLogitsLoss for transitions
        device: Compute device

    Returns:
        Dict of validation metrics including predictions
    """
    model.eval()

    total_stage_loss = 0.0
    total_transition_loss = 0.0
    n_batches = 0

    all_predictions = []
    all_labels = []
    all_transition_probs = []
    all_transition_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            signals = batch["signal"].to(device)
            stage_labels = batch["stage_label"].to(device)
            transition_labels = batch["transition_label"].to(device)

            outputs = model(signals)
            stage_logits = outputs["stage_logits"]
            transition_logits = outputs["transition_logits"]

            # Losses
            stage_loss = stage_criterion(stage_logits, stage_labels)
            transition_loss = transition_criterion(
                transition_logits.squeeze(-1), transition_labels
            )

            total_stage_loss += stage_loss.item()
            total_transition_loss += transition_loss.item()
            n_batches += 1

            # Collect predictions
            predictions = stage_logits.argmax(dim=1).cpu().numpy()
            labels = stage_labels.cpu().numpy()
            transition_probs = torch.sigmoid(transition_logits).squeeze(-1).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_transition_probs.extend(transition_probs)
            all_transition_labels.extend(transition_labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_transition_probs = np.array(all_transition_probs)

    # Calculate metrics
    accuracy = (all_predictions == all_labels).mean()
    kappa = cohen_kappa_score(all_labels, all_predictions)

    return {
        "stage_loss": total_stage_loss / n_batches,
        "transition_loss": total_transition_loss / n_batches,
        "accuracy": accuracy,
        "kappa": kappa,
        "predictions": all_predictions,
        "labels": all_labels,
        "transition_probs": all_transition_probs
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train TASA sleep staging model"
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
        default=Path("./results"),
        help="Output directory for checkpoints and results"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=5.0,
        help="Priority weight for transition loss (default: 5.0)"
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=64,
        help="Model dimension"
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=64,
        help="Local attention window size"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "training.log")
        ]
    )
    logger = logging.getLogger(__name__)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device setup
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load data splits
    metadata_file = args.data_dir / "metadata.csv"
    train_subjects, val_subjects, test_subjects = get_subject_splits(
        metadata_file, seed=args.seed
    )
    logger.info(f"Subjects - Train: {len(train_subjects)}, "
                f"Val: {len(val_subjects)}, Test: {len(test_subjects)}")

    # Create datasets
    train_dataset = SleepDataset(args.data_dir, subject_ids=train_subjects)
    val_dataset = SleepDataset(args.data_dir, subject_ids=val_subjects)
    test_dataset = SleepDataset(args.data_dir, subject_ids=test_subjects)

    logger.info(f"Samples - Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Weighted sampler for class imbalance
    stage_labels = [s["stage_label"] for s in train_dataset.samples]
    class_counts = np.bincount(stage_labels, minlength=5)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in stage_labels]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_dataset), replacement=True
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Initialize model
    model = TASAModel(
        input_channels=3,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        window_size=args.window_size,
        num_classes=5
    ).to(device)

    logger.info(f"Model parameters: {model.count_parameters():,}")
    logger.info(f"Alpha (transition weight): {args.alpha}")

    # Loss functions
    stage_criterion = nn.CrossEntropyLoss()
    transition_criterion = nn.BCEWithLogitsLoss()
    loss_wrapper = UncertaintyLossWrapper(num_tasks=2).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_wrapper.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Training loop
    best_kappa = -1.0
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_wrapper,
            stage_criterion, transition_criterion, device, alpha=args.alpha
        )
        history["train"].append(train_metrics)

        # Validate
        val_metrics = validate(
            model, val_loader, stage_criterion, transition_criterion, device
        )
        history["val"].append({
            "stage_loss": val_metrics["stage_loss"],
            "transition_loss": val_metrics["transition_loss"],
            "accuracy": val_metrics["accuracy"],
            "kappa": val_metrics["kappa"]
        })

        # Log progress
        task_weights = loss_wrapper.get_task_weights()
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Stage: {train_metrics['stage_loss']:.4f}, "
            f"Trans: {train_metrics['transition_loss']:.4f}"
        )
        logger.info(
            f"Val - Acc: {val_metrics['accuracy']*100:.2f}%, "
            f"Kappa: {val_metrics['kappa']:.4f}, "
            f"Stage: {val_metrics['stage_loss']:.4f}, "
            f"Trans: {val_metrics['transition_loss']:.4f}"
        )
        logger.info(
            f"Task weights - Stage: {task_weights['stage_weight']:.4f}, "
            f"Trans: {task_weights['transition_weight']:.4f}"
        )

        # Save best model
        if val_metrics["kappa"] > best_kappa:
            best_kappa = val_metrics["kappa"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss_wrapper_state_dict": loss_wrapper.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "kappa": best_kappa,
                "args": vars(args)
            }, args.output_dir / "best_model.pt")
            logger.info(f"Saved new best model (kappa: {best_kappa:.4f})")

        scheduler.step()

    # Final evaluation on test set
    logger.info("\n" + "="*60)
    logger.info("Final evaluation on test set")
    logger.info("="*60)

    # Load best model
    checkpoint = torch.load(args.output_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = validate(
        model, test_loader, stage_criterion, transition_criterion, device
    )

    logger.info(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    logger.info(f"Test Kappa: {test_metrics['kappa']:.4f}")

    # Save final results
    results = {
        "final_accuracy": float(test_metrics["accuracy"]),
        "final_kappa": float(test_metrics["kappa"]),
        "best_val_kappa": float(best_kappa),
        "args": vars(args),
        "timestamp": datetime.now().isoformat()
    }

    # Convert Path objects to strings for JSON serialization
    results["args"]["data_dir"] = str(results["args"]["data_dir"])
    results["args"]["output_dir"] = str(results["args"]["output_dir"])

    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions
    np.savez(
        args.output_dir / "predictions.npz",
        predictions=test_metrics["predictions"],
        labels=test_metrics["labels"],
        transition_probs=test_metrics["transition_probs"]
    )

    logger.info(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
