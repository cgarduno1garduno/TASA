"""
Sleep-EDF Dataset loader for TASA model.

Provides PyTorch Dataset for loading preprocessed sleep staging data
with per-channel Z-score normalization.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SleepDataset(Dataset):
    """
    PyTorch Dataset for preprocessed Sleep-EDF epochs.

    Loads .npy epoch files and applies per-channel Z-score normalization.

    Args:
        data_dir: Path to directory containing .npy files and metadata.csv
        metadata_file: Path to metadata CSV (default: data_dir/metadata.csv)
        subject_ids: Optional list of subject IDs to include (for splits)
        transform: Optional transform to apply to data
    """

    def __init__(
        self,
        data_dir: str,
        metadata_file: Optional[str] = None,
        subject_ids: Optional[List[str]] = None,
        transform=None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Load metadata
        if metadata_file is None:
            metadata_file = self.data_dir / "metadata.csv"
        else:
            metadata_file = Path(metadata_file)

        self.samples = []
        with open(metadata_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if subject_ids is None or row["subject_id"] in subject_ids:
                    self.samples.append({
                        "filename": row["filename"],
                        "subject_id": row["subject_id"],
                        "epoch_index": int(row["epoch_index"]),
                        "stage_label": int(row["stage_label"])
                    })

        # Build sequential pairs for transition labels
        self._build_transition_labels()

    def _build_transition_labels(self):
        """
        Compute transition labels based on consecutive epochs.

        A transition occurs when the sleep stage changes from
        the current epoch to the next epoch.
        """
        # Group samples by subject
        subject_epochs: Dict[str, List[int]] = {}
        for idx, sample in enumerate(self.samples):
            sid = sample["subject_id"]
            if sid not in subject_epochs:
                subject_epochs[sid] = []
            subject_epochs[sid].append(idx)

        # Sort each subject's epochs by epoch_index
        for sid in subject_epochs:
            subject_epochs[sid].sort(
                key=lambda i: self.samples[i]["epoch_index"]
            )

        # Assign transition labels
        for indices in subject_epochs.values():
            for i, idx in enumerate(indices):
                if i < len(indices) - 1:
                    next_idx = indices[i + 1]
                    current_stage = self.samples[idx]["stage_label"]
                    next_stage = self.samples[next_idx]["stage_label"]
                    self.samples[idx]["transition_label"] = int(
                        current_stage != next_stage
                    )
                else:
                    # Last epoch in sequence - no transition
                    self.samples[idx]["transition_label"] = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single epoch with normalized signal.

        Returns:
            Dict with keys:
                - 'signal': Normalized signal tensor (C, T)
                - 'stage_label': Sleep stage label (0-4)
                - 'transition_label': Binary transition label (0 or 1)
        """
        sample = self.samples[idx]
        filepath = self.data_dir / sample["filename"]

        # Load signal data
        signal = np.load(filepath)  # Shape: (C, T) e.g., (3, 3000)
        signal = torch.from_numpy(signal).float()

        # Per-channel Z-score normalization
        # Normalize each channel independently
        mean = signal.mean(dim=1, keepdim=True)
        std = signal.std(dim=1, keepdim=True)
        signal = (signal - mean) / (std + 1e-6)

        # Clamp extreme values to [-20, 20]
        signal = torch.clamp(signal, min=-20.0, max=20.0)

        if self.transform:
            signal = self.transform(signal)

        return {
            "signal": signal,
            "stage_label": torch.tensor(sample["stage_label"], dtype=torch.long),
            "transition_label": torch.tensor(
                sample["transition_label"], dtype=torch.float32
            )
        }


def get_subject_splits(
    metadata_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split subjects into train/val/test sets.

    Uses subject-level splitting to prevent data leakage between splits.

    Args:
        metadata_file: Path to metadata CSV
        train_ratio: Fraction of subjects for training (default: 0.8)
        val_ratio: Fraction of subjects for validation (default: 0.1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_subjects, val_subjects, test_subjects)
    """
    # Get unique subjects
    subjects = set()
    with open(metadata_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subjects.add(row["subject_id"])

    subjects = sorted(list(subjects))

    # Shuffle with seed
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    # Calculate split indices
    n_subjects = len(subjects)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]

    return train_subjects, val_subjects, test_subjects
