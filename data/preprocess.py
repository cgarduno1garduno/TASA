"""
Sleep-EDF Database Preprocessing Script

Preprocesses the Sleep-EDF Expanded dataset for sleep stage classification.
Extracts 30-second epochs from PSG recordings with corresponding hypnogram labels.

Dataset: Sleep-EDF Database Expanded (Physionet)
URL: https://physionet.org/content/sleep-edfx/1.0.0/

Channels used:
- EEG Fpz-Cz (frontal-central EEG)
- EEG Pz-Oz (parietal-occipital EEG)
- EOG horizontal (electrooculogram)

Usage:
    python preprocess.py --data-dir /path/to/sleep-edf --output-dir ./processed_data
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import mne
import numpy as np
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

# Target channels for TASA model
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]

# Target sampling rate (Hz)
SAMPLING_RATE = 100

# Epoch duration (seconds)
EPOCH_LENGTH = 30

# Sleep stage mapping (AASM standard)
# Sleep-EDF annotations -> numeric labels
STAGE_MAPPING = {
    "Sleep stage W": 0,   # Wake
    "Sleep stage 1": 1,   # N1
    "Sleep stage 2": 2,   # N2
    "Sleep stage 3": 3,   # N3
    "Sleep stage 4": 3,   # N3 (merge N3 and N4)
    "Sleep stage R": 4,   # REM
    "Movement time": -1,  # Exclude
    "Sleep stage ?": -1   # Exclude
}


# =============================================================================
# Preprocessing Functions
# =============================================================================

def load_psg_recording(psg_path: str) -> mne.io.Raw:
    """
    Load and preprocess a PSG recording.

    Args:
        psg_path: Path to PSG .edf file

    Returns:
        Preprocessed MNE Raw object
    """
    # Load EDF file
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)

    # Select target channels
    available_channels = [ch for ch in CHANNELS if ch in raw.ch_names]
    if len(available_channels) != len(CHANNELS):
        missing = set(CHANNELS) - set(available_channels)
        raise ValueError(f"Missing channels: {missing}")

    raw.pick(available_channels)

    # Resample to target rate if needed
    if raw.info['sfreq'] != SAMPLING_RATE:
        raw.resample(SAMPLING_RATE, verbose=False)

    # Apply bandpass filter (0.5-35 Hz)
    raw.filter(0.5, 35., verbose=False)

    return raw


def segment_signals(
    raw: mne.io.Raw,
    hypnogram_path: str
) -> List[Tuple[np.ndarray, int]]:
    """
    Segment raw signal into epochs based on hypnogram annotations.

    Args:
        raw: Preprocessed MNE Raw object
        hypnogram_path: Path to hypnogram .edf file

    Returns:
        List of tuples (epoch_data, label)
        - epoch_data: (C, T) numpy array
        - label: int (0-4)
    """
    # Read hypnogram annotations
    annot = mne.read_annotations(hypnogram_path)
    raw.set_annotations(annot, emit_warning=False)

    # Extract events from annotations
    events, event_id = mne.events_from_annotations(
        raw, event_id=None, chunk_duration=30., verbose=False
    )

    # Create inverse mapping: event_id -> description
    id_to_desc = {v: k for k, v in event_id.items()}

    # Create MNE Epochs object
    tmax = EPOCH_LENGTH - 1. / raw.info['sfreq']

    mne_epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0.,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    # Extract epochs with valid labels
    epochs_data = []
    expected_samples = int(EPOCH_LENGTH * raw.info['sfreq'])

    for i, event_code in enumerate(mne_epochs.events[:, 2]):
        desc = id_to_desc.get(event_code, "")

        if desc in STAGE_MAPPING:
            label = STAGE_MAPPING[desc]

            if label != -1:  # Valid sleep stage
                data = mne_epochs[i].get_data(copy=True)[0]

                if data.shape[1] == expected_samples:
                    epochs_data.append((data.astype(np.float32), label))

    return epochs_data


def save_epochs_with_metadata(
    epochs: List[Tuple[np.ndarray, int]],
    subject_id: str,
    output_dir: Path,
    metadata_file: Path
) -> None:
    """
    Save epochs as .npy files and update metadata CSV.

    Args:
        epochs: List of (epoch_data, label) tuples
        subject_id: Subject identifier
        output_dir: Directory for .npy files
        metadata_file: Path to metadata.csv
    """
    file_exists = metadata_file.exists()

    with open(metadata_file, "a", newline="") as f:
        fieldnames = ["filename", "subject_id", "epoch_index", "stage_label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for idx, (epoch_data, stage_label) in enumerate(epochs):
            filename = f"{subject_id}_epoch_{idx:04d}.npy"
            filepath = output_dir / filename

            np.save(filepath, epoch_data)

            writer.writerow({
                "filename": filename,
                "subject_id": subject_id,
                "epoch_index": idx,
                "stage_label": stage_label
            })


def preprocess_sleep_edf(
    data_dir: Path,
    output_dir: Path,
    max_subjects: int = None
) -> None:
    """
    Preprocess the entire Sleep-EDF dataset.

    Args:
        data_dir: Path to Sleep-EDF database directory
        output_dir: Path for preprocessed output
        max_subjects: Maximum number of subjects to process (optional)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = output_dir / "metadata.csv"

    # Find all PSG files (pattern: SC*PSG.edf or ST*PSG.edf)
    psg_files = sorted(data_dir.glob("**/*PSG.edf"))

    if max_subjects:
        psg_files = psg_files[:max_subjects]

    print(f"Found {len(psg_files)} PSG recordings")
    print(f"Output directory: {output_dir}")
    print(f"Channels: {CHANNELS}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    print(f"Epoch length: {EPOCH_LENGTH} s")

    total_epochs = 0

    for psg_path in tqdm(psg_files, desc="Processing"):
        # Find corresponding hypnogram file
        hypno_path = str(psg_path).replace("PSG.edf", "Hypnogram.edf")

        if not Path(hypno_path).exists():
            print(f"  Skipping {psg_path.name}: No hypnogram found")
            continue

        try:
            # Extract subject ID from filename
            subject_id = psg_path.stem.replace("-PSG", "")

            # Load and preprocess PSG
            raw = load_psg_recording(str(psg_path))

            # Segment into epochs
            epochs = segment_signals(raw, hypno_path)

            if len(epochs) > 0:
                # Save epochs and update metadata
                save_epochs_with_metadata(
                    epochs, subject_id, output_dir, metadata_file
                )
                total_epochs += len(epochs)

        except Exception as e:
            print(f"  Error processing {psg_path.name}: {e}")
            continue

    print(f"\nPreprocessing complete!")
    print(f"Total epochs: {total_epochs}")
    print(f"Metadata saved to: {metadata_file}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Sleep-EDF dataset for TASA model"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to Sleep-EDF database directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./processed_data"),
        help="Output directory for preprocessed data"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to process"
    )

    args = parser.parse_args()

    preprocess_sleep_edf(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_subjects=args.max_subjects
    )


if __name__ == "__main__":
    main()
