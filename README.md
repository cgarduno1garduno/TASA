# TASA: Transition-Aware Sparse Attention for Parameter-Efficient Sleep Stage Classification

Official PyTorch implementation of **TASA** (Transition-Aware Sparse Attention), a parameter-efficient deep learning model for automatic sleep stage classification from polysomnography (PSG) recordings.

## Abstract

Automatic sleep stage classification is crucial for diagnosing sleep disorders, yet existing deep learning approaches often require substantial computational resources or struggle with the inherent class imbalance in sleep data—particularly the underrepresented N1 stage. We present TASA, a lightweight multi-task architecture that combines sparse (local windowed) attention with explicit transition detection to achieve state-of-the-art performance with significantly fewer parameters. Our approach introduces: (1) a sparse attention mechanism that reduces computational complexity from O(n²) to O(n × window_size), (2) a multi-task learning framework that jointly optimizes sleep staging and transition detection, and (3) homoscedastic uncertainty weighting with priority bias for balanced task learning. On the Sleep-EDF Expanded dataset, TASA achieves **80.79% accuracy** and **0.7311 Cohen's Kappa** while maintaining strong N1 recall (**64.82%**)—addressing a critical limitation of prior work.

## Key Features

- **Sparse Attention**: Local windowed attention reduces memory and computational requirements
- **Multi-Task Learning**: Joint sleep staging (5-class) and transition detection (binary)
- **Uncertainty Weighting**: Automatic task balancing via homoscedastic uncertainty
- **Alpha Priority Bias**: Configurable weighting (default α=5.0) for transition task emphasis
- **Parameter Efficient**: ~214K parameters vs. millions in comparable models
- **Strong N1 Performance**: 64.82% N1 recall addresses class imbalance challenges

## Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | 80.79% |
| Cohen's Kappa | 0.7427 |
| N1 Recall | 64.82% |
| Parameters | ~214K |

### Per-Class Performance

| Stage | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Wake  | 0.92 | 0.88 | 0.90 |
| N1    | 0.41 | 0.65 | 0.50 |
| N2    | 0.85 | 0.88 | 0.86 |
| N3    | 0.88 | 0.84 | 0.86 |
| REM   | 0.84 | 0.82 | 0.83 |

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/tasa-sleep-staging.git
cd tasa-sleep-staging

# Create conda environment (recommended)
conda create -n tasa python=3.11
conda activate tasa

# Install dependencies
pip install -r requirements.txt

# For Apple Silicon (MPS acceleration)
pip install tensorflow-macos tensorflow-metal
```

## Dataset

This implementation uses the [Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) from PhysioNet.

### Download and Preprocess

1. Download the Sleep-EDF dataset from PhysioNet
2. Run preprocessing:

```bash
python data/preprocess.py \
    --data-dir /path/to/sleep-edf \
    --output-dir ./processed_data
```

### Input Channels

- **EEG Fpz-Cz**: Frontal-central EEG
- **EEG Pz-Oz**: Parietal-occipital EEG
- **EOG horizontal**: Electrooculogram

### Preprocessing Pipeline

1. Load PSG recordings (EDF format)
2. Select target channels (3 channels)
3. Resample to 100 Hz
4. Bandpass filter (0.5-35 Hz)
5. Segment into 30-second epochs
6. Per-channel Z-score normalization with [-20, 20] clamping

## Usage

### Training

```bash
python scripts/train.py \
    --data-dir ./processed_data \
    --output-dir ./results \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-3 \
    --alpha 5.0
```

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | (required) | Path to preprocessed data |
| `--output-dir` | `./results` | Output directory |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--alpha` | 5.0 | Transition loss priority weight |
| `--d-model` | 64 | Model dimension |
| `--n-layers` | 4 | Number of transformer layers |
| `--n-heads` | 4 | Number of attention heads |
| `--window-size` | 64 | Local attention window size |
| `--seed` | 42 | Random seed |

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint ./results/best_model.pt \
    --data-dir ./processed_data \
    --output-dir ./evaluation
```

## Model Architecture

```
Input: (B, 3, 3000)  # 3 channels × 30s × 100Hz
    │
    ▼
┌─────────────────────────────┐
│   Convolutional Embedding   │
│   Conv1D → BN → ReLU (×2)   │
└─────────────────────────────┘
    │
    ▼ (B, 64, 750)
┌─────────────────────────────┐
│   Sparse Transformer (×4)   │
│   ┌─────────────────────┐   │
│   │ Local Window Attn   │   │
│   │ (window_size=64)    │   │
│   └─────────────────────┘   │
│   ┌─────────────────────┐   │
│   │ Feed-Forward Net    │   │
│   │ (d_model × 4)       │   │
│   └─────────────────────┘   │
└─────────────────────────────┘
    │
    ├──────────────────────────┐
    ▼                          ▼
┌─────────────┐        ┌─────────────┐
│ Staging     │        │ Transition  │
│ Head        │        │ Head        │
│ (5-class)   │        │ (binary)    │
└─────────────┘        └─────────────┘
```

## Multi-Task Loss

The total loss combines sleep staging and transition detection using homoscedastic uncertainty weighting:

```
L_total = (1/2σ₁²) × L_stage + (1/2σ₂²) × α × L_transition + log(σ₁) + log(σ₂)
```

Where:
- `L_stage`: Cross-entropy loss for 5-class sleep staging
- `L_transition`: Binary cross-entropy for transition detection
- `σ₁, σ₂`: Learned uncertainty parameters
- `α`: Priority weight for transition task (default: 5.0)

## Project Structure

```
tasa-sleep-staging/
├── models/
│   ├── __init__.py
│   └── tasa.py              # TASA architecture
├── data/
│   ├── __init__.py
│   ├── preprocess.py        # Sleep-EDF preprocessing
│   └── dataset.py           # PyTorch Dataset
├── scripts/
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation script
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tasa2025,
  title={TASA: Transition-Aware Sparse Attention for Parameter-Efficient Sleep Stage Classification},
  author={[Authors]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Sleep-EDF Database: Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberye JJL. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE-BME 47(9):1185-1194 (2000).
- PhysioNet: Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 (2000).
