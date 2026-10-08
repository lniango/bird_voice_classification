# Bird Sound Classification

A small deep-learning project for classifying bird species from annotated soundscape recordings using PyTorch on an HPC cluster with Slurm and NVIDIA GPUs.

## Overview

The project uses the HSN dataset, *A collection of fully-annotated soundscape recordings from the southern Sierra Nevada mountain range*.

Dataset: https://doi.org/10.5281/zenodo.7525805

The main pipeline is:

```text
Annotated Audio
      ↓
Audio Segment
      ↓
Fixed 1-second Waveform
      ↓
Mel-Spectrogram
      ↓
CNN
      ↓
7 Bird Species
```

## Dataset

The HSN dataset contains:

- 100 soundscape recordings
- 10 minutes per recording
- 32 kHz mono audio used in this project
- 10,296 known bird annotations
- 21 bird species in the original dataset
- Unknown annotations (`????`) excluded

### Project classes

| Label | eBird Code |
|------:|------------|
| 0 | `amepip` |
| 1 | `clanut` |
| 2 | `daejun` |
| 3 | `gcrfin` |
| 4 | `herthr` |
| 5 | `rocwre` |
| 6 | `whcspa` |

### Dataset structure

```text
bird_classifier/
├── data/
│   ├── audio/
│   │   └── *.flac
│   ├── annotations.csv
│   ├── species.csv
│   ├── splits.csv
│   └── labels.csv
├── src/
│   └── bird_classifier/
├── scripts/
├── checkpoints/
├── logs/
└── README.md
```

`annotations.csv` contains:

```text
Filename
Start Time (s)
End Time (s)
Low Freq (Hz)
High Freq (Hz)
Species eBird Code
```

The data is split by **recording**, not by annotation, to avoid data leakage:

```text
80 recordings → train
10 recordings → validation
10 recordings → test
```

## Audio Processing

Each annotation is used to extract only the corresponding part of the recording.

Audio is converted to a fixed duration of **1 second**:

```text
32,000 Hz × 1 second = 32,000 samples
```

The dataset therefore returns:

```text
Waveform: [1, 32000]
```

Short segments are zero-padded and longer segments are cropped.

## Mel-Spectrogram

The 1-second waveform will be converted into a Mel-spectrogram representing the distribution of acoustic energy across frequency and time.

```text
Waveform [1 × 32000]
        ↓
Mel-Spectrogram
        ↓
CNN
```

Mel-spectrogram parameters will be selected and tested during development.

## CNN

The Mel-spectrogram will be used as the input of a Convolutional Neural Network.

```text
Mel-Spectrogram
      ↓
Convolution
      ↓
Activation
      ↓
Pooling
      ↓
Convolution
      ↓
Pooling
      ↓
Fully Connected Layer
      ↓
7-Class Output
```

The goal is to learn characteristic frequency and temporal patterns from bird vocalizations.

## Training Workflow

```text
annotations.csv
      ↓
BirdSoundDataset
      ↓
1-second waveform
      ↓
Mel-Spectrogram
      ↓
DataLoader
      ↓
CNN
      ↓
Prediction
      ↓
Loss
      ↓
Backpropagation
      ↓
Model update
```

The validation set is used during training, while the test set is reserved for final evaluation.

## HPC Environment

The project runs on an HPC cluster using Slurm.

Main environment:

- Python 3.12
- PyTorch
- torchaudio
- pandas
- NVIDIA A100 GPU

Project directory:

```text
/SCRATCH/l26niang/bird_classifier
```

Python environment:

```text
/SCRATCH/l26niang/bird-env
```

Activate the environment with:

```bash
source /SCRATCH/l26niang/bird-env/bin/activate
```

## Current Status

- [x] Download and extract dataset
- [x] Inspect audio and annotations
- [x] Select 7 target species
- [x] Create recording-level train/validation/test split
- [x] Create label mapping
- [x] Implement `BirdSoundDataset`
- [x] Extract annotated audio segments
- [x] Convert segments to fixed 1-second waveforms
- [ ] Implement Mel-spectrogram
- [ ] Implement CNN
- [ ] Implement DataLoader and training loop
- [ ] Train on GPU
- [ ] Evaluate on test set
- [ ] Analyze classification errors

## Goal

Build and understand a complete bird-sound classification pipeline:

**Audio → Mel-Spectrogram → CNN → 7-Class Classification**
