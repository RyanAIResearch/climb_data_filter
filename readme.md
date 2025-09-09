# ClimbLab Filter

A machine learning pipeline for data filtering and selection using teacher-student model scoring and curriculum learning.

## Overview

This project implements a three-stage data filtering and selection pipeline:
1. **Streaming Processing**: Coarse filtering, bucketing, and structural feature extraction
2. **Teacher-Student Scoring**: Small-sample scoring using ΔNLL (Negative Log-Likelihood difference)
3. **Curriculum Selection**: Three-phase difficulty-stratified mixed sampling

## System Requirements

- **OS**: Linux (Ubuntu 18.04+ recommended)
- **Python**: 3.8-3.11
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: Minimum 16GB RAM (32GB+ recommended)
- **Storage**: At least 50GB available space

## Installation

### 1. Create Conda Environment

```bash
# Create a new conda environment
conda create -n climblab python=3.9 -y

# Activate the environment
conda activate climblab
```

### 2. Install PyTorch

Install PyTorch with CUDA support (adjust CUDA version as needed):

```bash
# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# For CPU-only (not recommended)
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 3. Install Core Dependencies

```bash
# Install transformers and related libraries
pip install transformers>=4.35.0
pip install accelerate>=0.24.0
pip install datasets>=2.14.0

# Install 4-bit quantization support (recommended for GPU efficiency)
pip install bitsandbytes>=0.41.0
```

### 4. Install Project Dependencies

```bash
# Navigate to project directory
cd climblab_filter

# Install project-specific requirements
pip install -r requirements.txt
```

The `requirements.txt` contains:
```
tqdm>=4.65
tiktoken>=0.7
scikit-learn>=1.3
joblib>=1.3
```

## Usage

### Quick Start

1. **Activate environment**:
   ```bash
   conda activate climblab
   ```

2. **Run the complete filtering pipeline**:
   ```bash
   bash run_filter.sh
   ```