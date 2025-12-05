# Image Classification for Table Types

## Project Overview
This project implements a deep learning solution for classifying table images into different categories. The system processes document images, extracts features, and classifies them using convolutional neural networks (CNNs). The solution achieves **98% accuracy** on test data using EfficientNet-B3 architecture.

**weights available at release section**

## Installation
### Requirements
- Python 3.11.11
- PyTorch 2.9.1
- torchvision 0.24.1
- CUDA 12.1 (for GPU support)

### Install Dependencies
pip install torch torchvision pillow pandas scikit-learn matplotlib tqdm scikit-image opencv-python

### Verified Environment
Python 3.11.11
torch==2.9.1
torchvision==0.24.1
pillow==12.0.0
pandas==2.3.3
scikit-learn==1.7.2
matplotlib==3.10.7
tqdm==4.67.1
scikit-image==0.25.2
opencv-python==4.12.0.88

## Project Structure
- dataset.py - Custom dataset class and data loaders
- model.py - Model architectures (ResNet, EfficientNet, custom CNN)
- train.py - Training pipeline with configuration management
- inference.py - Inference pipeline with prediction and evaluation utilities
- experiments/ - Output directory for training runs (contains model checkpoints, metrics, and evaluation results)

## Quick Start
### 1. Data Preparation
Organize your data in the following structure:
- data/train.csv - CSV with columns: filename, category
- data/test.csv - CSV with columns: filename, category
- data/train/ - Folder containing training images
- data/test/ - Folder containing test images

### 2. Training a Model
python train.py --train_csv ./data/train.csv --train_dir ./data/train --test_csv ./data/test.csv --test_dir ./data/test --model efficientnet_b3 --epochs 50 --batch_size 32 --image_size 512 --output_dir ./experiments/efficientnet_b3

### 3. Running Inference
**Single Image:**
python inference.py --model_path ./experiments/efficientnet_b3/best_model.pth --image /path/to/your/image.jpg --verbose

**Directory of Images:**
python inference.py --model_path ./experiments/efficientnet_b3/best_model.pth --directory /path/to/images/folder --output_csv results.csv --batch_size 16

**Batch Processing from List:**
python inference.py --model_path ./experiments/efficientnet_b3/best_model.pth --image_list image_paths.txt --output_csv results.csv

### 4. Model Evaluation
python inference.py --mode evaluate --model_path ./experiments/efficientnet_b3/best_model.pth --eval_csv ./data/test.csv --eval_dir ./data/test --batch_size 16

## Model Performance

### Model Comparison Results

| Model | Accuracy | Precision | Recall | F1-Score | Misclassifications |
|-------|----------|-----------|--------|----------|-------------------|
| **EfficientNet-B3** | **98.00%** | **0.9805** | **0.9800** | **0.9801** | **6/300 (2.00%)** |
| ResNet-50 | 96.00% | 0.96 | 0.96 | 0.96 | 12/300 (4.00%) |


### Detailed Reports:
Complete classification reports and confusion matrices are available in the experiments directory:
- `./experiments/efficientnet_b3/evaluation_results.json`
- `./experiments/resnet50/evaluation_results.json`

## Detailed Usage

### Dataset Module (`dataset.py`)
The `ImageClassificationDataset` class handles image preprocessing:
- Aspect ratio preservation with white padding
- Grayscale conversion
- Orientation correction (ensures landscape mode)
- Size statistics collection
- Data augmentation for training

### Model Module (`model.py`)
**Available Models:**
- ResNet-based Classifiers (18, 34, 50, 101)
- EfficientNet Classifiers (B0-B3)
- Document CNN (custom architecture)

**Recommended: EfficientNet-B3** for best balance of accuracy and efficiency.

### Training Module (`train.py`)
**Features:**
- Automatic experiment tracking and checkpointing
- Learning rate scheduling
- Gradient clipping for stable training
- Comprehensive metrics logging
- Visualization of training curves

### Inference Module (`inference.py`)
**Features:**
- Single image and batch prediction
- Directory scanning with recursive option
- Probability outputs for all classes
- CSV export of results
- Device selection (CPU/GPU)
- **Evaluation Mode**: Comprehensive model performance analysis

## Deliverables

### 1. Code Repository
All source code is available in this repository, organized into modular components for dataset handling, model definition, training, and inference.

### 2. Inference Pipeline
The `inference.py` script provides a complete inference pipeline that can:
- Accept single images, directories, or file lists as input
- Output predictions with confidence scores
- Export results to CSV format
- Run on CPU or GPU
- Perform comprehensive model evaluation with detailed metrics

Weights for trained models are available at the release section.

### 3. Assignment Questions

**a. How long did it take to solve the problem?**
Approximately 8 hours over two training sessions. Additional time was spent exploring classical computer vision approaches initially, but the deep learning approach proved more effective.

**b. Explain your solution**
The solution uses a deep learning pipeline with the following components:
1. **Data Preprocessing**: Images are resized to 512x512, converted to grayscale, aspect ratio is maintained with white padding, and orientation is corrected to ensure landscape mode.
2. **Model Architecture**: EfficientNet-B3 was selected for its balance of accuracy and efficiency, modified to accept single-channel input.
3. **Training Pipeline**: Includes data augmentation, learning rate scheduling, and checkpointing to prevent overfitting.
4. **Inference System**: Provides flexible prediction capabilities with probability outputs and comprehensive evaluation.

**c. Which model did you use and why?**
**EfficientNet-B3** was chosen because:
- Achieved 98% accuracy vs 96% for ResNet-50
- Converged faster (30 vs 40+ epochs)
- More computationally efficient
- Better balanced performance across all table types
- CNNs are particularly effective for detecting patterns, edges, and textures in table images

**d. Any shortcomings and how can we improve the performance?**
**Shortcomings:**
1. **Border vs Row-Border Confusion**: Both models occasionally confuse bordered and row-bordered tables
2. **Computational Requirements**: Training requires GPU resources
3. **Limited Interpretability**: Deep learning models are black boxes

**Improvements:**
1. **Classical CV Pre-filtering**: Use traditional computer vision to detect table borders before classification
2. **Advanced Augmentation**: Add document-specific augmentations (perspective distortion, noise)

### 4. Model Performance on Test Data

- **Accuracy**: 98.00%
- **Precision**: 0.9805
- **Recall**: 0.9800
- **F1-Score**: 0.9801
- **Average Confidence**: 0.9859 ± 0.0573
- **Total Samples**: 300
- **Misclassified Samples**: 6 (2.00% error rate)

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **bordered** | 0.9897 | 0.9600 | 0.9746 | 100 |
| **borderless** | 1.0000 | 0.9900 | 0.9950 | 100 |
| **row_bordered** | 0.9519 | 0.9900 | 0.9706 | 100 |

### Confusion Matrix

| True \ Predicted | bordered | borderless | row_bordered |
|------------------|----------|------------|--------------|
| **bordered** | 96 | 0 | 4 |
| **borderless** | 0 | 99 | 1 |
| **row_bordered** | 1 | 0 | 99 |

### Usage Example:
python inference.py --mode evaluate --model_path ./experiments/efficientnet_b3/best_model.pth --eval_csv ./data/test.csv --eval_dir ./data/test --batch_size 16

