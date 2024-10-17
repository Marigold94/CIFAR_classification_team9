# Team9 CIFAR-100 DenseNet Model

This repository contains the implementation of a DenseNet model with various enhancements for training and evaluating on the CIFAR-100 dataset. It includes the following features:
- Model modifications with additional dropout layers.
- Enhanced data augmentation techniques (Cutout, AutoAugment, etc.).
- Cosine annealing with warm restarts for learning rate scheduling.
- Early stopping to prevent overfitting.
- MixUp augmentation (disabled by default but can be re-enabled).
- Random seed setup for reproducibility.

## Prerequisites

Ensure that you have the following libraries installed before running the notebook:

pip install torch torchvision matplotlib numpy

- `torch` >= 1.7.0
- `torchvision` >= 0.8.0
- `matplotlib` >= 3.3.0
- `numpy` >= 1.19.0

## Dataset

The CIFAR-100 dataset is used for this project. The dataset will be automatically downloaded when running the code, or you can download it manually from [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Running the Code

The main code is in the `team9.ipynb` notebook. To train and evaluate the model, follow these steps:

1. **Set up your environment**:
    - Clone the repository.
    - Open the notebook `team9.ipynb` and ensure that the required libraries are installed (see "Prerequisites").

2. **Train the model**:
    The training process can be initiated by running the cells in the notebook. The model uses the CIFAR-100 dataset with predefined data augmentations, optimizer, and learning rate schedule.

3. **Evaluate the model**:
    Evaluation metrics such as Top-1, Top-5 accuracy, and Superclass accuracy will be displayed after training. The results can be visualized using the provided plotting functions in the notebook.

## Random Seed

To ensure the reproducibility of the experiment, random seeds are set in PyTorch, NumPy, and Python's random module. The seed value is set to `42` by default but can be changed as needed. 

### How to Set the Random Seed

The `set_seed(seed)` function ensures consistent results. To change the seed, modify the `seed` value as follows:

set_seed(42)  # Default seed value is 42, change it if necessary

The random seed is set across the following libraries:
- Python's built-in `random` module
- `numpy`
- PyTorch (both CPU and GPU)

## Model Details

The model is based on a DenseNet architecture with the following modifications:
- **DenseNet Blocks**: Utilizes growth rate of 32, with 3 dense blocks.
- **Dropout**: A dropout layer is added to each dense block to prevent overfitting.
- **Augmentation**: Various augmentation techniques such as Cutout and AutoAugment are applied.

### Learning Rate Scheduler

- The model uses a **CosineAnnealingWarmRestarts** scheduler, combined with **ReduceLROnPlateau** for dynamic learning rate adjustment based on validation accuracy.

### Early Stopping

- Early stopping is applied with a patience of 10 epochs and a minimum delta of 0.01 to avoid overfitting.

## Evaluation Metrics

The model reports the following metrics:
- **Top-1 Accuracy**: The percentage of correctly classified images among the top-1 predictions.
- **Top-5 Accuracy**: The percentage of correctly classified images among the top-5 predictions.
- **Superclass Accuracy**: Accuracy averaged across 20 superclasses in CIFAR-100.

## License

This project is licensed under the MIT License.
