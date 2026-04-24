# The Self-Pruning Neural Network

A standard feed-forward neural network for image classification (CIFAR-10) augmented with a built-in mechanism to learn which of its own weights are unnecessary during training. Instead of a post-training pruning step, the network dynamically adapts its architecture on the fly.

## Project Overview

This project implements a clever application of $L_1$ regularization inside a custom training loop to encourage structural sparsity.

*   **Custom `PrunableLinear` layer**: Replaces standard `torch.nn.Linear`. Associates each weight in the network with a learnable scalar "gate" parameter between 0 and 1 (using a Sigmoid activation). 
*   **Dynamic Pruning Mechanism**: If a gate's output value gets pushed close to 0, it effectively "prunes" or removes the corresponding weight from the active network computationally.
*   **L1 Sparsity Regularization**: The loss function is formulated to aggressively penalize the network for having too many active gates: `Total Loss = ClassificationLoss + λ * SparsityLoss`. This drives underutilized weights toward explicit sparsity.

## Repository Structure
* `train_pruning.py`: The single end-to-end execution script containing the PyTorch model definitions, dataset loading (CIFAR-10), custom regularization loop, and plotting capabilities.
* `report.md`: Contains our detailed analysis of exactly *why* L1 penalty performs effectively on bounded sigmoid gates alongside tested metrics.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rahul-1812/Self-Pruning-Neural-Network.git
   ```

2. **Install requirements:**
   Ensure you have a valid PyTorch environment installed:
   ```bash
   pip install torch torchvision matplotlib numpy
   ```

3. **Train the network:**
   Executing the script runs a lightweight, ~10 epoch test loop across varying threshold values of the `lambda` ($\lambda$) regularization parameter.
   ```bash
   python train_pruning.py
   ```

*Upon execution, the script automatically generates a distribution plot tracking the physical deactivation of parameterized gates across the entire active network density array.*
