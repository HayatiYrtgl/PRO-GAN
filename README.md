Here’s a Progressive Growing GAN (ProGAN) implementation:

```markdown
# Progressive Growing GAN (ProGAN) with WGAN-GP

This repository contains the PyTorch implementation of a **Progressive Growing GAN** (ProGAN) with **Wasserstein Loss** and **Gradient Penalty** (WGAN-GP). ProGAN progressively increases the resolution of images during training, which improves the stability and quality of the generated images. The model starts with generating small images (e.g., 4x4 pixels) and gradually increases the resolution up to larger sizes (e.g., 256x256 pixels).

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Training](#training)
- [Customization](#customization)
- [Results](#results)
- [References](#references)

## Overview

This project implements a Progressive Growing GAN that uses:
- Wasserstein Loss with Gradient Penalty (WGAN-GP)
- Weight-scaled convolutions for better training stability
- Pixel normalization for regularization
- Progressive layers to grow image size incrementally during training

The generator and critic networks are built progressively, starting from a 4x4 image resolution and doubling the size with each step. The model stabilizes learning by fading in new layers instead of adding them abruptly.

## Features
- Progressive image size growth during training.
- Wasserstein GAN with Gradient Penalty (WGAN-GP) for stable training.
- Weight-scaled convolutions and pixel normalization layers for better training performance.
- Supports multi-resolution training from 4x4 to 256x256 (or higher).
- Training checkpoints and generation of sample images.

## Requirements

To run the code, you need the following dependencies:

- Python 3.x
- PyTorch
- torchvision
- tqdm
- matplotlib
- numpy
- PIL (Pillow)

You can install the required packages by running:

```bash
pip install torch torchvision tqdm matplotlib numpy pillow
```

## Usage

1. **Dataset Preparation**: 

   Organize your dataset in the following structure, where `gans_dataset` is the folder containing your images, with subfolders for each class (or just one folder for all images if it's unconditional GAN training):
   ```
   DATASETS/
   └── gans_dataset/
       └── class1/
           ├── img1.jpg
           ├── img2.jpg
           └── ...
   ```
   Make sure to update the `DATASET` path in the code to point to your dataset.

2. **Run Training**:
   To start training, simply run the script:
   ```bash
   python main.py
   ```

## Training

The training is split into several stages, where each stage corresponds to generating images of a different resolution (4x4, 8x8, 16x16, etc.). Each stage is trained for a certain number of epochs (defined by `PROGRESSIVE_EPOCHS`).

### Hyperparameters:
- `LEARNING_RATE`: Learning rate for the optimizer (default: `1e-3`).
- `BATCH_SIZES`: Batch sizes for different image resolutions (higher resolutions use smaller batches due to memory constraints).
- `LAMBDA_GP`: Coefficient for the gradient penalty in WGAN-GP (default: `10`).
- `Z_DIM`: Size of the latent vector (default: `256`).
- `IN_CHANNELS`: Number of input channels for the convolution layers (default: `256`).

### Checkpoints:
Generated images at each step will be saved in the `saved_examples` folder. You can visualize the training progress by inspecting the images.

## Customization

### Modify Image Resolution:
You can change the final image resolution by adjusting the `factors` array, which defines the scale factors for each layer. For example, the current setup goes from 4x4 to 256x256. You can add more steps to go up to 512x512 or even 1024x1024 by modifying `factors`, `BATCH_SIZES`, and `PROGRESSIVE_EPOCHS`.

### Change Latent Vector Size:
Modify the `Z_DIM` parameter to change the dimensionality of the input noise vector.

### Dataset:
The script expects an image dataset in the folder specified by the `DATASET` variable. The dataset should be in a structure compatible with `torchvision.datasets.ImageFolder`.

### Weight Scaling and Pixel Normalization:
The implementation includes custom `WSConv2d` layers and `PixelNorm` layers to improve training stability. These can be modified or replaced if needed.

## Results

Generated images from each training step are saved in the `saved_examples` folder. The generated examples should gradually improve in quality as the training progresses and the image resolution increases.

### Example Results:
Images will be added after training.

## References

- **Progressive Growing of GANs**: Tero Karras, et al. (https://arxiv.org/abs/1710.10196)
- **Wasserstein GAN with Gradient Penalty (WGAN-GP)**: Ishaan Gulrajani, et al. (https://arxiv.org/abs/1704.00028)
```

This `README.md` provides clear guidance on how to set up and run the project, as well as explanations for the key components and customization options.
