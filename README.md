# VS-MSIM

Official implementation of the paper:

**Enhancing Adversarial Example Attacks via Vector Smoothing and Multi-Scale Invariance**

This repository provides the implementation of **Vector Smoothing (VS)** and the **Multi-Scale Invariant Method (MSIM)** for improving the transferability of adversarial examples.

## Requirements

The experiments are implemented with Python and PyTorch.

- Python == 3.7.11
- PyTorch == 1.8.0
- torchvision == 0.8.0
- numpy == 1.21.2
- pandas == 1.3.5
- opencv-python == 4.5.4.60
- scipy == 1.7.3
- pillow == 8.4.0
- pretrainedmodels == 0.7.4
- tqdm == 4.62.3
- imageio == 2.6.1


Quick Start
1. Prepare the Dataset
The ImageNet-compatible dataset used in our experiments can be downloaded from:
[Dataset](https://github.com/Zhijin-Ge/STM/tree/main/dataset)
After downloading, place the dataset under:
./dataset/
The dataset directory should contain the input images and the corresponding label file required by the data loader.

2. Prepare the Models
Normally Trained Models
The normally trained models used in our experiments include:
- Inc-v3
- Inc-v4
- IncRes-v2
- Res-101
These models are loaded using the pretrainedmodels package.
When a model is used for the first time, the corresponding pretrained weights will be downloaded automatically. Please wait until the downloading process is completed.

Adversarially Trained Models
The adversarially trained models used in the experiments include:
- ens3_adv_inc_v3
- ens4_adv_inc_v3
- ens_adv_inc_res_v2
The pretrained weights and corresponding model implementations can be obtained from:
- SSA
- tf_to_torch_model
For detailed instructions on downloading and loading these models, please refer to the corresponding repositories.
