# Robust Traffic Forecasting  against Spatial Shift over Years

This repository contains demo code for training and evaluating the GWNET model on the PEMS08 dataset, with a focus on testing its performance on shifted data.

## Overview

The code demonstrates how to:
1. Train GWNET on the PEMS08-16 dataset
2. Test the trained model on the shifted dataset PEMS08-17

## Requirements

- Python 3.x
- CUDA-enabled GPU (for faster training)
- Required Python packages (list them here, e.g., PyTorch, NumPy, etc.)

## Usage

### Training

To train GWNET on the PEMS08-16 dataset, run the following command:

```bash
python experiments/gwnet/main.py --device cuda:0 --dataset PEMS08 --years 2016 --model_name gwnet
```

### Evaluation on Shifted Data

To evaluate the trained model's performance on the shifted dataset (PEMS08-17), use this command:

```bash
python experiments/gwnet/main.py --device cuda:0 --dataset PEMS08 --years 2016 --model_name gwnet --mode test --target 2017
```

## Parameters

- `--device`: Specify the CUDA device (e.g., `cuda:0`)
- `--dataset`: Dataset name (PEMS08 in this case)
- `--years`: Training data year
- `--model_name`: Model to use (gwnet)
- `--mode`: Mode of operation (train by default, set to 'test' for evaluation)
- `--target`: Target year for shifted data evaluation

## Acknowledgement

This code is developed based on LargeST, an easy-to-use and powerful open-source ST-GNNs training framework.

## Citation

If you use ST-Expert in your research, please cite our paper:

```bibtex
@article{wang2024robust,
  title={Robust Traffic Forecasting  against Spatial Shift over Years},
  author={Wang, Hongjun and Chen, Jiyuan and Pan, Tong and Dong, Zheng and Zhang, Lingyu and Jiang, Renhe and Song, Xuan},
  journal={https://arxiv.org/abs/2410.00373},
  year={2024}
}
```

