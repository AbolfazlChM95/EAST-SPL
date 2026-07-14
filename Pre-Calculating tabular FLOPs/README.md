# FLOPs Calculation

This folder contains scripts for measuring the computational cost of EAST-SPL
network components over different input dimensions.

## Main script

[`calculate_flops.py`](calculate_flops.py) profiles:

- the complete U-Net localization network;
- the shared feature-extraction block used by the rejection network.

THOP reports multiply-accumulate operations. This implementation estimates
FLOPs as twice the reported MAC count.

## Requirements

The script requires:

- PyTorch
- NumPy
- THOP
- tqdm
- the SPL-BEV (DTSPL-BEV) U-Net
- the shared feature-extraction model

## Usage

From the repository root:

```bash
python flops_calculation/calculate_flops.py
