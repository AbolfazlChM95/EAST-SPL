'''
Compute FLOPs for EAST-SPL network components.

This module calculates the computational cost of model blocks and
stores the results for later use during tiling optimization and evaluation.

THOP reports multiply-accumulate operations (MACs). This script estimates FLOPs as twice the reported MAC count.

Output:
    FeatureExtractionFLOPs.npz:
        FLOPs for the shared feature-extraction block.

    Unet_in3_out16_chan_8_16_32.npz:
        FLOPs for the complete U-Net.

    completed_idx.npy:
        Grid indices completed during the current or previous runs.


Project:
    EAST-SPL
    https://github.com/AbolfazlChM95/EAST-SPL/
'''

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from thop import profile

import UNet

MIN_INPUT_SIZE = 50
MAX_INPUT_SIZE = 1200
EVALUATION_STEP = 1
SAVE_EVERY = 1000

# Load Paths
root = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flops_unet_file = root / "Unet_in3_out16_chan_8_16_32.npz"
flops_shared_block_file = root / "FeatureExtractionFLOPs.npz"
log_file = root / "completed_idx.npy" # previously calculated indices in case of cancel/error within the loop

# Load shared Block torch network
shared_block = torch.load(root /"My_Feature_Extractor.pt", weights_only=False).to(device)
shared_block.eval()

# Load UNet
u_net = UNet.UNet(in_channels= 3, out_channels=16).to(device)
u_net.eval()

# Set up tabular grid space for input size
width_list = np.arange(MIN_INPUT_SIZE, MAX_INPUT_SIZE + EVALUATION_STEP, EVALUATION_STEP)
height_list = np.arange(MIN_INPUT_SIZE, MAX_INPUT_SIZE + EVALUATION_STEP, EVALUATION_STEP)


def main() -> None:

    shape = (len(width_list), len(height_list))

    # Load previously calculated indices
    if (log_file.exists() and flops_unet_file.exists() and flops_shared_block_file.exists()):
        completed = set(map(tuple, np.load(log_file, allow_pickle=True)))
        FLOPs_UNet = np.load(flops_unet_file)['FLOPs']
        FLOPs_Shared_Block = np.load(flops_shared_block_file)['FLOPs']
    else:
        completed = set()
        FLOPs_UNet = np.zeros(shape)
        FLOPs_Shared_Block = np.zeros(shape)

    counter = 0

    # Loop through input sizes
    with torch.inference_mode():

        for idx, width in enumerate(tqdm(width_list, desc="Outer loop")):
            for idy, height in enumerate(tqdm(height_list, desc="Inner loop", leave = False)):

                # Symmetric for input shape, hence also completed for rotated rectangle with same area
                if height < width:
                    continue
                
                # Skip if previously computed
                if (idx, idy) in completed:
                    continue 
                
                input_tensor = torch.randn(1, 3, width, height).to(device)
                
                # Call the MACs calculator from thop package
                flops, _ = profile(u_net, inputs=(input_tensor,), verbose=False)
                FLOPs_UNet[idx, idy] = 2 * flops
                FLOPs_UNet[idy, idx] = 2 * flops

                flops, _ = profile(shared_block, inputs=(input_tensor,), verbose=False)
                FLOPs_Shared_Block[idx, idy] = 2 * flops
                FLOPs_Shared_Block[idy, idx] = 2 * flops

                # Add to computed indices
                completed.add((idx, idy))

                counter += 1

                # Save the FLOPs
                if counter % SAVE_EVERY == 0:
                    np.savez(flops_unet_file, FLOPs=FLOPs_UNet)  
                    np.savez(flops_shared_block_file, FLOPs=FLOPs_Shared_Block)
                    np.save(log_file, np.array(list(completed), dtype=object))

    # Final Saving of the FLOPs
    np.savez(flops_unet_file, FLOPs=FLOPs_UNet)
    np.savez(flops_shared_block_file, FLOPs=FLOPs_Shared_Block)
    np.save(log_file, np.array(list(completed), dtype=object))

if __name__ == "__main__":
    main()