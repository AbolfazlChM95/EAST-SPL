# Isolated training of Rejection Head

This folder contains scripts for isolated training of rejection head using pre-trained shared block from [DTSPL-BEV](https://github.com/IvarPersson/SPL-BEV).


## Feature extraction stage

To generate a dataset from [SynLoc dataset](https://github.com/Spiideo/sskit), use [`feature_extraction.py`](feature_extraction.py).

To run the script you need corresponding mask of pitch for every camera configuration within the [SynLoc dataset](https://github.com/Spiideo/sskit). This is provided within the folder [`Masks`](./Masks/). The JSON file [`mask_mapping.json`](mask_mapping.json) maps each dataset image to its corresponding pitch mask. If the file is missing, it is generated automatically when [`feature_extraction.py`](feature_extraction.py) is run.

To Run:
```bash
python3 feature_extraction.py "path/to/SpiideoSynLoc"
```

The dataset folder should contain:

```text
SpiideoSynLoc_fullhd/
├── train/
├── test/
├── val/
└── annotations/
    ├── train.json
    ├── test.json
    └── val.json
```

The generated files will be saved under:

```text
Training_Rejection_Head/
└── Features/
    ├── train/
    ├── test/
    └── val/
```

## Download a generated dataset
You can skip the [`feature_extraction.py`](feature_extraction.py) and download one sampled generated feature dataset from [here](https://drive.google.com/file/d/1kepOvSs2l3rjyPTu4eFC-nfTR1TNg6Tv/view?usp=sharing).

## Training of MLP on generated feature dataset

The rejection head is trained using [`rejnet_training.py`](rejnet_training.py).

The script loads the extracted tile features from the `Features` directory and evaluates multiple fully connected rejection-head architectures. Each configuration is trained across multiple runs to reduce the effect of random initialization.

The script reports:

- training and test accuracy;
- sensitivity across multiple rejection thresholds;
- rejection rate across multiple thresholds;
- computational cost in FLOPs;
- mean and standard deviation across repeated runs.

The script expects the following structure:

```text
Training_Rejection_Head/
├── rejnet_training.py
└── Features/
    ├── train/
    ├── test/
    └── val/
```

The training process generates:
```text
Training_Rejection_Head/
├── search_summary.csv
└── search_curves.npz
```