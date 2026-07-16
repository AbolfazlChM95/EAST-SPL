# Isolated training of Rejection Head

This folder contains scripts for isolated training of rejection head using pre-trained shared block from [DTSPL-BEV](https://github.com/IvarPersson/SPL-BEV).


## Feature extraction stage

To generate a dataset from [SynLoc dataset](https://github.com/Spiideo/sskit), use [`feature_extraction.py`](feature_extraction.py).

To run the script you need corresponding mask of pitch for every camera configuration within the [SynLoc dataset](https://github.com/Spiideo/sskit). This is provided within the folder [`Masks`](./Masks/). The json file [`mask_mapping.json`](mask_mapping.json) contains the correspondence between each image in the dataset with a mask in [`Masks`](./Masks/), If missing, by running the [`feature_extraction.py`](feature_extraction.py), the json file will be created and saved again.

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
