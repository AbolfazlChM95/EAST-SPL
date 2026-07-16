# Tile Optimization with Genetic Algorithm

This folder contains the tile-configuration optimization stage of EAST-SPL.

The optimization uses a genetic algorithm to search for efficient horizontal and vertical tile boundaries for a given sample field. Candidate configurations are evaluated using statistical and non-statistical FLOPs objectives.

## Files

### [`main.py`](main.py)

Main entry point for running the optimization.

The script:

- loads a sample field image;
- loads its corresponding field mask;
- loads the projected player-location heatmap;
- loads the precomputed network FLOPs tables;
- runs three genetic-algorithm optimizations;
- saves the optimized configurations and convergence history;
- optionally saves convergence and tiling visualizations.

### [`TileGenOpt.py`](TileGenOpt.py)

Contains the `EvolutionaryTiling` class and the genetic-algorithm operations:

- population initialization;
- crossover;
- coordinate shifting;
- coordinate insertion and removal;
- elitism;
- random immigrants;
- convergence-based stopping.

### [`StatsTilings.py`](StatsTilings.py)

Contains utilities for:

- constructing tiles from collocation points;
- filtering tiles using the field mask;
- calculating tile probabilities;
- calculating statistical and non-statistical FLOPs;
- merging adjacent tiles;
- visualizing tile configurations.

## Requirements

The scripts require Python 3 and the following packages:

```text
numpy
matplotlib
```

The standard-library modules `argparse`, `pathlib`, `random`, `copy`, and `time` are also used.

## Required Input Structure

The sample image directory must contain:

```text
sample_field/
├── 023626.jpg
├── 023626-mask.jpg
└── ProjHeatmap.npy
```

The image and mask must follow this naming convention:

```text
<image-name>.jpg
<image-name>-mask.jpg
```

The precomputed FLOPs tables must be placed in the [`Network_FLOPs`](./Network_FLOPs/) directory:

```text
Tile_Optimization_GA/
├── main.py
├── StatsTilings.py
├── TileGenOpt.py
├── Network_FLOPs/
│   ├── Unet_in3_out16_chan_8_16_32.npz
│   └── FeatureExtractionFLOPs.npz
└── sample_field/
    ├── 023626.jpg
    ├── 023626-mask.jpg
    └── ProjHeatmap.npy
```

## Usage

Run the optimization from this folder:

```bash
python3 main.py sample_field/023626.jpg
```

To also save the convergence and tile-configuration plots:

```bash
python3 main.py sample_field/023626.jpg --plots
```

The mask and projected heatmap paths are inferred automatically from the sample image path.

Running the optimizer will print the results in the terminal:
```text
Objective      ETF (GFLOPs)    TF (GFLOPs)    TF + M (GFLOPs)   Runtime (s)
---------------------------------------------------------------------------
ETF                   36.32         290.08             200.42          91.5
TF                    91.59         176.82             172.93           9.9
TF + M                71.84         201.79             157.99          26.6
```
Each row corresponds to the objective used during optimization, while the three FLOPs columns evaluate the resulting tile configuration under all objectives.

## Optimization Objectives

The script runs the genetic algorithm separately for three objectives:

| Objective | Description |
|---|---|
| `ETF` | Expected Total FLOPs using projected player-location probabilities and the rejection network |
| `TF` | Total FLOPs without tile merging |
| `TF + M` | Total FLOPs after applying the tile-merging procedure |

After optimization, each final tile configuration is evaluated under all three objectives and printed as a table in GFLOPs.

## Hyperparameters

The main optimization parameters are currently defined in [`main.py`](main.py):

```python
margin = 50
max_length = 1000
min_length = 50

pop_size = 50
max_generations = 500
convergence_exit = 5
```

| Parameter | Default | Description |
|---|---:|---|
| `margin` | `50` | Additional tile margin required by the network receptive field |
| `max_length` | `1000` | Maximum allowed distance between neighboring collocation points |
| `min_length` | `50` | Minimum allowed distance between neighboring collocation points |
| `pop_size` | `50` | Number of candidate solutions in each generation |
| `max_generations` | `500` | Maximum number of genetic-algorithm generations |
| `convergence_exit` | `5` | Stop after this number of generations without improvement |

The genetic-algorithm mutation parameters are defined in [`TileGenOpt.py`](TileGenOpt.py).

## Outputs

The script creates an `optimization_results` directory beside the sample image:

```text
sample_field/
├── 023626.jpg
├── 023626-mask.jpg
├── ProjHeatmap.npy
└── optimization_results/
    ├── opt_results.npz
    ├── GA_convergence.png
    ├── statistical_tiling.png
    ├── statistical_tiling_prob.png
    └── non_statistical_tiling.png
```

### `opt_results.npz`

Contains:

- optimized collocation points for each objective;
- best-score history for each genetic-algorithm run.

> [!NOTE] If this file already exists, the saved results are loaded instead of rerunning the optimization!

### Plot outputs

Plot files are generated only when the `--plots` flag is provided:

- `GA_convergence.png`: ETF convergence over generations;
- `statistical_tiling.png`: tile configuration optimized for ETF;
- `statistical_tiling_prob.png`: ETF tile configuration with tile probabilities;
- `non_statistical_tiling.png`: tile configuration optimized for `TF + M`.

## Notes

The projected heatmap can be generated using the scripts in the [`dataset-transformation`](../dataset_transformation/) stage of the repository.

The FLOPs lookup tables can be generated using the scripts in[`Pre-Calculating_tabular_FLOPs`](../Pre-Calculating_tabular_FLOPs/).


## [DTSPL-BEV Grid Search](https://doi.org/10.5220/0014468500004067)

[`DTSPLBEV-tiling_methods.py`](/DTSPLBEV_grid_search/DTSPLBEV_tiling_methods.py) contains tiling method from [DTSP-BEV: Decomposable Tiled Soccer Player Localization↗](https://doi.org/10.5220/0014468500004067). Simple example of running for given `initial_tile_length`:

```python
Tiles, _, _ = Tiling_Methods.Tiling(
    'DynFlopsA', # method of division
    img.shape, 
    initial_tile_size = tile_length, #initial tile length
    mask = mask,
    Tiles2FLOPs = Tiles2FLOPs, # calculate the FLOPs from tabular array
    margin = margin, # marging to add for receptive field
    minimum_length = min_length,
    outlier_pixels_threshold = 1,
    Full_process_verbos = False
)
```

## Future work
> [!WARNING]
> The random number generators used in [`TileGenOpt.py`](TileGenOpt.py) are not initialized with a fixed seed. To reproduce exactly the same numerical results, fixed seeds should be added for both Python's `random` module and `NumPy`.