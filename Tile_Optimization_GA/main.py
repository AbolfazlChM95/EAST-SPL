'''

Run EAST-SPL tile optimization using a genetic algorithm.

The script loads a sample field image, its corresponding mask, and the
projected player-location heatmap. It then constructs statistical and
non-statistical FLOPs objectives and optimizes tile configurations using
the genetic algorithm defined in TileGenOpt.py.

The script saves optimization results in an `optimization_results` folder
next to the sample image. Optional plots include GA convergence and the
optimized tile configurations.

Example:
    python3 main.py sample_field/023626.jpg

Save convergence and tiling plots:
    python3 main.py sample_field/023626.jpg --plots

Project:
    EAST-SPL
    https://github.com/AbolfazlChM95/EAST-SPL
'''

from argparse import ArgumentParser

import StatsTilings
import TileGenOpt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
import time



def parse_args():
    parser = ArgumentParser(
        description="Run genetic-algorithm tile optimization for one sample field."
    )

    parser.add_argument(
        "sample_image",
        type=Path,
        help="Path to the sample field image.",
    )

    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save the GA convergence and optimized tiling figures.",
    )

    return parser.parse_args()


def main()-> None:
    args = parse_args()
    
    # Load image and root path
    sample_image = args.sample_image
    sample_dir = sample_image.parent
    image_stem = sample_image.stem
    
    # Load mask and projected heatmap
    mask_path = sample_dir / f"{image_stem}-mask.jpg"
    projected_heatmap_path = sample_dir / "ProjHeatmap.npy"

    if not sample_image.exists():
        raise FileNotFoundError(f"Sample image not found: {sample_image}")
    
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask image not found: {mask_path}")
    
    if not projected_heatmap_path.exists():
        raise FileNotFoundError(f"Projected heamap not found: {projected_heatmap_path}")

    img = np.array(plt.imread(sample_image))

    mask = plt.imread(mask_path)
    mask = np.int8(np.array(mask[:,:,0]) == 255)
    
    projected_stats = np.load(projected_heatmap_path)

    # Load FLOPs tabular arrays
    root = Path(__file__).resolve().parent
    network_flops_dir = root / "Network_FLOPs"
    
    unet_flops_file = network_flops_dir / "Unet_in3_out16_chan_8_16_32.npz"
    shared_flops_file = network_flops_dir / "FeatureExtractionFLOPs.npz" 

    if not unet_flops_file.exists():
        raise FileNotFoundError(f"U-Net FLOPs table not found: {unet_flops_file}")

    if not shared_flops_file.exists():
        raise FileNotFoundError(
            f"Shared-block FLOPs table not found: {shared_flops_file}"
        )

    unet_flops = np.load(unet_flops_file)["FLOPs"]
    shared_flops = np.load(shared_flops_file)["FLOPs"]

    unet_flops_shift = 10
    shared_flops_shift = 50


    # Set output path for saving results
    output_dir = sample_dir / "optimization_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "opt_results.npz"
    convergence_plot_path = output_dir / "GA_convergence.png"


    # Optimization hyperparameters
    margin = 50
    max_length = 1000
    min_length = 50

    pop_size = 50
    max_generations = 500
    convergence_exit = 5

    # Instanciate the GA algorithm
    algorithm = TileGenOpt.EvolutionaryTiling(
        mask,
        max_len=max_length,
        min_len=min_length,
        pop_size=pop_size
    )

    # Instanciate the objective function
    loss = StatsTilings.FLOPsCalculators(
        mask, 
        StatsTilings.RejNetFLOPs,
        unet_flops,
        unet_flops_shift,
        shared_flops,
        shared_flops_shift,
        projected_stats,
        margin
    )

    if results_path.exists():
        # Load the result if the optimization has been saved
        data = np.load(results_path, allow_pickle=True)
        Statistical_Points = data['min_Statistical_Points']
        Stat_best_scores = data['min_Statistical_best_scores']
        t_opt_0 = data['Statistical_runtime']
        NonStatNoMerge_Points = data['min_NonStatNoMerge_Points']
        nonstat_no_merge_best_scores = data['nonstat_no_merge_best_scores']
        t_opt_1 = data['nonstat_no_merge_runtime']
        NonStatMerge_Points = data['min_NonStatMerge_Points']
        nonstat_merge_best_scores = data['nonstat_merge_best_scores']
        t_opt_2 = data['nonstat_merge_runtime']
    else:

        t_start = time.time()
        Statistical_Points, Stat_Score, Stat_best_scores = algorithm.evolve(objective_func=loss.StatFLOPsPoints, max_num_gens= max_generations, convergence_exit=convergence_exit)
        t_opt_0 = time.time()-t_start 


        t_start = time.time()
        obj = lambda p : loss.NonStatFLOPsPoints( p , merge=False)
        NonStatNoMerge_Points, NonStatNoMerge_Score, nonstat_no_merge_best_scores = algorithm.evolve(objective_func=obj, max_num_gens= max_generations, convergence_exit=convergence_exit)
        t_opt_1 = time.time()-t_start 


        t_start = time.time()
        obj = lambda p : loss.NonStatFLOPsPoints( p , merge=True, max_len = max_length)
        NonStatMerge_Points, NonStatMerge_Score, nonstat_merge_best_scores = algorithm.evolve(objective_func=obj, max_num_gens= max_generations, convergence_exit=convergence_exit)
        t_opt_2 = time.time()-t_start 

        np.savez(
            results_path,
        
            # Wrap ragged lists with dtype=object
            min_Statistical_Points = np.array(Statistical_Points, dtype=object),
            min_Statistical_best_scores = Stat_best_scores,
            Statistical_runtime = t_opt_0,
            
            min_NonStatNoMerge_Points = np.array(NonStatNoMerge_Points, dtype=object),
            nonstat_no_merge_best_scores = nonstat_no_merge_best_scores,
            nonstat_no_merge_runtime = t_opt_1,
            
            min_NonStatMerge_Points = np.array(NonStatMerge_Points, dtype=object),
            nonstat_merge_best_scores = nonstat_merge_best_scores,
            nonstat_merge_runtime = t_opt_2
        )

    results = [
        {
            "objective": "ETF",
            "etf": loss.StatFLOPsPoints(Statistical_Points) * 1e-9,
            "tf": loss.NonStatFLOPsPoints(
                Statistical_Points,
                merge=False,
            ) * 1e-9,
            "tf_merge": loss.NonStatFLOPsPoints(
                Statistical_Points,
                merge=True,
                max_len=max_length,
            ) * 1e-9,
            "runtime": t_opt_0,
        },
        {
            "objective": "TF",
            "etf": loss.StatFLOPsPoints(NonStatNoMerge_Points) * 1e-9,
            "tf": loss.NonStatFLOPsPoints(
                NonStatNoMerge_Points,
                merge=False,
            ) * 1e-9,
            "tf_merge": loss.NonStatFLOPsPoints(
                NonStatNoMerge_Points,
                merge=True,
                max_len=max_length,
            ) * 1e-9,
            "runtime": t_opt_1,
        },
        {
            "objective": "TF + M",
            "etf": loss.StatFLOPsPoints(NonStatMerge_Points) * 1e-9,
            "tf": loss.NonStatFLOPsPoints(
                NonStatMerge_Points,
                merge=False,
            ) * 1e-9,
            "tf_merge": loss.NonStatFLOPsPoints(
                NonStatMerge_Points,
                merge=True,
                max_len=max_length,
            ) * 1e-9,
            "runtime": t_opt_2,
        },
    ]

    print()
    print(
        f"{'Objective':<12}"
        f"{'ETF (GFLOPs)':>15}"
        f"{'TF (GFLOPs)':>15}"
        f"{'TF + M (GFLOPs)':>19}"
        f"{'Runtime (s)':>14}"
    )

    print("-" * 75)

    for result in results:
        print(
            f"{result['objective']:<12}"
            f"{result['etf']:>15.2f}"
            f"{result['tf']:>15.2f}"
            f"{result['tf_merge']:>19.2f}"
            f"{result['runtime']:>14.1f}"
        )

    if args.plots:
        ## genetic algorithm cost plot
        plt.figure(figsize=(6, 7))
        stat_best_scores = np.asarray(Stat_best_scores, dtype=float)
        min_flops = np.min(stat_best_scores) * 1e-9
        plt.plot(stat_best_scores *1e-9, linestyle='-', color='#008080', linewidth=5)
        
        current_ticks = plt.yticks()[0]
        plt.yticks(np.sort(np.append(current_ticks, min_flops)))
        
        plt.xlabel('Generations', fontdict={'size':14})
        plt.ylabel('Expected Total FLOPs (GFLOPs)', fontdict={'size':14})
        plt.grid(True)
        y_lims = plt.ylim()
        # plt.ylim((0.0 , y_lims[1]))
        # plt.show()
        plt.savefig(
            convergence_plot_path,
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()


        ## Plot tiles on image
        Tiles_statsFLOPs = StatsTilings.P2Tiles(Statistical_Points, mask = mask)
        StatsTilings.PlotTilesOnImage(
            Tiles_statsFLOPs,
            img,
            title = f'Tile Configuration with minimum Statistical Total FLOPs, {loss.StatFLOPsPoints(Statistical_Points)*1e-9:0.2f} GFLOPs',
            save_path=output_dir / "statistical_tiling.png"
            )
        StatsTilings.PlotTilesProbOnImage(
            Tiles_statsFLOPs,
            img,
            projected_stats,
            title = f'Tile Configuration with minimum Statistical Total FLOPs, {loss.StatFLOPsPoints(Statistical_Points)*1e-9:0.2f} GFLOPs',
            save_path=output_dir / "statistical_tiling_prob.png"
            )

        Tiles_statsFLOPs = StatsTilings.P2Tiles(NonStatMerge_Points, mask = mask, merge=True, merge_max_length=max_length)
        StatsTilings.PlotTilesOnImage(
            Tiles_statsFLOPs,
            img,
            title= f"Tile configuration with minimum Total FLOPs, {loss.NonStatFLOPsPoints(NonStatMerge_Points, merge=True, max_len = max_length)*1e-9:0.2f} GFLOPs",
            save_path= output_dir / "non_statistical_tiling.png"
            )

if __name__=="__main__":
    main()