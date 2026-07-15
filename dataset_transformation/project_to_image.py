'''
Create a projected heatmap for a given image and camera setting.

This script aggregates player-position statistics for the selected camera setting and projects them onto the image plane.

This script saves points2D, Cam_var.npz, and projHeatmap.npy

Outputs:
    Cam_var.npz         # Camera parameters for given image
    Point2D.npy         # Projected player_positions into image plane 
    ProjHeatmap.npy     # Projected heatmap as statistics on the image plane

example to run:
    python dataset_transformation/project_to_image.py \
        /path/to/SpiideoSynLoc \
        player_positions.pkl \
        sample_images/017770.jpg \
        --image-split train
        
Project:
    EAST-SPL
    https://github.com/AbolfazlChM95/EAST-SPL 
'''
from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from sskit import unnormalize, world_to_image

def parse_args():
    parser = ArgumentParser(
        description="Project player-location statistics onto a sample image."
    )

    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the extracted Spiideo SoccerNet SynLoc dataset.",
    )

    parser.add_argument(
        "positions_file",
        type=Path,
        help="Path to player_positions.pkl.",
    )

    parser.add_argument(
        "sample_image",
        type=Path,
        help="Path to the sample image used for projection.",
    )

    parser.add_argument(
        "--image-split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split containing the sample image.",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the heatmap on image and save figure"
    )
    return parser.parse_args()


def main() -> None :

    # parsing the pathes
    args = parse_args()

    annotation_dir = args.dataset_path / "annotations"
    annotation_file = annotation_dir / f"{args.image_split}.json"

    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_file}"
        )

    if not args.positions_file.exists():
        raise FileNotFoundError(
            f"Player-position file not found: {args.positions_file}"
        )

    if not args.sample_image.exists():
        raise FileNotFoundError(
            f"Sample image not found: {args.sample_image}"
        )
    
    sample_dir = args.sample_image.parent

    camera_var_path = sample_dir / "Cam_var.npz"
    points2d_path = sample_dir / "points2D.npy"
    projected_heatmap_path = sample_dir / "ProjHeatmap.npy"

    positions_df = pd.read_pickle(args.positions_file)


    # calculate normalization factor
    datasets = ['test', 'train', 'val']
    num_time_stamps = 0
    for dataset in datasets:
        split_file = annotation_dir / f"{dataset}.json"
        with open(split_file, encoding="utf-8") as file:
            Annotation_dataset = json.load(file)
            num_time_stamps += len(Annotation_dataset['images'])
            print(f"dataset {dataset}, num image: {len(Annotation_dataset['images'])}")

    # Find the image camera matrix and distortion coefficients
    image_name = args.sample_image.name
    with open(annotation_file, encoding="utf-8") as file:
        Annotation_dataset = json.load(file)
        images = Annotation_dataset['images']

    matching_images = [
    image for image in images
    if image["file_name"] == image_name
    ]

    if not matching_images:
        raise ValueError(
            f"Image {image_name} was not found in {annotation_file}."
        )

    image_data = matching_images[0]

    camera_matrix = np.array(image_data["camera_matrix"])
    dist_poly = np.array(image_data["dist_poly"])
    undist_poly = np.array(image_data["undist_poly"])

    # Save the cam params for future if needed!
    np.savez(
        camera_var_path, 
        cam_matrix=camera_matrix,
        dist=dist_poly,
        undist=undist_poly
    )

    ## normalizing the statistics (positions) to the pitch corresponds to the pitch of the inputted sample image
    camera_max_coords = positions_df.groupby('Cam_id')[['x','y', 'camera_matrix']].agg(
                max_abs_x = ('x', lambda x_ : x_.abs().max()),
                max_abs_y = ('y', lambda y_ : y_.abs().max()),
                camera_matrix = ('camera_matrix', 'first')
                ).reset_index()
    
    is_target_matrix_mask = camera_max_coords['camera_matrix'].apply(
    lambda c_matrix: np.array_equal(c_matrix, camera_matrix)
    )

    target_cam_row = camera_max_coords[is_target_matrix_mask]

    target_max_x = target_cam_row['max_abs_x'].iloc[0]
    target_max_y = target_cam_row['max_abs_y'].iloc[0]

    positions_df = positions_df.merge(camera_max_coords, on = 'Cam_id',how = 'left')

    # normalize all the coordinates to target pitch dimention
    positions_df['x_normalized'] = (positions_df['x'] / positions_df['max_abs_x']) * target_max_x
    positions_df['y_normalized'] = (positions_df['y'] / positions_df['max_abs_y']) * target_max_y

    x_normalized = positions_df['x_normalized'].to_numpy()
    y_normalized = positions_df['y_normalized'].to_numpy()
    
    z_column = np.zeros(len(positions_df))

    # convert to sskit compatible format
    normalized_coordinates_3D = np.column_stack((x_normalized, y_normalized, z_column))


    # Project points to the camera plane and save for later use!
    img = Image.open(args.sample_image).convert('RGB')

    if points2d_path.exists():
        points_image = np.load(points2d_path)
    else:
        points_image = []
        for coord in tqdm(normalized_coordinates_3D):
            p_im_tuple = unnormalize(
                world_to_image(camera_matrix, dist_poly, coord),
                [3, img.size[1], img.size[0]]
            )
            points_image.append([p_im_tuple[0], p_im_tuple[1]])

        points_image = np.array(points_image)
        np.save(points2d_path, points_image)


    ### Calculate and Save heat map with same resulotion as image
    heatmap_grid = np.zeros((img.size[1], img.size[0]), dtype=np.float64)

    for x_f, y_f in points_image:
        x_int = int(x_f)
        y_int = int(y_f)
        if 0 <= x_int < img.size[0] and 0 <= y_int < img.size[1]:
            heatmap_grid[y_int, x_int] += 1

    heatmap_grid /= num_time_stamps
    np.save(projected_heatmap_path, heatmap_grid)


    # Save projected heatmap visualization
    if args.plot:
        heatmap_width = 100
        heatmap_height = 100
        heatmap, _, _ = np.histogram2d(
            points_image[:, 1],
            points_image[:, 0],
            bins=(heatmap_height, heatmap_width),
            range=((0, img.size[1]), (0, img.size[0])),
        )

        heatmap = np.ma.masked_where(heatmap == 0, np.log1p(heatmap))

        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(img)
        ax.imshow(
            heatmap,
            extent=(0, img.size[0], img.size[1], 0),
            cmap="rainbow",
            interpolation="bilinear",
            alpha=0.5,
        )
        ax.axis("off")

        figure_path = sample_dir / "ProjectedHeatmap.jpg"
        fig.savefig(figure_path, bbox_inches="tight", dpi=200)
        plt.close(fig)

if __name__ == "__main__":
    main()