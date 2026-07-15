"""
Extract player locations from the Spiideo SoccerNet SynLoc dataset.

The script aggregates annotated 3D player positions across selected dataset
splits and associates each position with a unique camera configuration.

The Spiideo SoccerNet SynLoc dataset is not distributed with this repository
and must be downloaded separately.

Outputs:
    player_positions.pkl:
        Table containing dataset split, image ID, player position, and camera ID.

Project:
    EAST-SPL
    https://github.com/AbolfazlChM95/EAST-SPL
"""

from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser(
        description="Extract player locations from the Spiideo SoccerNet SynLoc dataset."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the extracted Spiideo SoccerNet SynLoc dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("."),
        help="Directory where transformed data will be saved. Default: current directory.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test", "train", "val"],
        choices=["test", "train", "val"],
        help="Dataset splits to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    annotation_dir = args.dataset_path / "annotations"
    output_dir = args.output_dir

    
    if not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"Annotation directory not found: {annotation_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    camera_matrices = []

    # loop over dataset splits (train, validation, test)
    for split in args.splits:
        annotation_file = annotation_dir / f"{split}.json"

        if not annotation_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_file}"
            )

        # load the annotation's json file
        with annotation_file.open("r", encoding="utf-8") as file:
            annotation_dataset = json.load(file)

        # load images and annotations from json
        images = annotation_dataset["images"]
        annotations = annotation_dataset["annotations"]

        print(f"Split: {split}, images: {len(images)}")

        annotations_by_image = {}

        # tag annotations based on images once
        for annotation in annotations:
            image_id = annotation["image_id"]
            annotations_by_image.setdefault(image_id, []).append(annotation)

        # loop through the images to extract locations with saving the cam config
        for image in tqdm(images, desc=split):
            camera_matrix = np.asarray(image["camera_matrix"])

            for id, cam in enumerate(camera_matrices):
                if np.array_equal(camera_matrix, cam):
                    cam_id = id
                    break
            else:
                camera_matrices.append(camera_matrix)
                cam_id = len(camera_matrices) - 1
                print(f"Cam Ids expanded to: {len(camera_matrices)}")

            for annotation in annotations_by_image.get(image["id"], []):
                x, y = np.array(annotation["keypoints_3d"])[0][:2]
                
                rows.append(
                    {
                        "dataset": split,
                        "image_id": image['id'],
                        "x": x,
                        "y": y,
                        "Cam_id": cam_id,
                        "camera_matrix": camera_matrix
                    }
                )
    positions_df = pd.DataFrame(rows)
    positions_file = output_dir / "player_positions.pkl"

    positions_df.to_pickle(positions_file)
    print(f"Saved {len(positions_df)} positions to {positions_file}")

if __name__ == "__main__":
    main()