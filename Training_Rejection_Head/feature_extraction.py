"""
Generate pooled tile features for isolated rejection-head training.

The script loads first double-conv block of the pretrained EAST-SPL feature extractor, generates labeled
tiles from the Spiideo SoccerNet SynLoc dataset, applies average and max pooling
to each tile feature map, and saves the resulting features as compressed .npz
files for the train, validation, and test splits.

to run:
python3 feature_extraction.py "/home/username/Datasets/SpiideoSynLoc_fullhd"

Project:
    EAST-SPL
    https://github.com/AbolfazlChM95/EAST-SPL
"""

import cv2
import torch
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

root = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_tiles(num_tiles, mask, players_bboxes, min_tile_size, max_tile_size ):
    """
    Generate random labeled tiles constrained by the pitch mask.

    Tiles are labeled positive when they overlap at least one player bounding
    box. When players are present, sampling maintains a minimum positive-label
    rate.
    """


    player_flag = len(players_bboxes) > 0
    true_labels_min_rate = 0.3 if player_flag else 0.0

    ones_x, ones_y = np.where(mask == 1)
    
    x_lims = [ones_x.min(), ones_x.max()]
    y_lims = [ones_y.min(), ones_y.max()]

    max_w_size = min(max_tile_size, x_lims[1]-x_lims[0])
    max_h_size = min(max_tile_size, y_lims[1]-y_lims[0]) 
   
    if player_flag:
        bboxes_x     = players_bboxes[:, 1]
        bboxes_y     = players_bboxes[:, 0]
        bboxes_x_end = players_bboxes[:,1] + players_bboxes[:,3]
        bboxes_y_end = players_bboxes[:,0] + players_bboxes[:,2]

    Tiles, labels = [], []
    while len(Tiles)< num_tiles:
        w = np.random.randint(min_tile_size, max_w_size)
        h = np.random.randint(min_tile_size, max_h_size)
        x = np.random.randint(x_lims[0], x_lims[1] - w)
        y = np.random.randint(y_lims[0], y_lims[1] - h)

        tile_mask = mask[x : x+w, y : y+h]
        in_pixels = np.sum(tile_mask == 1)

        # selecting tiles with better aspect ratio
        aspect_ratio_prob = min( w/h, h/w)
        if np.random.random() > aspect_ratio_prob:
            continue

        # selecting tiles based on being inside field or having player withing them
        if in_pixels > 0.2* (w*h):
            
            append_flag = True
            player_in_tile = False

            if player_flag:
                # Check if there's a player inside
                
                # Bounding boxes
                y_condition = (bboxes_x < x+w) & (bboxes_x_end > x)
                x_condition = (bboxes_y < y+h) & (bboxes_y_end > y)
                player_in_tile = np.any(y_condition & x_condition)

                positive_rate = np.mean(labels) if labels else 0.0

                if not player_in_tile and positive_rate < true_labels_min_rate:
                    append_flag = False
            
            if append_flag:
                Tiles.append([x, y, w, h])
                labels.append(player_in_tile)

    Tiles, labels = np.array(Tiles), np.array(labels)

    return Tiles, labels

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the extracted Spiideo SoccerNet SynLoc dataset.",
    )

    args = parser.parse_args()
    data_dir = args.dataset_path
    annotation_dir = data_dir / "annotations"
    mask_mapping_path = root / "mask_mapping.json"

    if not data_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {data_dir}")

    if not annotation_dir.is_dir():
        raise NotADirectoryError(
            f"Annotation directory not found: {annotation_dir}"
        )
    

    num_tiles = 500
    min_tile_size = 100
    max_tile_size = 700
    dataset_splits = ("train", "test", "val")

    ### Finding corresponding mask for each image
    if mask_mapping_path.exists():
        print('Mask Map found')
        with mask_mapping_path.open("r", encoding="utf-8") as file:
            corresponding_mask_file = json.load(file)
            
    else:

        masked_camera_matrices = {}
        corresponding_mask_file = {}

        for dataset in dataset_splits:
            annotation_path = data_dir / "annotations" / f"{dataset}.json"
            with annotation_path.open("r", encoding="utf-8") as file:
                annotations = json.load(file)

            mask_dir = root / "Masks" / dataset
            mask_images = [mask_image.name for mask_image in mask_dir.iterdir()]
            
            annotations_images = annotations.get('images')
            print(f"{dataset} length: {len(annotations_images)}")

            for mask_image in mask_images:
                mask_image_ann = [ann for ann in annotations_images if ann['file_name'] == mask_image][0]
                masked_camera_matrices[mask_image] = np.array(mask_image_ann['camera_matrix'])

            for image in annotations_images:
                file_name = image['file_name']
                camera_matrix = np.array(image['camera_matrix'])

                for mask_image in mask_images:
                    camera_matrix_ref = masked_camera_matrices[mask_image]

                    if np.array_equal(camera_matrix, camera_matrix_ref):
                        corresponding_mask_file[f"{dataset}/{file_name}"] = f"{dataset}/{mask_image}"
                        break

        with mask_mapping_path.open("w", encoding="utf-8") as file:
            json.dump(corresponding_mask_file, file, indent=2)
    

    ### Load the pretrained feature extractor and instanciate the nets
    feature_extractor_path = root / "My_Feature_Extractor.pt"
    if not feature_extractor_path.is_file():
        raise FileNotFoundError(
            f"Feature extractor weights not found: {feature_extractor_path}"
        )
    feature_extractor = torch.load(
        feature_extractor_path,
        weights_only=False
    ).to(device)

    feature_extractor.eval()

    avg_pooling = torch.nn.AdaptiveAvgPool2d((1,1))
    max_pooling = torch.nn.AdaptiveMaxPool2d((1,1))


    with torch.no_grad():

        for dataset in dataset_splits:

            dataset_dir = data_dir / dataset
            output_dir = root / "Features" / dataset

            output_dir.mkdir(parents=True, exist_ok=True)

            annotation_path = data_dir / "annotations" / f"{dataset}.json"

            with annotation_path.open("r", encoding="utf-8") as annotation_file:
                annotations = json.load(annotation_file)
                annotations_images = annotations['images']
                annotations_players = annotations["annotations"]


            # Look at the annotations once!
            filename_to_id = {img['file_name']: img['id'] for img in annotations_images}
            image_id_to_annotations = {}
            for ann in annotations_players:
                img_id = ann['image_id']
                if img_id not in image_id_to_annotations:
                    image_id_to_annotations[img_id] = []
                image_id_to_annotations[img_id].append(ann)


            for image_path in tqdm(list(dataset_dir.iterdir())):

                image_name = image_path.name
                
                output_path = output_dir / f"{image_path.stem}.npz"
                if output_path.exists():
                    continue

                # Load the mask
                image_tag = f"{dataset}/{image_name}"
                
                if image_tag not in corresponding_mask_file:
                    raise KeyError(
                        f"No corresponding mask found for image: {image_tag}"
                    )

                corresponding_mask_name = corresponding_mask_file[image_tag]
                mask_path = root / "Masks" / corresponding_mask_name

                if not mask_path.is_file():
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")
                mask = plt.imread(mask_path)
                mask = np.int8(np.array(mask[:,:,0]) == 255)

                # Load image annotations
                image_id = filename_to_id[image_name]

                player_anns = image_id_to_annotations.get(image_id, [])
                selected_bboxes = np.array([ann['bbox'] for ann in player_anns], dtype=int)
                
                # Generate tiles
                Tiles, labels = generate_tiles(num_tiles, mask, selected_bboxes, min_tile_size, max_tile_size)

                # Loading the image
                image = cv2.imread(image_path).transpose(2, 0, 1)
                input_tensor = torch.FloatTensor(image / 255).unsqueeze(0).to(device)                    
                
                # Forward pass from Neural block
                output_tensor = feature_extractor(input_tensor)
                

                # Run the net on tiles
                feature_list = []
                label_list = [ 1 if label else 0 for label in labels]
                for tile in Tiles:
                    x, y, w, h = tile
                    
                    if image.shape[1] == 1080:
                        x, y, w, h = int(x/2), int(y/2), int(w/2), int(h/2) # full-hd version                    

                    # get tile from one time forwarded image
                    tile_output_tensor = output_tensor[:,:, x:x+w, y:y+h]

                    tile_avg_pooling = avg_pooling(tile_output_tensor).squeeze()
                    tile_max_pooling = max_pooling(tile_output_tensor).squeeze()
                                        
                    combined_features = torch.cat((tile_avg_pooling, tile_max_pooling), dim=0)

                    feature_list.append(combined_features.cpu().numpy())


                features, labels_array = np.array(feature_list), np.array(label_list) # features.shape: N_tile x 16, labels_array.shape: N_tiles

                np.savez_compressed(
                    output_path, 
                    feature = features, 
                    labels = labels_array
                )


if __name__ == '__main__':
    main()