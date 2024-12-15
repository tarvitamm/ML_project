import numpy as np
from PIL import Image
import os
import argparse
from sklearn.model_selection import train_test_split

def transform_to_shape(size):
    """
    Slice images into tiles of the specified size, save them as images,
    and save the tile arrays as train and validation .npy files.

    Args:
        size (int): Desired tile size (e.g., 512 or 256).
    """
    # Define the final shape of the tiles
    final_shape = (size, size)

    # Define input directories
    input_X_directory = 'images/X_train'
    input_y_directory = 'labels/y_train'
    output_X_directory = f'images/X_train{size}'
    output_y_directory = f'labels/y_train{size}'

    # Create output directories if they don't exist
    os.makedirs(output_X_directory, exist_ok=True)
    os.makedirs(output_y_directory, exist_ok=True)

    # Accumulate tiles into lists for saving as numpy arrays
    X_tiles = []
    y_tiles = []

    # Function to process tiles from a directory
    def process_tiles(input_directory, output_directory):
        tiles = []
        for file in sorted(os.listdir(input_directory)):
            input_path = os.path.join(input_directory, file)
            try:
                img = Image.open(input_path)
            except Exception as e:
                print(f"Exception for file {file}: {e}")
                continue

            img = np.asarray(img)
            original_shape = img.shape

            # Compute steps and overlays
            split_x = max(1, original_shape[0] // final_shape[0])
            split_y = max(1, original_shape[1] // final_shape[1])
            residue_x = original_shape[0] % final_shape[0]
            residue_y = original_shape[1] % final_shape[1]
            overlay_x = (final_shape[0] - residue_x) // split_x if split_x > 0 else 0
            overlay_y = (final_shape[1] - residue_y) // split_y if split_y > 0 else 0
            step_x = final_shape[0] - overlay_x
            step_y = final_shape[1] - overlay_y

            # Determine start positions
            start_positions_x = sorted(set(range(0, original_shape[0] - final_shape[0] + 1, step_x)))
            start_positions_y = sorted(set(range(0, original_shape[1] - final_shape[1] + 1, step_y)))

            # Save tiles
            for x_idx, start_x in enumerate(start_positions_x):
                end_x = start_x + final_shape[0]
                for y_idx, start_y in enumerate(start_positions_y):
                    end_y = start_y + final_shape[1]
                    tile = img[start_x:end_x, start_y:end_y]

                    # Save tile as image
                    output_filename = f"{file[:3]}_split_{x_idx}_{y_idx}.png"
                    output_path = os.path.join(output_directory, output_filename)
                    im = Image.fromarray(tile)
                    im.save(output_path)

                    # Save tile into array
                    tiles.append(tile)

        return tiles

    # Process and save tiles for X_train
    print(f"Processing X_train tiles into {output_X_directory}")
    X_tiles = process_tiles(input_X_directory, output_X_directory)
    X_tiles = np.array(X_tiles)
    print(f"Total X tiles: {X_tiles.shape}")

    # Process and save tiles for y_train
    print(f"Processing y_train tiles into {output_y_directory}")
    y_tiles = process_tiles(input_y_directory, output_y_directory)
    y_tiles = np.array(y_tiles)
    print(f"Total y tiles: {y_tiles.shape}")

    # Perform train-validation split
    print("Performing train-validation split (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_tiles, y_tiles, test_size=0.2, random_state=42
    )

    # Save train and validation datasets
    np.save(f"X_train_{size}.npy", X_train)
    np.save(f"y_train_{size}.npy", y_train)
    np.save(f"X_val_{size}.npy", X_val)
    np.save(f"y_val_{size}.npy", y_val)

    print(f"Saved X_train_{size}.npy with shape {X_train.shape}")
    print(f"Saved y_train_{size}.npy with shape {y_train.shape}")
    print(f"Saved X_val_{size}.npy with shape {X_val.shape}")
    print(f"Saved y_val_{size}.npy with shape {y_val.shape}")

    print("Image tiling and dataset saving complete.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Slice images into tiles of specified size and save .npy files.")
    parser.add_argument("tile_size", type=int, help="Tile size (e.g., 512 for 512x512 tiles)")
    args = parser.parse_args()

    # Run the function with the provided tile size
    transform_to_shape(args.tile_size)
