import tensorflow as tf
from keras.saving.save import load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score
import argparse

# Define helper functions
def image_and_label_to_numpy_array(image_directory, label_directory):
    """
    Convert images and corresponding labels from directories to numpy arrays.
    """
    images = []
    labels = []
    for filename in sorted(os.listdir(label_directory)):
        try:
            img = Image.open(os.path.join(image_directory, filename))
            lbl = Image.open(os.path.join(label_directory, filename))
            img = np.asarray(img)
            lbl = np.asarray(lbl)
            img, lbl = format_data(img, lbl)
            images.append(img)
            labels.append(lbl)
        except Exception as e:
            print(f"Cannot process file {filename}: {e}")
            continue
    return np.array(images, dtype=object), np.array(labels, dtype=object)


def format_data(images, labels):
    """
    Normalize image and label data and convert labels to binary format.
    """
    images_expanded = np.expand_dims(images, axis=-1)
    labels_expanded = np.expand_dims(labels, axis=-1)
    images_normalized = images_expanded / 255
    labels_normalized = labels_expanded / 255
    labels_binary = (labels_normalized > 0).astype(np.float32)
    return images_normalized, labels_binary


def predict_on_tiles(model, val_images, tile_size):
    """
    Predict on tiles for the validation images.

    Args:
        model: Trained model used for predictions.
        val_images: Validation images.
        tile_size: Tile size (e.g., 512 or 256).

    Returns:
        A list of predictions for the validation images.
    """
    predictions = []

    for img_idx, img in enumerate(val_images):
        h, w, c = img.shape
        tiles = []
        coords = []

        # Create tiles
        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                tile = img[i:i + tile_size, j:j + tile_size]
                tile_h, tile_w = tile.shape[:2]

                if tile_h < tile_size or tile_w < tile_size:
                    # Pad tiles at edges
                    pad_h = tile_size - tile_h
                    pad_w = tile_size - tile_w
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

                tiles.append(tile)
                coords.append((i, j))

        tiles = np.array(tiles, dtype=np.float32)  # Ensure tiles are float32
        # Predict tiles
        tile_predictions = model.predict(tiles)

        # Reconstruct the image from tiles
        reconstructed = np.zeros((h, w), dtype=np.float32)
        for (i, j), tile_pred in zip(coords, tile_predictions):
            tile_pred = tile_pred[..., 0]  # Extract single channel prediction
            tile_h, tile_w = min(tile_pred.shape[0], h - i), min(tile_pred.shape[1], w - j)
            reconstructed[i:i + tile_h, j:j + tile_w] = tile_pred[:tile_h, :tile_w]

        predictions.append(reconstructed)

    return predictions


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run inference on a trained UNet model with specified tile size.")
    parser.add_argument("--tile_size", type=int, required=True, help="Tile size to use for inference (e.g., 256, 512).")
    args = parser.parse_args()

    # Check for the model file corresponding to the tile size
    model_path = f'models/unet_{args.tile_size}.h5'
    if not os.path.exists(model_path):
        print(f"Model for tile size {args.tile_size} not found at {model_path}. Please provide a valid tile size.")
        exit(1)

    print(f"Using tile size: {args.tile_size}")
    print(f"Loading model from {model_path}")

    # Directories for test images and labels
    test_images_dir = 'images/X_test'
    test_labels_dir = 'labels/y_test'

    # Convert test images and labels to numpy arrays
    X_test, y_test = image_and_label_to_numpy_array(test_images_dir, test_labels_dir)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    # Load trained model
    loaded_model = load_model(model_path)

    # Predict on test images
    predictions = predict_on_tiles(loaded_model, X_test, args.tile_size)

    # Evaluate predictions at different thresholds
    accuracies, precisions, recalls, thresholds = [], [], [], []
    y_test_binary = [(label > 0).astype(np.int8) for label in y_test]

    for i in range(1, 20):
        threshold = i / 20
        thresholds.append(threshold)
        total_accuracy, total_precision, total_recall = 0, 0, 0
        sample_count = len(y_test_binary)

        for idx in range(sample_count):
            predicted_label = (predictions[idx] > threshold).astype(np.int8)
            y_test_flat = y_test_binary[idx].flatten()
            predicted_label_flat = predicted_label.flatten()

            total_accuracy += accuracy_score(y_test_flat, predicted_label_flat)
            total_precision += precision_score(y_test_flat, predicted_label_flat)
            total_recall += recall_score(y_test_flat, predicted_label_flat)

        accuracies.append(total_accuracy / sample_count)
        precisions.append(total_precision / sample_count)
        recalls.append(total_recall / sample_count)

    # Plot metrics
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, accuracies, label="Accuracy")
    plt.plot(thresholds, recalls, label="Recall")
    plt.legend()
    plt.xlabel('Threshold')
    plt.ylabel('Metric')
    plt.title('Metrics vs Threshold')
    plt.show()

    # Visualize predictions
    fig, axes = plt.subplots(4, 3, figsize=(8, 12))
    threshold = 0.36

    for pic_id in range(4):  # Assuming 4 images in test set
        pred = (predictions[pic_id].squeeze() > threshold).astype(np.int8)
        label = y_test[pic_id].squeeze().astype(np.int8)
        img = X_test[pic_id].squeeze()

        axes[pic_id, 0].imshow(pred, cmap='gray')
        axes[pic_id, 0].set_title(f'Prediction (Pic {pic_id})')
        axes[pic_id, 0].axis('off')

        axes[pic_id, 1].imshow(label, cmap='gray')
        axes[pic_id, 1].set_title(f'True Label (Pic {pic_id})')
        axes[pic_id, 1].axis('off')

        axes[pic_id, 2].imshow(img, cmap='gray')
        axes[pic_id, 2].set_title(f'Original Image (Pic {pic_id})')
        axes[pic_id, 2].axis('off')

        # Print metrics for the specific image
        accuracy = accuracy_score(label.flatten(), pred.flatten())
        precision = precision_score(label.flatten(), pred.flatten())
        recall = recall_score(label.flatten(), pred.flatten())
        print(f'Pic {pic_id}: Threshold: {threshold}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

    plt.tight_layout()
    plt.savefig('predictions')
    plt.show()
