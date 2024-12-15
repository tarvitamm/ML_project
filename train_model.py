import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import argparse
import os

# Define U-Net Model
def unet_model(input_size):
    """
    U-Net model generator.

    Args:
        input_size (tuple): The input shape of the model, e.g., (512, 512, 1).

    Returns:
        Model: A compiled U-Net model.
    """
    inputs = Input(input_size)

    # Encoder (Downsampling path)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder (Upsampling path)
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# Main function
def main(image_size):
    """
    Main function to train the U-Net model based on the given image size.

    Args:
        image_size (int): The size of the images (e.g., 512 or 256).
    """
    # Load data based on the image size
    train_file = f"X_train_{image_size}.npy"
    label_file = f"y_train_{image_size}.npy"
    val_file = f"X_val_{image_size}.npy"
    val_label_file = f"y_val_{image_size}.npy"

    print(f"Loading training data: {train_file}, {label_file}")
    X_train = np.load(train_file)
    y_train = np.load(label_file)

    print(f"Loading validation data: {val_file}, {val_label_file}")
    X_val = np.load(val_file)
    y_val = np.load(val_label_file)

    print("Data loaded successfully!")
    print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

    # Initialize and compile the model
    input_size = (image_size, image_size, 1)
    model = unet_model(input_size)

    learning_rate = 0.00005
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    model.summary()

    # Define early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    # Train the model
    print("Training the model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=20,
        epochs=150,
        callbacks=[early_stopping]
    )

    # Save the model
    model_save_path = f"models/unet_{image_size}.h5"
    os.makedirs("models", exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a U-Net model with specified image size.")
    parser.add_argument(
        "image_size", type=int, choices=[256, 512],
        help="The size of the input images (256 or 512)."
    )
    args = parser.parse_args()

    # Run the main function
    main(args.image_size)
