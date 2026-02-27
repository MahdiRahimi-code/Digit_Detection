# train_mnist_mlp.py
# Train an MLP model on MNIST and save it to disk.

import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import BatchNormalization
from sklearn.utils.class_weight import compute_class_weight

def augment_7s(x_train, y_train, factor=3):
    sevens = x_train[y_train == 7]
    aug_list = []

    for img in sevens:
        img_2d = img.reshape(28, 28)

        for _ in range(factor):
            # small random shift
            tx = np.random.randint(-2, 3)
            ty = np.random.randint(-2, 3)

            M = np.float32([[1, 0, tx], [0, 1, ty]])
            shifted = cv2.warpAffine(img_2d, M, (28, 28), borderValue=0)

            aug_list.append(shifted.reshape(28 * 28))

    x_aug = np.array(aug_list, dtype=np.float32)
    y_aug = np.full(len(x_aug), 7, dtype=y_train.dtype)

    x_new = np.concatenate([x_train, x_aug], axis=0)
    y_new = np.concatenate([y_train, y_aug], axis=0)

    return x_new, y_new

def main():
    # -------------------------------------------------------
    # 1. Load MNIST dataset
    # -------------------------------------------------------
    # MNIST images are 28x28 grayscale images of handwritten digits (0-9)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # -------------------------------------------------------
    # 2. Preprocess data
    # -------------------------------------------------------
    # Convert from uint8 [0, 255] to float32 [0.0, 1.0]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flatten 28x28 images into 784-length vectors for MLP
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    x_train, y_train = augment_7s(x_train, y_train, factor=3)


    # y_train and y_test are integer labels [0..9], so we can use sparse_categorical_crossentropy
    num_classes = 10

    # -------------------------------------------------------
    # 3. Build the MLP model
    # -------------------------------------------------------
    model = models.Sequential([
        layers.Input(shape=(28 * 28,)),

        layers.Dense(256),
        BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.2),

        layers.Dense(128),
        BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.2),

        layers.Dense(64),
        BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation="softmax")
    ])

    # Print a summary of the model
    model.summary()

    # -------------------------------------------------------
    # 4. Compile the model
    # -------------------------------------------------------
    model.compile(
        optimizer="adam",                        # Adam optimizer
        loss="sparse_categorical_crossentropy",  # Suitable for integer labels
        metrics=["accuracy"]
    )

    # -------------------------------------------------------
    # 5. Train the model
    # -------------------------------------------------------
    # You can increase epochs for better accuracy (but more training time)
    model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=128,
        validation_split=0.2,  # Use 10% of training data for validation
        verbose=1
    )

    # -------------------------------------------------------
    # 6. Evaluate the model
    # -------------------------------------------------------
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    y_pred_probs = model.predict(x_test)

# 2. Convert probabilities to class indices (0–9)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 3. Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # 4. Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))

    plt.figure(figsize=(8, 8))
    disp.plot(include_values=True, cmap="Blues", ax=plt.gca(), xticks_rotation="vertical")
    plt.title("Confusion Matrix - MNIST MLP")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------
    # 7. Save the model to disk
    # -------------------------------------------------------
    # We save the model in HDF5 format
    model_filename = "mnist_mlp.h5"

    # Create output directory if needed (optional)
    out_dir = "saved_models"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, model_filename)

    model.save(model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
