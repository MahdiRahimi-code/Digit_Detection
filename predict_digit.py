# predict_digit.py
# Load the saved MLP model and use it to predict one or multiple digits from an input image.
# Now includes a simple GUI to select an image and show the predicted multi-digit number.

import sys
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt

# GUI imports
import tkinter as tk
from tkinter import filedialog, messagebox

# Path to the saved model (should match the path from the training script)
MODEL_DIR = "saved_models"
MODEL_NAME = "mnist_mlp.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)


def load_trained_model(model_path: str):
    """
    Load the pre-trained Keras model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from: {model_path}")
    return model


def extract_digits(image_path):
    """
    Detect multiple handwritten digits in an image.
    1) Find contours
    2) For each contour: compute bounding box + center
    3) Merge boxes whose centers are very close (parts of one digit)
    Returns:
        digit_images  -> list of cropped digit images (numpy arrays, thresholded)
        boxes         -> list of (x, y, w, h) after merge
        centers       -> list of (cx, cy) (one center per digit)
    """

    # Load grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Input image not found or cannot be read: {image_path}")

    # Threshold + invert (digits white, background black)
    _, thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Small noise removal (optional)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # --- Step 1: initial boxes and centers for each contour ---
    raw_boxes = []
    raw_centers = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter out very small noise
        if w * h < 50:
            continue

        cx = x + w // 2
        cy = y + h // 2

        raw_boxes.append((x, y, w, h))
        raw_centers.append((cx, cy))

    if not raw_boxes:
        print("No digits found in the image.")
        return [], [], []

    # --- Step 2: merge centers that are close to each other ---
    h_img, w_img = thresh.shape
    merge_dist = min(h_img, w_img) * 0.15  # e.g., 15% of the smaller side

    used = [False] * len(raw_boxes)
    merged_boxes = []
    merged_centers = []

    for i in range(len(raw_boxes)):
        if used[i]:
            continue

        # Start a new cluster (digit)
        cluster_indices = [i]
        used[i] = True
        cx_i, cy_i = raw_centers[i]

        # Add other centers close to this one to that cluster
        for j in range(i + 1, len(raw_boxes)):
            if used[j]:
                continue
            cx_j, cy_j = raw_centers[j]
            dist = np.hypot(cx_i - cx_j, cy_i - cy_j)
            if dist < merge_dist:
                used[j] = True
                cluster_indices.append(j)

        # Build a union box from all boxes in this cluster
        xs = []
        ys = []
        x2s = []
        y2s = []
        for idx in cluster_indices:
            x, y, w, h = raw_boxes[idx]
            xs.append(x)
            ys.append(y)
            x2s.append(x + w)
            y2s.append(y + h)

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(x2s)
        y_max = max(y2s)
        w_merge = x_max - x_min
        h_merge = y_max - y_min

        merged_boxes.append((x_min, y_min, w_merge, h_merge))
        merged_centers.append((x_min + w_merge // 2, y_min + h_merge // 2))

    # --- Step 3: crop digits from thresholded image & sort left-to-right ---
    digit_images = []
    for (x, y, w, h) in merged_boxes:
        digit = thresh[y:y + h, x:x + w]
        digit_images.append(digit)

    # Sort by x coordinate (left to right)
    sorted_indices = np.argsort([b[0] for b in merged_boxes])
    digit_images = [digit_images[i] for i in sorted_indices]
    boxes = [merged_boxes[i] for i in sorted_indices]
    centers = [merged_centers[i] for i in sorted_indices]

    return digit_images, boxes, centers


def preprocess_digit_for_model(digit_img: np.ndarray) -> np.ndarray:
    """
    Take a cropped digit image (binary, digit=white(255), background=black(0))
    and convert it to MNIST-like 28x28 centered with margin.
    """

    digit = digit_img.copy().astype("uint8")

    # Find inner bounding box of white pixels
    coords = cv2.findNonZero(digit)
    if coords is None:
        canvas = np.zeros((28, 28), dtype="float32")
        return canvas.reshape(1, 28 * 28)

    x, y, w, h = cv2.boundingRect(coords)
    digit_cropped = digit[y:y + h, x:x + w]

    # Resize so that max side = 20 (keep aspect ratio)
    aspect_ratio = w / h
    if w > h:
        new_w = 20
        new_h = int(round(20 / aspect_ratio))
    else:
        new_h = 20
        new_w = int(round(20 * aspect_ratio))

    digit_resized = cv2.resize(
        digit_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA
    )

    # Create 28x28 black canvas and center the digit
    canvas = np.zeros((28, 28), dtype="uint8")
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit_resized

    # Normalize to [0,1] and flatten
    canvas = canvas.astype("float32") / 255.0
    canvas_flat = canvas.reshape(1, 28 * 28)

    return canvas_flat


def predict_digit(model, processed_img: np.ndarray) -> int:
    """
    Predict the digit for a preprocessed image using the given model.
    """
    preds = model.predict(processed_img, verbose=0)
    predicted_class = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds, axis=1)[0])

    print(f"Predicted digit: {predicted_class}  (confidence: {confidence * 100:.2f}%)")
    return predicted_class


def visualize_boxes(image_path, boxes, centers):
    """
    Draw bounding boxes and single centers on the original image for visualization.
    (Used mainly for debugging in CLI mode.)
    """
    img = cv2.imread(image_path)
    for (x, y, w, h), (cx, cy) in zip(boxes, centers):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

    cv2.imshow("Digits with boxes and centers", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def recognize_number_from_image(image_path: str, model, debug: bool = False) -> str:
    """
    Full pipeline:
    - detect digits
    - preprocess each one
    - predict using the MLP model
    - return multi-digit number as string
    """
    digit_images, boxes, centers = extract_digits(image_path)
    if not digit_images:
        raise RuntimeError("No digits detected in the image.")

    if debug:
        visualize_boxes(image_path, boxes, centers)

    full_number = ""

    for idx, (digit_img, center) in enumerate(zip(digit_images, centers)):
        print(f"\n--- Digit #{idx + 1} ---")
        print(f"Bounding box: {boxes[idx]}, Center: {center}")

        processed_img = preprocess_digit_for_model(digit_img)

        if debug:
            # Show processed single digit
            plt.imshow(processed_img.reshape(28, 28), cmap="gray")
            plt.title(f"Processed digit #{idx + 1}")
            plt.axis("off")
            plt.show()

        pred_digit = predict_digit(model, processed_img)
        full_number += str(pred_digit)

    print(f"\nFinal detected number (left to right): {full_number}")
    return full_number


# ---------- CLI MODE ----------

def cli_main():
    """
    Command-line entry:
    python predict_digit.py path_to_image
    """
    if len(sys.argv) != 2:
        print("Usage: python predict_digit.py path_to_image")
        sys.exit(1)

    image_path = sys.argv[1]

    model = load_trained_model(MODEL_PATH)

    # debug=True -> shows bounding boxes and per-digit images
    number = recognize_number_from_image(image_path, model, debug=True)

    print("\n==============================")
    print(f"Final detected number (left to right): {number}")
    print("==============================")


# ---------- GUI MODE ----------

def gui_main():
    """
    Simple GUI:
    - Button to select an image
    - Label to show the predicted multi-digit number
    """
    # Load model once for the GUI
    try:
        model = load_trained_model(MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # Create main window
    root = tk.Tk()
    root.title("Handwritten Digit Recognizer (MLP)")
    root.geometry("500x200")

    # StringVar to show selected file path and result
    selected_file_var = tk.StringVar(value="No image selected.")
    result_var = tk.StringVar(value="Result will appear here.")

    def on_select_image():
        """
        Callback for 'Select Image' button.
        Opens a file dialog, runs recognition, and updates the result label.
        """
        file_path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")]
        )
        if not file_path:
            return

        selected_file_var.set(f"Selected: {os.path.basename(file_path)}")

        try:
            number = recognize_number_from_image(file_path, model, debug=False)
            result_var.set(f"Detected number: {number}")
        except Exception as e:
            result_var.set("Error during recognition.")
            messagebox.showerror("Error", str(e))

    # GUI widgets
    btn_select = tk.Button(root, text="Select Image", command=on_select_image)
    lbl_file = tk.Label(root, textvariable=selected_file_var, wraplength=480, anchor="w", justify="left")
    lbl_result = tk.Label(root, textvariable=result_var, font=("Helvetica", 16, "bold"))

    # Layout
    btn_select.pack(pady=10)
    lbl_file.pack(pady=5, fill="x", padx=10)
    lbl_result.pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    # If an image path is given as argument -> CLI mode
    # Otherwise -> GUI mode
    if len(sys.argv) > 1:
        cli_main()
    else:
        gui_main()
