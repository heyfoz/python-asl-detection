# This utility script reads the image path, and displays a UI to view a random augmentation of the image on each button click
# Demonstrates the mediapipe preprocessing and the custom augmentation of the training images

import cv2  # For image processing such as reading, resizing, and manipulation
import imgaug.augmenters as iaa  # For image augmentation
import matplotlib.pyplot as plt  # For plotting images
import os  # For file operations
import tkinter as tk  # For creating the UI
from PIL import ImageTk, Image  # For displaying images in the UI
import mediapipe as mp  # For hand tracking
import numpy as np  # To use numerical operations

# Global variables for augmentation parameters
image_path = '/path/to/test/image.png'  # Replace this with the path to your image ('C:\\path\\to\\test\\image.png for Windows)
rotation_range = (-5, 5)
zoom_range = (0.9, 1.1)
shear_range = (-0.5, 0.5)
x_shift_range = (-0.05, 0.05)
y_shift_range = (-0.05, 0.05)

# Function to process a single image with user-defined augmentation parameters
def process_single_image(image_path, target_size=(200, 200), occupy_percent=0.8):
    global img_label

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return False

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.05,
        min_tracking_confidence=0.5)

    # Process the image with MediaPipe Hands
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:  # Check if hands were detected in the image
        print(f"No hands detected in image: {image_path}")
        return False

    # Extract hand landmarks and create a binary mask for the hands
    height, width, _ = image.shape
    hand_mask = np.zeros((height, width), dtype=np.uint8)
    for hand_landmarks in results.multi_hand_landmarks:
        points = []
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * width), int(lm.y * height)
            points.append((x, y))
        cv2.fillPoly(hand_mask, [np.array(points, dtype=np.int32)], 255)

    # Dilate and blur hand mask to smooth edges
    kernel = np.ones((5, 5), np.uint8)
    hand_mask_dilated = cv2.dilate(hand_mask, kernel, iterations=1)
    hand_mask_blurred = cv2.GaussianBlur(hand_mask_dilated, (21, 21), 0)

    # Create an image with the detected hand on a black background
    hand_on_black_background = np.where(hand_mask_blurred[..., None] > 0, image, np.zeros_like(image))

    # Find bounding box around hands and resize to target size
    contours, _ = cv2.findContours(hand_mask_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found in image: {image_path}")
        return False
    x_min, x_max, y_min, y_max = width, 0, height, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x + w), max(y_max, y + h)
    hand_area = hand_on_black_background[y_min:y_max, x_min:x_max]
    hand_width, hand_height = x_max - x_min, y_max - y_min
    scale_factor = min(target_size[0] / hand_width, target_size[1] / hand_height) * occupy_percent
    scaled_w, scaled_h = int(hand_width * scale_factor), int(hand_height * scale_factor)
    resized_hand_area = cv2.resize(hand_area, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    # Center hand in an image of target size
    centered_image = np.zeros(target_size + (3,), dtype=np.uint8)
    x_offset = (target_size[0] - scaled_w) // 2
    y_offset = (target_size[1] - scaled_h) // 2
    centered_image[y_offset:y_offset + scaled_h, x_offset:x_offset + scaled_w] = resized_hand_area

    # Apply custom augmentation based on user-defined parameters
    transform = iaa.Sequential([
        iaa.Rotate(rotation_range),  # Random rotation
        iaa.Affine(
            scale=zoom_range,  # Random zoom
            shear=shear_range,  # Random shear
            translate_percent={"x": x_shift_range, "y": y_shift_range}  # Random shifts
        )
    ], random_order=True)  # Apply transformations in random order
    augmented_image = transform.augment_image(centered_image)

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

    # Display the augmented image
    display_image(grayscale_image)

# Function to display the image in the UI
def display_image(image):
    global img_label
    # Convert image to RGB format for displaying in tkinter
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)

    # Update the image label in the UI
    img_label.config(image=image_tk)
    img_label.image = image_tk  # Keep reference to the image to prevent garbage collection

# Function to handle button click event
def on_button_click():
    global current_image_path
    process_single_image(current_image_path)

# Function to update augmentation parameters based on user input
def update_parameters():
    global rotation_range, zoom_range, shear_range, x_shift_range, y_shift_range
    rotation_range = (float(entry_rotation_min.get()), float(entry_rotation_max.get()))
    zoom_range = (float(entry_zoom_min.get()), float(entry_zoom_max.get()))
    shear_range = (float(entry_shear_min.get()), float(entry_shear_max.get()))
    x_shift_range = (float(entry_x_shift_min.get()), float(entry_x_shift_max.get()))
    y_shift_range = (float(entry_y_shift_min.get()), float(entry_y_shift_max.get()))

# Create a Tkinter window
root = tk.Tk()
root.title("Image Augmentation")

# Load an example image
image = Image.open(current_image_path)
image_tk = ImageTk.PhotoImage(image)

# Create a label to display the image
img_label = tk.Label(root, image=image_tk)
img_label.pack()

# Create entry fields for augmentation parameters
tk.Label(root, text="Rotation Range (min, max):").pack()
entry_rotation_min = tk.Entry(root)
entry_rotation_min.pack()
entry_rotation_min.insert(0, str(rotation_range[0]))
entry_rotation_max = tk.Entry(root)
entry_rotation_max.pack()
entry_rotation_max.insert(0, str(rotation_range[1]))

tk.Label(root, text="Zoom Range (min, max):").pack()
entry_zoom_min = tk.Entry(root)
entry_zoom_min.pack()
entry_zoom_min.insert(0, str(zoom_range[0]))
entry_zoom_max = tk.Entry(root)
entry_zoom_max.pack()
entry_zoom_max.insert(0, str(zoom_range[1]))

tk.Label(root, text="Shear Range (min, max):").pack()
entry_shear_min = tk.Entry(root)
entry_shear_min.pack()
entry_shear_min.insert(0, str(shear_range[0]))
entry_shear_max = tk.Entry(root)
entry_shear_max.pack()
entry_shear_max.insert(0, str(shear_range[1]))

tk.Label(root, text="X Shift Range (min, max):").pack()
entry_x_shift_min = tk.Entry(root)
entry_x_shift_min.pack()
entry_x_shift_min.insert(0, str(x_shift_range[0]))
entry_x_shift_max = tk.Entry(root)
entry_x_shift_max.pack()
entry_x_shift_max.insert(0, str(x_shift_range[1]))

tk.Label(root, text="Y Shift Range (min, max):").pack()
entry_y_shift_min = tk.Entry(root)
entry_y_shift_min.pack()
entry_y_shift_min.insert(0, str(y_shift_range[0]))
entry_y_shift_max = tk.Entry(root)
entry_y_shift_max.pack()
entry_y_shift_max.insert(0, str(y_shift_range[1]))

# Create a button to update parameters and trigger augmentation
btn_update = tk.Button(root, text="Update Parameters", command=update_parameters)
btn_update.pack()

btn_augment = tk.Button(root, text="Apply Augmentation", command=on_button_click)
btn_augment.pack()

# Start the Tkinter main loop
root.mainloop()
