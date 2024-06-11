# This utility script reads the image path, and displays a UI to view a random augmentation of the image on each button click
# Demonstrates the mediapipe preprocessing and the custom augmentation of the training images

import cv2  # For image processing such as reading, resizing, and manipulation
import imgaug.augmenters as iaa  # For image augmentation
import matplotlib.pyplot as plt  # For plotting images
import tkinter as tk  # For creating the UI
from PIL import ImageTk, Image  # For displaying images in the UI
import mediapipe as mp  # For hand tracking
import numpy as np  # To use numerical operations

image_path = '/path/to/test/image.png'  # Replace this with the path to your image ('C:\\path\\to\\test\\image.png for Windows)

# Function to process a single image with random augmentation
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

    # Apply custom augmentation
    transform = iaa.Sequential([
        iaa.Rotate((-5, 5)),  # Random rotation
        iaa.Affine(
            #scale=(0.9, 1.1),  # Random zoom
            #shear=(-.5, .5),  # Random shear
            #translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}  # Random shifts
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

# Create a Tkinter window
root = tk.Tk()
root.title("Image Augmentation")

# Load an example image
current_image_path = image_path
image = Image.open(current_image_path)
image_tk = ImageTk.PhotoImage(image)

# Create a label to display the image
img_label = tk.Label(root, image=image_tk)
img_label.pack()

# Create a button to trigger augmentation
btn_augment = tk.Button(root, text="Apply Augmentation", command=on_button_click)
btn_augment.pack()

# Run the Tkinter event loop
root.mainloop()
