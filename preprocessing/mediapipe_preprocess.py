# This script utilizes MediaPipe Hands, OpenCV, and  NumPy libraries
# to process images of hands, making the background black except for the detected hands,
# which are centered and scaled to a specified target size of 200x200 pixels
# The script iterates through a directory of images, detects hands, and saves the processed images, 
# while also keeping track of the number of deleted files. 
# The final count of remaining files is displayed after processing.

import cv2 # For image processing such as reading, resizing, and manipulation
import mediapipe as mp # For hand tracking
import numpy as np # To use numerical operations
import os # For file operations

val_dir = '/path/to/validation/images'
train_dir = '/path/to/training/images'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.05,
    min_tracking_confidence=0.5)

def make_background_black_except_hands_and_center(image_path, target_size=(200, 200), occupy_percent=0.8):
    image = cv2.imread(image_path)  # Reading the image
    if image is None:  # Check if the image was loaded successfully
        return False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
    results = hands.process(image_rgb)  # Process image with MediaPipe Hands

    if not results.multi_hand_landmarks: # Check if hands were detected in the image
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

    # Save the processed image
    base_name, ext = os.path.splitext(image_path)
    new_image_path = f"{base_name}_centered_scaled{ext}"
    cv2.imwrite(new_image_path, centered_image)
    return True

# Function to process all images in a directory
def process_directory(directory):
    deleted_count = {}  # Store count of deleted files in each subdirectory
    total_files_remaining = 0  # Store total remaining files after processing

    for subdir, dirs, files in os.walk(directory):
        num_deleted_in_subdir = 0
        subdir_name = os.path.basename(subdir)
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')): # Check if file is an image
                file_path = os.path.join(subdir, file)
                if not make_background_black_except_hands_and_center(file_path):  # Process image
                    os.remove(file_path)  # Remove image if processing fails
                    num_deleted_in_subdir += 1
                else:
                    total_files_remaining += 1

        deleted_count[subdir_name] = num_deleted_in_subdir

    return deleted_count, total_files_remaining

# List of directories containing images to be processed
# directories = ['/mnt/mydisk/asl33/blank2/Test_Alphabet', '/mnt/mydisk/asl33/blank2/Train_Alphabet']
directories = [val_dir, train_dir]
total_deleted = {}  # Dictionary to store total deleted files in each subdirectory
grand_total_files_remaining = 0  # Variable to store total remaining files after processing

# Process each directory and accumulate results
for directory in directories:
    deleted_count, total_files_remaining = process_directory(directory)
    total_deleted.update(deleted_count)
    grand_total_files_remaining += total_files_remaining

# Print results
for subdir, count in total_deleted.items():
    print(f"{subdir}: {count} deleted")

print(f"Total number of files remaining: {grand_total_files_remaining}")
hands.close() # Close MediaPipe Hands
