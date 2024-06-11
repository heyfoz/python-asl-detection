# This script utilizes MediaPipe Hands, OpenCV, and  NumPy libraries
# to process images of hands, making the background black except for the detected hands,
# which are centered and scaled to a specified target size of 200x200 pixels
# The script iterates through a directory of images, detects hands, and saves the processed images, 
# while also keeping track of the number of deleted files. 
# The final count of remaining files is displayed after processing.

import cv2  # For image processing such as reading, resizing, and manipulation
import mediapipe as mp  # For hand tracking
import numpy as np  # To use numerical operations
import os  # For file operations
import imgaug.augmenters as iaa  # For image augmentation

val_dir = '/path/to/validation/images'
train_dir = '/path/to/training/images'
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.05,
    min_tracking_confidence=0.5)

def process_image(image_path, target_size=(200, 200), occupy_percent=0.8, is_training=False):
    image = cv2.imread(image_path)  # Reading the image
    if image is None:  # Check if the image was loaded successfully
        print(f"Failed to read image: {image_path}")
        return False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
    results = hands.process(image_rgb)  # Process image with MediaPipe Hands

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

    # Apply custom augmentation only for training images
    if is_training:
        centered_image = augment_training_imgs(centered_image)

    # Convert image to grayscale
    grayscale_image = cv2.cvtColor(centered_image, cv2.COLOR_RGB2GRAY)

    # Save the processed grayscale image with '_mediapiped' appended to the filename
    base_name, ext = os.path.splitext(image_path)
    new_image_path = f"{base_name}_mediapiped{ext}"
    if cv2.imwrite(new_image_path, grayscale_image):
        print(f"Saved processed file: {new_image_path}")
        print(f"Deleted original file: {image_path}")
        os.remove(image_path)
        return True
    else:
        print(f"Failed to save processed file: {new_image_path}")
        return False

def augment_training_imgs(image):
    # Apply custom augmentation only for training images
    transform = iaa.Sequential([
        iaa.Affine(
            scale=(0.95, 1.05),  # Random zoom
            rotate=(-5, 5),  # Random rotation
            # Translate after zooming and rotating
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}  # Random shifts
        )
    ], random_order=True)  # Apply transformations in random order

    augmented_image = transform.augment_image(image)
    return augmented_image

# Function to process all images in a directory
def process_directory(directory):
    deleted_count = {}  # Store count of deleted files in each subdirectory
    total_files_remaining = 0  # Store total remaining files after processing
    total_files_processed = 0  # Store total processed files
    total_files_deleted = 0  # Store total deleted files

    for subdir, dirs, files in os.walk(directory):
        num_deleted_in_subdir = 0
        subdir_name = os.path.basename(subdir)
        processed_files = 0  # Track the number of processed files
        total_files = len(files)  # Total number of files in the subdirectory

        print(f"Working on directory: {subdir_name}")

        is_training = directory == train_dir  # Check if processing training directory
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check if file is an image
                file_path = os.path.join(subdir, file)
                print(f"Processing file: {file_path}")
                if not process_image(file_path, is_training=is_training):  # Process image and handle deletion
                    os.remove(file_path)  # Remove image if no hands detected
                    num_deleted_in_subdir += 1
                    print(f"Deleted file: {file_path}")
                else:
                    total_files_remaining += 1

                processed_files += 1
                print(f"Processed {processed_files}/{total_files} files in {subdir_name}")

        deleted_count[subdir_name] = num_deleted_in_subdir
        total_files_processed += total_files
        total_files_deleted += num_deleted_in_subdir

    return deleted_count, total_files_remaining, total_files_processed, total_files_deleted

# List of directories containing images to be processed
directories = [val_dir, train_dir]
total_deleted = {}

# Dictionary to store total deleted files in each subdirectory
grand_total_files_remaining = 0  
grand_total_files_processed = 0  
grand_total_files_deleted = 0  

# Process each directory and accumulate results
for directory in directories:
    deleted_count, total_files_remaining, total_files_processed, total_files_deleted = process_directory(directory)
    total_deleted.update(deleted_count)
    grand_total_files_remaining += total_files_remaining
    grand_total_files_processed += total_files_processed
    grand_total_files_deleted += total_files_deleted

# Print results
for subdir, count in total_deleted.items():
    print(f"{subdir}: {count} deleted")

print(f"Total number of files processed: {grand_total_files_processed}")
print(f"Total number of files deleted: {grand_total_files_deleted}")
print(f"Total number of files remaining: {grand_total_files_remaining}")

hands.close()  # Close MediaPipe Hands
