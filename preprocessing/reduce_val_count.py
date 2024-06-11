# Utility to reduce number of validation image files to a target number of images, randomly deleting the rest
import os
import glob
import random

def reduce_files_in_subdir(subdir_path, target_file_count=80):
    file_paths = glob.glob(os.path.join(subdir_path, '*.*'))  # List all files
    current_count = len(file_paths)

    if current_count > target_file_count:
        files_to_delete = random.sample(file_paths, current_count - target_file_count)
        for file_path in files_to_delete:
            os.remove(file_path)
        print(f"Reduced files in {os.path.basename(subdir_path)} from {current_count} to {target_file_count}.")
    else:
        print(f"No change needed in {os.path.basename(subdir_path)}. Current count: {current_count}")

def process_directories(base_directory):
    for char in map(chr, range(65, 91)):  # ASCII values for A to Z
        subdir_path = os.path.join(base_directory, char)
        if os.path.isdir(subdir_path):
            reduce_files_in_subdir(subdir_path)

# Base directory path
base_dir_path = '/path/to/validation/dir'

# Process each subdirectory
print("Processing directories...")
process_directories(base_dir_path)
print("Directory processing completed.")
