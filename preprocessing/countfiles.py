# Utility prints number of files in the subdirectories inside training and validation
# Used to count number of images after mediapipe_preprocess.py is executed
import os

train_dir = '/path/to/training/dir'
val_dir = '/path/to/validation/dir'

def count_files_in_subdirectories(directory):
    subdirs = sorted(os.listdir(directory))  # Sort the directory names alphabetically
    total_files = 0

    for subdir in subdirs:
        path = os.path.join(directory, subdir)
        if os.path.isdir(path):
            file_count = len(os.listdir(path))
            print(f"  {subdir}: {file_count} files")
            total_files += file_count

    print(f"Total files in {directory}: {total_files}")

# Count files in training and testing directories
train_total = count_files_in_subdirectories(train_dir)
val_total = count_files_in_subdirectories(val_dir)
