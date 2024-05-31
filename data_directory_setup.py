import os
import string

# Create main directories
main_dirs = ["training", "validation"]

for main_dir in main_dirs:
    # Create the main directory if it doesn't exist
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    # Create subdirectories A-Z
    for letter in string.ascii_uppercase:
        sub_dir = os.path.join(main_dir, letter)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)