# This is an automated example of how to manually create training and validation directories within the current direct
# with subdirectories A-Z also created in both trainind and validation 

# Note: This is not needed if usilizing a dataset with directories already created, but for this project, only classes A-Z were used.
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
