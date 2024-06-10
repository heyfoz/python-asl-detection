# Optional script to add git keep file in the sub directories so they are pushed to repo even if empty.
import os

def create_gitkeep_in_empty_dirs(parent_dir):
    for root, dirs, files in os.walk(parent_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):  # Check if the directory is empty
                gitkeep_path = os.path.join(dir_path, '.gitkeep')
                with open(gitkeep_path, 'w') as f:
                    pass  # Create an empty .gitkeep file
                print(f"Created .gitkeep in {dir_path}")

def main():
    current_dir = os.getcwd()
    training_dir = os.path.join(current_dir, 'training')
    validation_dir = os.path.join(current_dir, 'validation')

    create_gitkeep_in_empty_dirs(training_dir)
    create_gitkeep_in_empty_dirs(validation_dir)

if __name__ == '__main__':
    main()
