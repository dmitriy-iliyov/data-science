import os


def count_files_in_directory(directory):
    files_and_dirs = os.listdir(directory)
    files = [f for f in files_and_dirs if os.path.isfile(os.path.join(directory, f))]
    return len(files)