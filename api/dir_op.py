import os


def delete_all_files_in_dir(path):
    fn_array = os.listdir(path)
    # clear
    for fn in fn_array:
        os.remove(path + fn)


def delete_all_files_in_dirs(paths):
    for path in paths:
        delete_all_files_in_dir(path=path)
