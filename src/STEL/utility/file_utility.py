import os
import pickle
from typing import List

import pandas as pd


class PickleObjectGenerator:

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename, "rb") as f:
            while True:
                try:
                    file_object = pickle.load(f)
                    yield file_object
                except EOFError:
                    break


def file_lines_to_list(file_name):
    with open(file_name, 'r') as f:
        result = list(f.readlines())
    return [sentence[:-1] for sentence in result]


# PANDAS file utility
def read_tsv_to_pd(file_tsv: str):
    cur_df = pd.read_csv(file_tsv, sep='\t')
    return cur_df


def get_dir_to_src():
    dir_path = os.path.dirname(os.path.normpath(__file__))
    base_dir = os.path.basename(dir_path)
    if base_dir == "utility":
        return os.path.dirname(os.path.dirname(dir_path))
    elif base_dir == "STEL":
        return os.path.dirname(dir_path)
    else:
        return dir_path


def read_tsv_list_to_pd(csv_file_list: List[str]):
    """
    :param csv_file_list: list of file paths to csv files separated with tabs
    :return: dataframe of concatenated dataframe, beginning with the first element in the string list
    """
    csv_df = read_tsv_to_pd(csv_file_list[0])  # , index_col=0
    for file_tsv in csv_file_list[1:]:
        cur_df = read_tsv_to_pd(file_tsv)
        csv_df = pd.concat([csv_df, cur_df])
    return csv_df


def ensure_path_exists(save_path):
    # https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
    dir_name = os.path.dirname(save_path)
    from pathlib import Path
    Path(dir_name).mkdir(parents=True,
                         exist_ok=True)
