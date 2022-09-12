import glob
import os.path
from typing import List


def globSSF(path_name, root_dir) -> List[str]:
    root_length = len(root_dir)
    return [pth[root_length + 1:] for pth in glob.glob(os.path.normpath(os.path.join(root_dir, path_name)))]


def __always_true(*args, **kwargs) -> bool: return True


def _to_str(i):
    if not 99 >= i >= 0 or not round(i) == i:
        raise ValueError(f"i should be 2 digits, but instead it is {i}")
    return str(i) if i >= 10 else f"0{i}"
