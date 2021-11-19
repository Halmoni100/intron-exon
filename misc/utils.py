import os
import shutil
import pathlib
from subprocess import Popen

def rm_and_mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)