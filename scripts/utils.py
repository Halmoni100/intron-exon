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

def run_bash_script(script_str, temp_dir):
    script_str_lines = script_str.split('\n')
    script_str_lines_processed = [line.strip() for line in script_str_lines]
    script_str_processed = '\n'.join(script_str_lines_processed)

    mkdir_if_not_exist(temp_dir)

    script_file_path = os.path.join(temp_dir, 'script.sh')
    f = open(script_file_path, 'w')
    f.write('#!/usr/bin/env bash\n')
    f.write(script_str_processed)
    f.close()
    os.chmod(script_file_path, 0b111111101)

    proc = Popen(script_file_path)
    proc.wait()