import os
import numpy as np

from loadData import get_data

def initial_process_data(seq_file, exon_file):
    seq_file_path = os.path.join(os.environ['INTRON_EXON_ROOT'], seq_file)
    exon_file_path = os.path.join(os.environ['INTRON_EXON_ROOT'], exon_file)
    seq, seq_dict, exon_coord = get_data(seq_file_path, exon_file_path)
    return seq, seq_dict, exon_coord

def preprocess(seq, exon_coord, window_size):
    num_nucleotides = len(seq)
    num_windows = num_nucleotides // window_size
    windowed_seq = np.array(seq[:num_windows * window_size]).reshape((num_windows, window_size))
    windowed_exon = exon_coord.reshape(num_windows, window_size)
    return windowed_seq, windowed_exon
