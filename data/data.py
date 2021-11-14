import os, sys
sys.path.append(os.environ['INTRON_EXON_ROOT'])

import numpy as np
import math

if __name__ == "__main__":
    from loadData import get_data
else:
    from data.loadData import get_data
from scripts.utils import rm_and_mkdir

def initial_process_data(seq_file, exon_file):
    seq_file_path = os.path.join(os.environ['INTRON_EXON_ROOT'], seq_file)
    exon_file_path = os.path.join(os.environ['INTRON_EXON_ROOT'], exon_file)
    seq, seq_dict, exon_coord = get_data(seq_file_path, exon_file_path)
    return seq, seq_dict, exon_coord

def preprocess(seq, exon_coord, window_size):
    num_nucleotides = len(seq)
    num_windows = num_nucleotides // window_size
    windowed_seq = np.array(seq[:num_windows * window_size]).reshape((num_windows, window_size))
    windowed_exon = exon_coord[:num_windows * window_size].reshape(num_windows, window_size)
    return windowed_seq, windowed_exon[:,:,None]

def split_train_val_test(windowed_seq, windowed_exon, train_ratio=0.7, val_ratio=0.15):
    num_examples = windowed_seq.shape[0]
    indices = np.arange(num_examples)
    np.random.seed(1)
    np.random.shuffle(indices)
    windowed_seq_shuffled = windowed_seq[indices]
    windowed_exon_shuffled = windowed_exon[indices]
    train_end_idx = math.floor(num_examples * train_ratio)
    val_end_idx = math.floor(num_examples * (train_ratio + val_ratio))
    train_seq = windowed_seq_shuffled[:train_end_idx]
    train_exon = windowed_exon_shuffled[:train_end_idx]
    val_seq = windowed_seq_shuffled[train_end_idx:val_end_idx]
    val_exon = windowed_exon_shuffled[train_end_idx:val_end_idx]
    test_seq = windowed_seq_shuffled[val_end_idx:]
    test_exon = windowed_exon_shuffled[val_end_idx:]
    return train_seq, train_exon, val_seq, val_exon, test_seq, test_exon

def get_exon_ratio(exon_coord):
    return np.sum(exon_coord) / len(exon_coord)

def main():
    window_size = 4000
    seq_file = "dataInput_exonIntron500/seqAllInOne.txt"
    exon_file = "dataInput_exonIntron500/exonCoordinates_allInOne.csv"
    seq, seq_dict, exon_coord = initial_process_data(seq_file, exon_file)
    windowed_seq, windowed_exon = preprocess(seq, exon_coord, window_size)
    train_seq, train_exon, val_seq, val_exon, test_seq, test_exon = \
        split_train_val_test(windowed_seq, windowed_exon)

    data_dir = os.path.join(os.environ['INTRON_EXON_ROOT'], "data/split_" + str(window_size))
    rm_and_mkdir(data_dir)

    train_seq_file = os.path.join(data_dir, "train_seq.npy")
    np.save(train_seq_file, train_seq)
    train_exon_file = os.path.join(data_dir, "train_exon.npy")
    np.save(train_exon_file, train_exon)
    val_seq_file = os.path.join(data_dir, "val_seq.npy")
    np.save(val_seq_file, val_seq)
    val_exon_file = os.path.join(data_dir, "val_exon.npy")
    np.save(val_exon_file, val_exon)
    test_seq_file = os.path.join(data_dir, "test_seq.npy")
    np.save(test_seq_file, test_seq)
    test_exon_file = os.path.join(data_dir, "test_exon.npy")
    np.save(test_exon_file, test_exon)

if __name__ == "__main__":
    main()
