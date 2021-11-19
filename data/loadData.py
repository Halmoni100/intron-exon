import tensorflow as tf
import numpy as np
import itertools
#from functools import reduce


def get_data(seqFile, exonCoord, windowSize=1):
    """
    Read and load sequence file, create one long concatenated vector for all genes.
    If windowSize >1 then each position will be represended by itself and the several nucleotides behind it.
    Create corresponding binary indicator vector for positions of exons.

    i.e. if all sequence concatenated has length n,
    then the output Seq array will have dimensions windowSize by (n-windowSize+1)
    and the ExonIndicator will have dimensions 1 by (n-windowSize+1)


    :param seqFile: Path to the sequence file. Formated as output from Bedtools
    :param exonCoord: Path to the exon cordinate files. Each row corresponds to an exon, start/end coordinate that has concatenated all genes in the order presented in the sequence file
    :return: Tuple of Seq,dict, ExonIndicator
    """

    # create dictionary
    keys = itertools.product("AGCT", repeat=windowSize)
    keys = ["".join(p) for p in keys]
    dict = {keys[i]:i for i in range(len(keys))}

    # read sequence data
    with open(seqFile) as f:
        Seq = f.read()
        Seq = Seq.strip().split()
        # remove rows that has the gene information, which start with >
        Seq = [s for s in Seq if s[0]!=">"]
        Seq = "".join(Seq) # join genes
        Seq = Seq.upper() # turn to upper case
        totalLen = len(Seq)
        # create array given windowSize
        Seq = [Seq[0+p:windowSize+p] for p in range(len(Seq)-windowSize+1)]
        # tokenize sequence get_data
        Seq = [dict[w] for w in Seq]
    f.close()

    # read exon coordinates
    with open(exonCoord) as f:
        coord = f.read()
        coord = coord.strip().split()
        # create binary vector
        exonCoord = np.zeros(totalLen, dtype=np.int32)
        # mark each exon
        for c in coord:
            start = int(c.split(",")[0])-1
            end = int(c.split(",")[1])
            exonCoord[start:end] = 1
        # clean extra
        exonCoord = exonCoord[0:totalLen-windowSize+1]

    f.close()

    return Seq, dict, exonCoord
