from conf import Conf
import numpy as np
from Bio import SeqIO
from Bio.Alphabet import generic_dna
from Bio.Seq import Seq
import pandas as pd
from conf import Conf

### MUST USE PYTHON 2!! FOR BIOPYTHON... ###


def getSurroundingSeqTablePerChr(chr, dfOut, offset=Conf.numSurrounding):

    dfProbeToSeq = []
    # for row in dfMethyl.iterrows():
    dfMethyl["Chr"] = dfMethyl["Chr"].apply(str)
    dfMethylCurrent = dfMethyl[dfMethyl["Chr"]==chr]
    record = SeqIO.read(dir_hg19 + "chr" + str(chr) + ".fa", "fasta")

    for i in range(len(dfMethylCurrent)):
        # if i == 1:
        #     break
        print(str(i)+" out of %d" %len(dfMethylCurrent))
        probeID = dfMethylCurrent["Probe"].iloc[i]
        print("chr ", str(dfMethylCurrent["Chr"].iloc[i]))
        pos = int(dfMethylCurrent["Pos"].iloc[i])
        # print pos
        seq = record[pos-offset:pos+offset]
        dfProbeToSeq.append([probeID, str(seq.seq)])
    dfSurroundingSeqInterim = pd.DataFrame(dfProbeToSeq)
    try:
        dfSurroundingSeqInterim.columns = ["Probe", "Seq"]
    except:
        pass
    dfSurroundingSeqInterim.to_csv(Conf.probeToSurroundingSeqFilePrefixChr+chr+'_.csv', index=None)

    return dfProbeToSeq


# SETTINGS #
dir_hg19 = '../res/hg19/'
# SETTINGS #

chrArr = [str(i) for i in range(1,23)]
chrArr.extend(['X','Y'])
dfMethyl = pd.read_csv('../res/DNA_methylation_probe_info_location.txt', header=0, sep='\t', dtype={'Chr': object})
dfSurroundingSeq = []

for c in chrArr:
    probeToSeqPerChr = getSurroundingSeqTablePerChr(c, dfSurroundingSeq, Conf.numSurrounding)
    dfSurroundingSeq.extend(probeToSeqPerChr)
    # print dfSurroundingSeq

dfSurroundingSeq = pd.DataFrame(dfSurroundingSeq)
dfSurroundingSeq.columns = ["Probe", "Seq"]
dfSurroundingSeq.to_csv(Conf.probeToSurroundingSeqFilePrefixAll+'_.csv', index=None)

# with open("output.fas", "w") as out:
#     SeqIO.write(record[100:500], out, "fasta")

