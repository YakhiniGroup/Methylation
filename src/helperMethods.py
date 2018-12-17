from sklearn import preprocessing
import pickle as pkl
import sys
from scipy import stats
import pandas as pd
import csv
import numpy as np
from conf import Conf, ConfSample

def save_obj(obj, name, directory='../res/'):
    with open(directory + name + '.pkl', 'wb') as f:
        pkl.dump(obj, f)


def load_obj(name, directory='../res/'):
    with open(directory + name + '.pkl', 'rb') as f:
        return pkl.load(f)

def calculate_zscore(row):
    return (row - row.mean())/float(row.std())

def oneHotMtrxToSeq(mtrx):
    mtrx = mtrx.reshape([-1, Conf.numBases])
    seq = ''
    idx_to_nt_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
    for row in mtrx:
        idx_1 = np.argmax(row)
        nt = idx_to_nt_dict[idx_1]
        seq += nt
    return seq

def seqToOneHotMtrx(seq):
    if 'N' in seq:
        print("N in seq. replaced with 5th value - 4")
    seq = seq.replace('A','0').replace('a','0').replace('C', '1').replace('c','1').replace('G', '2').replace('g','2').replace('T', '3').replace('t','3').replace('N','4')
    seqList = list(seq)
    seqList = [int(i) for i in seqList]
    zeros = np.zeros((len(seq),Conf.numBases))
    zeros[np.arange(len(seq)),seqList] = 1
    return zeros

def applyZscores(filename, numDecimals, directory='../res/', on="cpgs"):
    data = pd.read_csv(directory + filename)
    firstCol = data.iloc[:,0]
    cols = data.columns
    # z-score scaling "other" & "dist" data (gene expr. E)
    if on == "genes":
        std_scale = preprocessing.StandardScaler().fit(data[data.columns[1:]])
        data[data.columns[1:]] = std_scale.transform(data[data.columns[1:]])

    elif on == "cpgs":
        data[data.columns[1:]] = data[data.columns[1:]].apply(calculate_zscore, axis=1)
    for col in data.columns[1:]:
        data[col] = data[col].astype(float).round(numDecimals)
    data.to_csv(directory + filename.split('.')[0] + "_Z.csv", index=None)

if __name__ == '__main__':
    # matrx = seqToOneHotMtrx('AGGTCNNNTC')
    # matrx = matrx.flatten()
    # print(oneHotMtrxToSeq(matrx))
    print(np.append([1,0], [0,0]))
    # for i in range(len(data)):
    #     if i % 1 == 0:
    #         print("aaaaaa",i)
    #     data.iloc[i,1:] = (data.iloc[i,1:] - data.iloc[i,1:].mean()) / data.iloc[i,1:].std()
    # data = stats.zscore(data[data.columns[1:]], axis=1)
    # data.round(numDecimals)
    # data = pd.DataFrame(data)
    # data.columns = cols

    # data = np.insert(data, 0, cols, axis=0)


    # with open(directory+filename.split('.')[0]+"_Z.csv", "wb") as csv_file:
    #     writer = csv.writer(csv_file, delimiter=',')
    #     for line in data:
    #         writer.writerow(line)
    # with open(directory+filename.split('.')[0]+"_Z.csv", "wb") as output:
    #     writer = csv.writer(output, lineterminator='\n')
    #     writer.writerows(data)
    # np.savetxt(directory+filename.split('.')[0]+"_Z.csv", data, delimiter=',')  # X is an array