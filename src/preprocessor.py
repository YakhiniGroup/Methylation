import random
import tensorflow as tf
import pickle as pkl
import numpy as np
import pandas as pd
from conf import Conf, ConfSample
from helperMethods import *
import os
from distances import *
from distances import *

dirInterims = '../res/interims/'
dirProcessed = '../res/processed/'

def createListOfProbesInData():
    data = pd.read_csv(Conf.dfMethyl)
    probes = np.array(data.iloc[:,0])
    print(probes[0:10])
    print(len(probes))
    save_obj(probes,'probes')


def createProbeDict():
    chrToProbeList = {}

    for chr in Conf.chrArr:
        data = pd.read_csv(Conf.probeToSurroundingSeqFilePrefixChr + chr + '_.csv')

        for i in range(len(data)):
            probe = data["Probe"].iloc[i]
            if chr in chrToProbeList.keys():
                chrToProbeList[chr].append(probe)
            else:
                chrToProbeList[chr] = [probe]

    save_obj(chrToProbeList, "chrToProbeList")

#
# def save_obj(obj, name):
#     with open('../res/' + name + '.pkl', 'wb') as f:
#         pkl.dump(obj, f)
#
#
# def load_obj(name):
#     with open('../res/' + name + '.pkl', 'rb') as f:
#         return pkl.load(f)


def _seqToOneHotMtrx(seq):
    if 'N' in seq:
        print("N in seq. replaced with 5th value - 4")
    seq = seq.replace('A','0').replace('a','0').replace('C', '1').replace('c','1').replace('G', '2').replace('g','2').replace('T', '3').replace('t','3').replace('N','4')
    seqList = list(seq)
    seqList = [int(i) for i in seqList]
    zeros = np.zeros((len(seq),Conf.numBases))
    zeros[np.arange(len(seq)),seqList] = 1
    return zeros


def surroundingSeqToOneHotMtrx(perChromosome):
    if perChromosome:
        for c in Conf.chrArr:
            probeToOneHotMtrx = {}
            data = pd.read_csv(Conf.probeToSurroundingSeqFilePrefixChr + c +'_.csv', index_col=False)

            for i in range(len(data)):
                probe = data["Probe"].iloc[i]
                seq = data["Seq"].iloc[i]
                oneHotMtrx = _seqToOneHotMtrx(seq)
                probeToOneHotMtrx[probe] = oneHotMtrx
            # print probeToOneHotMtrx

            save_obj(probeToOneHotMtrx, Conf.probeToOneHotMtrxFilePrefixChr+c, directory='')
            # d =load_obj('probeToOneHotMtrx_'+c)
            print(c + " done")
            # print d
    else:
        probeToOneHotMtrx = {}
        for c in Conf.chrArr:
            data = pd.read_csv(Conf.probeToSurroundingSeqFilePrefixChr+c+'_.csv', index_col=False)
            for i in range(len(data)):
                probe = data["Probe"].iloc[i]
                seq = data["Seq"].iloc[i]
                oneHotMtrx = _seqToOneHotMtrx(seq)
                probeToOneHotMtrx[probe] = oneHotMtrx
            print(c + " done")
        save_obj(probeToOneHotMtrx, Conf.probeToOneHotMtrxFilePrefixAll, directory='')


def transposeE(filename_E='TCGA_E_final', filename_CH3='TCGA_CH3_final', probeDataFileName=Conf.probeToOneHotMtrxFilePrefixAll+'.csv'):

    probeData = pd.read_csv(probeDataFileName, index_col=None)
    df_E = pd.read_csv('../res/'+filename_E+'.csv')
    df_CH3 = pd.read_csv('../res/'+filename_CH3+'.csv')

    df_E = df_E.transpose()

    # renaming columns
    df_E.columns = df_E.iloc[0]
    df_E = df_E.iloc[1:,:]

    print(df_E.head())

    _assertAllDataInOrder(df_CH3, probeData, df_E, transposedE=True)

    df_E.to_csv('../res/'+filename_E+'_transposed.csv')
    # df_CH3.to_csv('../res/'+filename_CH3+'_t.csv', index=False)


#TODO:remove
# def createFullDataCSV():
#     left_E_t = pd.read_csv('../res/TCGA_E_t.csv')
#     right_CH3_t = pd.read_csv('../res/TCGA_CH3_t.csv')
#     merged = pd.merge(left_E_t, right_CH3_t, on='subject')
#     print(merged.head())
#     merged.to_csv('../res/fullData.csv', index=False)


def createSampleData(filename_E='TCGA_E_final_transposed', filename_CH3='TCGA_CH3_final', withDistances=False):
    df_E = pd.read_csv('../res/'+filename_E+'.csv', index_col=None)
    df_CH3 = pd.read_csv('../res/'+filename_CH3+'.csv', index_col=None)
    df_probeToOneHot = pd.read_csv(Conf.probeToOneHotPrefixAll+'.csv', index_col=None)
    if withDistances:
        df_dist = pd.read_csv('../res/distances.csv', index_col=None)

    print("num cols in E: %s" %(len(df_E.columns)))
    print("num cols in CH3: %s" % (len(df_CH3.columns)))

    # take only a few columns (i.e. subjects, say 100) and only a few rows (i.e. a few genes from E and a few CpGs from CH3)
    sample_E = df_E.iloc[0:Conf.numSampleSubjects, 0:Conf.numSampleGenes+1]
    sample_CH3 = df_CH3.iloc[0:Conf.numSampleProbes, 0:Conf.numSampleSubjects+1]
    sample_probeToOneHot = df_probeToOneHot.iloc[0:Conf.numSampleProbes,:]
    if withDistances:
        sample_dist = df_dist.iloc[0:Conf.numSampleProbes,0:Conf.numSampleGenes+1]
        print(sample_dist.head())
    print(sample_E.head())
    print(sample_CH3.head())
    print(sample_probeToOneHot.head())


    # _assertAllDataInOrder(sample_CH3, sample_probeToOneHot, sample_E, transposedE=True, distancesFile=sample_dist)
    suffix = "closest_gene_in20k"
    if withDistances:
        suffix = "_withDist"
        sample_dist.to_csv('../res/distances_sample'+suffix+'.csv', index=False)

    sample_E.to_csv('../res/TCGA_E_sample'+suffix+'.csv', index=False)
    sample_CH3.to_csv('../res/TCGA_CH3_sample'+suffix+'.csv', index=False)
    sample_probeToOneHot.to_csv('../res/probeToOneHotAll_sample'+suffix+'.csv', index=False)

    # colnames = ['subject'] + [str(i) for i in range(100)]
    # df_E = df_E.rename(columns=df_E.iloc[1])
    # df_CH3 = df_CH3.rename(columns=df_CH3.iloc[1])
    # df_CH3.columns = df_CH3.iloc[0]
    # df_E.columns = df_E.iloc[0]
    # df_CH3 = df_CH3[1:]  # take the data less the header row
    # df_E = df_E[1:]  # take the data less the header row
    #
    # df_E['subject'] = df_E.index
    # df_CH3['subject'] = df_CH3.index

    # left_E_t = pd.read_csv('../res/TCGA_E_t.csv')
    # right_CH3_t = pd.read_csv('../res/TCGA_CH3_t.csv')
    # merged = pd.merge(df_E, df_CH3, on='subject')
    # print(merged.head())
    # merged.to_csv('../res/fullData.csv', index=False)

def probeToOneHotAsCSVcolumnsTooLargeForPkl(alreadyHaveSeqToOneHotDictPerChr=True):
    if not alreadyHaveSeqToOneHotDictPerChr:
        surroundingSeqToOneHotMtrx(True)
    with open(Conf.probeToOneHotMtrxFilePrefixChr+str(1)+'.csv', 'a') as f:
        for c in Conf.chrArr[1:]:
            df = pd.read_csv(Conf.probeToOneHotMtrxFilePrefixChr+str(c)+'.csv', index_col=False)
            df.to_csv(f, header=False)


def probeToOneHotAsCSVcolumns(isSample, perChromosome=False, chromosome=None, enhancersOnly=False):

    if isSample:
        df_CH3 = pd.read_csv('../res/TCGA_CH3_sample.csv')
        probesInData = df_CH3.columns
    else:
        probesInData = load_obj('probes')
    if enhancersOnly:
        dfEnhancer = pd.read_csv('../res/450k_ESR1_FOXA1_GATA3.txt',index_col=False, delimiter='\t')
        enhancerProbes = np.array(dfEnhancer.Probe[dfEnhancer.Enhancer == True])
        probesInData = np.intersect1d(probesInData,enhancerProbes)
    if perChromosome:
        probeToOneHot = load_obj(Conf.probeToOneHotMtrxFilePrefixChr+str(chromosome), directory='')
    else:
        probeToOneHot = load_obj(Conf.probeToOneHotMtrxFilePrefixAll, directory='')
    data = []
    rowLength = 0
    i=0

    probes = set(probeToOneHot.keys()).intersection(probesInData)
    for k in probes:
        print(i)
        row = [k]
        row.extend(list(probeToOneHot[k].flatten()))
        # print(row)
        data.append(row)
        if i == 0:
            rowLength = len(row)
        i+=1
        # if i>10:
        #     break

    data = pd.DataFrame(data)
    print(data.head())
    colnames = ['Probe']
    colnames.extend([str(i) for i in range(rowLength-1)])
    data.columns = colnames
    print(data.head())
    print(data.shape)
    return data


#TODO:remove
def createSingleDataFile():
    # load dictionary to extract CpG matrix
    df_CH3_sample = pd.read_csv('../res/TCGA_CH3_sample.csv')
    df_E_sample = pd.read_csv('../res/TCGA_E_sample.csv')
    df_oneHot = probeToOneHotAsCSVcolumns(isSample=True)

    save_obj(df_E_sample.columns, 'df_E_columnNames')

    # merge E with oneHot
    df_oneHot['key'] = 1
    df_E_sample['key'] = 1
    merged_oneHot_E = pd.merge(df_E_sample, df_oneHot, on='key')

    # add labels
    melted_CH3 = df_CH3_sample.melt(id_vars=['subject'])   # becomes: [subject, variable (cpgID), value (cpgValue)]
    melted_CH3.columns = ['subject', 'Probe', 'ProbeVal']
    merged_all = pd.merge(merged_oneHot_E, melted_CH3, on=['subject', 'Probe'])
    merged_all = merged_all.drop(['key'], 1)
    print(merged_all.head(15))
    merged_all.to_csv('../res/dataSample.csv', index=False)


#TODO:remove
def createTrainTestFilesDeepWideReady(isSample):
    assert isSample, "No non-run_example implemented yet"
    allData = pd.read_csv('../res/dataSample.csv')
    trainSubjects = random.sample(set(allData['subject']), int(0.8 * Conf.numSampleSubjects))
    trainProbes = random.sample(set(allData['Probe']), int(0.8 * Conf.numSampleProbes))

    testSubjects = [s for s in set(allData['subject']) if s not in trainSubjects]
    testProbes = [p for p in set(allData['Probe']) if p not in trainProbes]

    print(len(trainSubjects))
    print(len(testSubjects))

    trainData = allData.loc[allData['subject'].isin(trainSubjects)]
    trainData = trainData.loc[trainData['Probe'].isin(trainProbes)]

    testData = allData.loc[allData['Probe'].isin(testProbes)]
    testDataSubjectsInterim = allData.loc[allData['subject'].isin(testSubjects)]
    testDataSubjectsNotInclInProbe = testDataSubjectsInterim.loc[testDataSubjectsInterim['Probe'].isin(trainProbes)]
    testData = testData.append(testDataSubjectsNotInclInProbe)

    print(trainData.shape)
    print(testData.shape)

    trainData = trainData.drop(['subject','Probe'], 1)
    testData = testData.drop(['subject','Probe'], 1)

    trainData.to_csv('../res/trainSample.csv',index=False)
    testData.to_csv('../res/testSample.csv', index=False)


def _assertAllDataInOrder(df_CH3, probeData, df_E, transposedE, distancesProbes=[], distancesGenes=[]):
    assert len(df_CH3) == len(probeData), "ProbeToOneHot and CH3 not the same length!"
    if transposedE:
        try:
            assert (df_E.index == df_CH3.columns[1:]).all(), "E and CH3 columns not the same order"  +df_E.index+" df_CH3.cols "+df_CH3.columns[1:]
        except:
            assert (df_E.iloc[:,0] == df_CH3.columns[1:]).all(), "E and CH3 columns not the same order: df_E.cols" +df_E.iloc[:,0]+" df_CH3.cols "+df_CH3.columns[1:]
    else:
        assert (df_E.columns[1:] == df_CH3.columns[1:]).all(), "E and CH3 columns not the same order"  +df_E.columns+" df_CH3.cols "+df_CH3.columns
    assert (df_CH3['Probe'] == probeData['Probe']).all(), "probe seq data and CH3 probe values not the same order, probe: " +probeData['Probe']+" df_CH3.probe "+df_CH3['Probe']
    if len(distancesProbes)>0:
        assert (df_CH3['Probe'] == distancesProbes).all(), "distances file's probe doesn't match CH3 probe. CH3: \n" + df_CH3['Probe'] + "dist \n:" + distancesProbes
        if transposedE:
            assert (df_E.columns[1:] == distancesGenes.columns[1:]).all(), "E and dist columns not the same order" + df_E.index + " dist.cols " + distancesGenes.columns[1:]
        else:
            try:
                assert (df_E.iloc[:, 0] == distancesGenes.columns[1:]).all(), "E and dist columns not the same order: df_E.cols" + df_E.iloc[:,0] + " dist.cols " + distancesGenes.columns[1:]
            except:
                assert (df_E.index == distancesGenes.columns[1:]).all(), "E and dist columns not the same order" + df_E.index + " dist.cols " + distancesGenes.columns[1:]


def createExpressionFileWithHighVarGenes(percentToKeep=None):
    from operator import itemgetter
    geneToCV = []
    df_E = pd.read_csv(Conf.dfExpression, index_col=False)
    n_genesTot = len(df_E.iloc[:,0])
    for i in range(n_genesTot):
        geneName = df_E.iloc[i,0]
        try:
            CV = df_E.iloc[i,1:].std()*100 / float(df_E.iloc[i,1:].mean())
        except:
            CV = 0
            print("failed CV calc. for gene %s" %geneName)
        geneToCV.append([geneName, CV])
    geneToCV = sorted(geneToCV, key=itemgetter(1), reverse=True)
    droppedGenes = [gene for gene, _ in geneToCV[round(len(geneToCV)*percentToKeep):]]
    df_E = df_E.set_index(df_E.iloc[:, 0])
    df_E = df_E.drop(droppedGenes)
    assert len(df_E.iloc[:,0]) < n_genesTot
    df_E.to_csv(Conf.dfExpression, index=False)


def createFileWithHighVarItems(filename, dstFilename, percentToKeep=None, identifier_col=0):
    from operator import itemgetter
    itemToCV = []
    df = pd.read_csv(filename, index_col=False)
    n_items = len(df.iloc[:,0])
    for i in range(n_items):
        identifier = df.iloc[i,0]
        try:
            CV = df.iloc[i,1:].std()*100 / float(df.iloc[i,1:].mean())
        except:
            CV = 0
            print("failed CV calc. for item %s" %identifier)
        itemToCV.append([identifier, CV])
    print("starting SORT")
    itemToCV = sorted(itemToCV, key=itemgetter(1), reverse=True)
    droppedItems = [item for item, _ in itemToCV[round(len(itemToCV)*percentToKeep):]]
    df = df.set_index(df.iloc[:, 0])
    df = df.drop(droppedItems)
    assert len(df.iloc[:,0]) < n_items
    # df.to_csv(dstFilename, index=False)

def split_data_indices_into_train_test_val(CH3_shape):
    res_dir = '../res/'
    try:
        train_probes_idx = load_obj("train_probes_idx", res_dir)
        train_subjects_idx = load_obj("train_subjects_idx", res_dir)
        validation_probes_idx = load_obj("validation_probes_idx", res_dir)
        validation_subjects_idx = load_obj("validation_subjects_idx", res_dir)
        test_probes_idx = load_obj("test_probes_idx", res_dir)
        test_subjects_idx = load_obj("test_subjects_idx", res_dir)
    except:
        num_probes = CH3_shape[0]
        num_subjects = CH3_shape[1]

        print("splitting subjects into train/test_ch3_e_blind/val")
        # subjects idx train/test_ch3_e_blind/val
        train_subjects_idx = random.sample(range(num_subjects), int(round(num_subjects * Conf.train_portion_subjects)))
        test_subjects_idx = [i for i in range(num_subjects) if i not in train_subjects_idx]
        validation_subjects_size = int(round(len(train_subjects_idx) * Conf.validation_portion_subjects))
        validation_subjects_idx = train_subjects_idx[0:validation_subjects_size]
        train_subjects_idx = train_subjects_idx[validation_subjects_size:]

        print("splitting probes into train/test_ch3_e_blind/val")
        # probes idx train/test_ch3_e_blind/val
        perm_num_probes = np.arange(num_probes)
        np.random.shuffle(perm_num_probes)
        # train_probes_idx = random.run_example(range(num_probes), int(round(num_probes * train_portion)))
        train_probes_idx = perm_num_probes[:int(round(num_probes * Conf.train_portion_probes))]
        test_probes_idx = perm_num_probes[int(round(num_probes * Conf.train_portion_probes)):]
        validation_probes_size = int(round(len(train_probes_idx) * Conf.validation_portion_probes))
        validation_probes_idx = train_probes_idx[0:validation_probes_size]
        train_probes_idx = train_probes_idx[validation_probes_size:]

        save_obj(train_probes_idx, "train_probes_idx", res_dir)
        save_obj(train_subjects_idx, "train_subjects_idx", res_dir)
        save_obj(validation_probes_idx, "validation_probes_idx", res_dir)
        save_obj(validation_subjects_idx, "validation_subjects_idx", res_dir)
        save_obj(test_probes_idx, "test_probes_idx", res_dir)
        save_obj(test_subjects_idx, "test_subjects_idx", res_dir)

    return train_probes_idx, train_subjects_idx, validation_probes_idx, validation_subjects_idx, test_probes_idx, test_subjects_idx


def createMatchingDataSets(probeToOneHotFileName, perChromosome, chromosome=None, withDistances=False, withRandomCpGs=False, filenameSuffix=''):
    # probe data
    probeToOneHot = pd.read_csv(probeToOneHotFileName, index_col=None)
    # CH3 and E data
    df_CH3 = pd.read_csv(Conf.dfMethyl, index_col=None)
    df_E = pd.read_csv(Conf.dfExpression, index_col=None)
    newColsCH3 = [i if i not in 'Unnamed: 0' else 'Probe' for i in df_CH3.columns]
    df_CH3.columns = newColsCH3
    newColsE = [i if i not in 'Unnamed: 0' else 'Gene' for i in df_E.columns]
    df_E.columns = newColsE

    if withDistances:
        distancesGenes = pd.read_csv(Conf.dfDistances, nrows=1)
        distancesProbes = pd.read_csv(Conf.dfDistances, usecols=[0])
        distancesProbes = list(distancesProbes.iloc[:, 0])

        # organizing CH3
        df_CH3 = df_CH3[df_CH3['Probe'].isin(distancesProbes)]
        print("sorting CH3")
        df_CH3.sort_values(by='Probe', inplace=True)  # by Probe (rows)
        df_CH3.sort_index(axis=1, inplace=True)  # by subject (columns)
        assert (df_CH3.iloc[:, 0] == distancesProbes).all()

        # organizing probeToOneHot:
        probeToOneHot = probeToOneHot[probeToOneHot['Probe'].isin(distancesProbes)]
        print("sorting probeToOneHot")
        probeToOneHot.sort_values(by='Probe', inplace=True)  # by Probe (rows)
        assert (probeToOneHot.iloc[:, 0] == distancesProbes).all()

        print("finished probe and ch3")

        # df_E
        # removing genes not in distance file
        df_E = df_E[df_E['Gene'].isin(distancesGenes)]
        indices_to_remove = [i for i, v in enumerate(list(df_E.iloc[:, 0])) if v not in distancesGenes]
        if len(indices_to_remove) > 0:
            print("%d found genes to remove" % len(indices_to_remove))
            df_E = df_E.drop(df_E.index[indices_to_remove])
        assert (df_E.iloc[:, 0] == distancesGenes.columns[1:]).all()
    else:
        df_CH3 = df_CH3[df_CH3['Probe'].isin(probeToOneHot['Probe'])]
        probeToOneHot = probeToOneHot[probeToOneHot['Probe'].isin(df_CH3['Probe'])]

    # organizing subjects in both probeData and E
    # probeData.sort_values(by='Probe', inplace=True)
    if not (df_E.columns[1:] == df_CH3.columns[1:]).all():
        print("sorting E")
        df_E.sort_index(axis=1, inplace=True)
        # put gene col back at first col position
        cols = list(df_E)
        cols.insert(0, cols.pop(cols.index('Gene')))
        df_E = df_E.loc[:, cols]

    # index resetting
    probeToOneHot = probeToOneHot.reset_index(drop=True)
    df_CH3 = df_CH3.reset_index(drop=True)

    # checking all in matching order
    if withDistances:
        _assertAllDataInOrder(df_CH3, probeToOneHot, df_E, transposedE=False, distancesProbes=distancesProbes, distancesGenes=distancesGenes)
    else:
        _assertAllDataInOrder(df_CH3, probeToOneHot, df_E, transposedE=False)

    # verifying that when including random CpGs, they're also in matching order
    if withRandomCpGs:
        # their distances
        randCpGDistData = pd.read_csv('../res/distancesCpGToCpG.csv', index_col=None)
        randCpGDistData.sort_values(by='Probe', inplace=True)
        assert (randCpGDistData.iloc[:,0] == df_CH3.iloc[:,0]).all()
        randCpGDistData.to_csv('../res/distancesCpGToCpG.csv', index=None)
        # their CH3 values
        randCpGmethylData = pd.read_csv('../res/CH3_inputCpGs_sample.csv', index_col=None) #TODO: change from run_example
        assert (randCpGmethylData['Probe'] == randCpGDistData['Probe']).all() # should already be OK because this was verified during construction
        #TODO: need to assert for subjects as well (although dealth with upon creation)
        # if run_example:
        #     relevantData.to_csv('../res/CH3_inputCpGs_sample.csv', index=None)
    # assert len(df_CH3) == len(probeData), "Not the same length!"
    # assert (df_E.columns[1:] == df_CH3.columns[1:]).all(), "E and CH3 columns not the same order"
    # assert (df_CH3['Probe'] == probeData['Probe']).all(), "probe seq data and CH3 probe values not the same order"

    # saving
    df_E.to_csv('../res/TCGA_E_final'+filenameSuffix+'.csv', index=False)
    if perChromosome:
        df_CH3.to_csv('../res/TCGA_CH3_final_Chr_'+str(chromosome)+filenameSuffix+'.csv', index=False)
        probeToOneHot.to_csv(Conf.probeToOneHotPrefixChr + str(chromosome) + filenameSuffix + '.csv', index=False)
    else:
        df_CH3.to_csv('../res/TCGA_CH3_final'+filenameSuffix+'.csv', index=False)
        probeToOneHot.to_csv(Conf.probeToOneHotPrefixAll + filenameSuffix + '.csv', index=False)
        save_obj(df_CH3.shape, "CH3_final_shape")
        save_obj(probeToOneHot.shape, 'probeToOneHot_final_shape')
        # if withDistances:
        #     distances.to_csv('../res/distances.csv', index=None)

    # distances = pd.read_csv(Conf.dfDistances, index_col=False)
    # train_probes_idx, train_subjects_idx, validation_probes_idx, validation_subjects_idx, test_probes_idx, test_subjects_idx = split_data_indices_into_train_test_val(CH3_shape=df_CH3.shape)
    #
    # save_subset_data(df_CH3, train_probes_idx, train_subjects_idx,"df_CH3", "_train")
    # save_subset_data(df_CH3, validation_probes_idx, validation_subjects_idx,"df_CH3", "_val")
    # save_subset_data(df_CH3, test_probes_idx, test_subjects_idx,"df_CH3", "_test")
    # save_subset_data(df_E, [], train_subjects_idx,"df_E", "_train")
    # save_subset_data(df_E, [], validation_subjects_idx,"df_E", "_val")
    # save_subset_data(df_E, [], test_subjects_idx,"df_E", "_test")
    # save_subset_data(distances, train_probes_idx, [],"dist", "_train")
    # save_subset_data(distances, validation_probes_idx, [],"dist", "_val")
    # save_subset_data(distances, test_probes_idx, [],"dist", "_test")
    # save_subset_data(probeToOneHot, train_probes_idx, [],"probeToOneHot", "_train")
    # save_subset_data(probeToOneHot, validation_probes_idx, [],"probeToOneHot", "_val")
    # save_subset_data(probeToOneHot, test_probes_idx, [],"probeToOneHot", "_test")


def save_subset_data(original, row_indices, columns_indices, original_name, suffix_name):
    index=False
    if len(columns_indices) > 1:
        if 0 in columns_indices:
            columns_indices.remove(0)
        col_indices_incl_row_names = [0]
        col_indices_incl_row_names.extend(columns_indices)
    if len(row_indices) < 1:
        subset = original.iloc[:, col_indices_incl_row_names]
    elif len(columns_indices) < 1:
        subset = original.iloc[row_indices, :]
    else:
        subset = original.iloc[row_indices, col_indices_incl_row_names]
    if original_name == "df_E":
        subset = subset.transpose()
        # renaming columns
        subset.columns = subset.iloc[0]
        subset = subset.iloc[1:, :]
        index=True
    subset.to_csv('../res/' + original_name + suffix_name+'.csv', index=index)


def createSmallerSizedFileUsingLessDecimals(filenameOrigin='TCGA_CH3_final', filenameDestSuffix='_lessDecimals', numDecimals=3):
    df = pd.read_csv('../res/'+filenameOrigin+'.csv', index_col=None)
    print(df.head())

    for col in df.columns:
        if col == 'Probe':
            continue
        df[col] = df[col].astype(float).round(numDecimals)
    print(df.head())
    df.to_csv('../res/'+filenameOrigin+filenameDestSuffix+'.csv', index=False)


def combineDataSets(pathToFile1, pathToFile2, fname):
    """
    assumes that index is first column
    :param pathToFile1:
    :param pathToFile2:
    :return:
    """
    data_1 = pd.read_csv(pathToFile1)
    data_2 = pd.read_csv(pathToFile2)
    combined = data_1.merge(data_2, on=data_1.columns[0], how='inner')
    print(combined.tail())
    print(len(combined))
    combined.to_csv('../res/'+ fname + '.csv', index=False)


def create_data_for_task1_task2_comparison(full_path_proximity_data_dir, full_path_random_data_dir):
    probes_to_remove = []
    probes_to_keep = []

    for fname in [Conf.filename_labels, Conf.filename_img, Conf.filename_dist]:
        df_proximity = pd.read_csv(full_path_proximity_data_dir+fname, index_col=False, usecols=[0]) #need only the probes column from the proximity data
        if len(probes_to_remove) < 1:
            probes_to_remove = list(df_proximity.iloc[:,0]) # remove any probes that are contained in the proximity dataset
        df_rand = pd.read_csv(full_path_random_data_dir+fname, index_col=False, usecols=[0])
        df_rand_test_probes_idx = load_obj('test_probes_idx')
        # df_rand = df_rand.iloc[df_rand_test_probes_idx,:] # keep only probes from test set for task1 (random CpGs)
        for idx, val in df_rand.iloc[:, 0].iteritems():
            if idx in df_rand_test_probes_idx and val not in probes_to_remove:
                probes_to_keep.append(idx)
        save_obj(probes_to_keep,"test_probes_comp_task1_task2")

        # if len(probes_to_remove) < 1:
        #     probes_to_remove = list(df_proximity.iloc[:,0]) # remove any probes that are contained in the proximity dataset
        #     probes_to_keep = [probe for probe in df_rand.iloc[:,0] if probe not in probes_to_remove]
        # df_rand_no_proximity = df_rand.loc[df_rand.iloc[:,0].isin(probes_to_keep), :]
        # df_rand_no_proximity.to_csv('../out/'+fname.split('.csv')[0]+'_no_proximity.csv', index=False)



def createMethylWithHighestVariance(n=50000):
    df = pd.read_csv(Conf.dfMethyl, index_col=None)
    df['var'] = df.var(1)
    df_nlargest = df.nlargest(n, 'var')
    df_nlargest.drop('var', axis=1, inplace=True)
    print(df_nlargest.tail())
    df_nlargest.to_csv('../res/' + Conf.dfMethylName + '_nlargest.csv', index=None)

# def verifyLargeFileMatches():
#     with open('../res/probeToOneHotMtrx_400_1.csv', 'r') as f:
#         counter = 0
#         for row in f:
#             print(row[0])
#             counter +=1
#             if counter >5:
#                 break

# createListOfProbesInData()

def getInputCpGvaluesOverSubjects(sample):
    CH3_original = pd.read_csv('../res/combined_CH3.csv', index_col=None)
    cols = [c for c in CH3_original.columns]
    cols[0] = 'Probe'
    CH3_original.columns = cols
    if sample:
        randCpGdistancesData = pd.read_csv('../res/distancesCpGToCpG_sample.csv', index_col=None)
    else:
        randCpGdistancesData = pd.read_csv('../res/distancesCpGToCpG.csv', index_col=None)
    relevantProbes = randCpGdistancesData.columns[1:]
    relevantData = CH3_original.loc[CH3_original['Probe'].isin(relevantProbes)]
    relevantData.sort_values(by='Probe', inplace=True)
    relevantData = relevantData.reset_index(drop=True)
    # match distances columns (new cpgs) to rows in methyl (relevantData) - for this, have to sort it and re-save
    randCpGdistancesData.sort_index(axis=1, inplace=True)
    print(relevantData.head())
    print(randCpGdistancesData.head())
    # asserting over subjects and CpGs
    TCGA_E = pd.read_csv('../res/TCGA_E_final_transposed.csv', index_col=None)
    relevantData.sort_index(axis=1, inplace=True)
    assert (relevantData.columns[1:] == TCGA_E['Unnamed: 0']).all()
    assert (relevantData['Probe'] == randCpGdistancesData.columns[1:]).all()
    relevantData = relevantData.transpose()
    # fixing columns (which now have a funny added index to them)
    relevantData.columns = relevantData.iloc[0, :]
    relevantData = relevantData.drop(relevantData.index[0])
    if sample:
        relevantData.to_csv('../res/CH3_inputCpGs_sample.csv', index=None)
        randCpGdistancesData.to_csv('../res/distancesCpGToCpG_sample.csv', index=None)
    else:
        relevantData.to_csv('../res/CH3_inputCpGs.csv', index=None)
        randCpGdistancesData.to_csv('../res/distancesCpGToCpG.csv', index=None)
    print("Final data:\n")
    print(relevantData.head())
    print(randCpGdistancesData.head())

# def read_rows_from_large_file_to_dataframe(filepath, rowIDs):
#     data = []
#     rowIDs.sort()
#     rowIDsIter = iter(rowIDs)
#     nextRowID = next(rowIDsIter)
#     counter = 0
#     with open(filepath) as f:
#         for line in f:
#             if counter==nextRowID:
#                 line = line.replace('\n', '')
#                 line = line.split(',')
#                 data.append(line)
#                 if nextRowID == rowIDs[-1]:
#                     break
#                 nextRowID = next(rowIDsIter)
#             counter += 1
#     df = pd.DataFrame(data)
#     df.columns = df.iloc[0,:]
#     df = df.drop(axis=0, index=0)
#     return df
#
# def generate_samples(filepath, nrows_sample, nrows_original):
#     rowsToRead = [0]
#     a = [i for i in range(1,nrows_original)]
#     a = random.run_example(a, nrows_sample)
#     rowsToRead.extend(a)
#     df = read_rows_from_large_file_to_dataframe(filepath, rowsToRead)
#     print(df)
#     return df



if __name__ == "__main__":

    # createMethylWithHighestVariance()

    #e = pd.read_csv('../res/TCGA_E_final.csv', index_col=None)
    # distances = pd.read_csv('../res/distances.csv', nrows=1)
    # cols = distances.columns[1:]
    # genesToRemove = [g for g in cols if str(g) not in np.array(e.iloc[:, 0])]
    # print(genesToRemove)
    # print(len(e.iloc[:,0]))
    # print(len([e.iloc[i, 0] for i in range(len(e.iloc[:, 0])) if e.iloc[i, 0] == cols[i]]))
    # distances = pd.read_csv('../res/distances.csv', index_col=None)
    # distances = distances.drop(genesToRemove, 1)
    # distances.to_csv('../res/distances_.csv', index=False)

    # distances = pd.read_csv('../res/distances.csv')
    # distances = distances.drop('Unnamed: 0', 1)
    # print(distances.head())
    # print(distances.columns)
    # distances = distances.to_csv('../res/distances__.csv', index=False)
    # methyl with largest var:

    # createExpressionFileWithHighVarGenes(0.05)
    # createFileWithHighVarItems(Conf.dfMethyl,Conf.dfMethyl,0.05)

    # all if NOT too large:
    # ------------------
    # surroundingSeqToOneHotMtrx(False)
    # probeToOneHotAll = probeToOneHotAsCSVcolumns(isSample=False, enhancersOnly=False)
    # probeToOneHotAll.to_csv(Conf.probeToOneHotPrefixAll+'.csv', index=False)
    # createDistanceMatrx(numProbes=1000, sort_probes=True, preSelectedProbes=False, useInverseDist=True, window_limit=4000)
    # Conf = ConfSample
    # createMatchingDataSets(probeToOneHotFileName=Conf.probeToOneHotPrefixAll+'.csv', perChromosome=False, withDistances=True) #TODO
    # transposeE(probeDataFileName=Conf.probeToOneHotPrefixAll+'.csv')

    # getInputCpGvaluesOverSubjects(run_example=False)

    # combine:
    # ---------------
    # combineDataSets('../res/expressi_lung.csv', '../res/combined_E_breast_prostate.csv', 'E')
    # combineDataSets('../res/methyl_lung.csv', '../res/combined_CH3_breast_prostate.csv', 'CH3')
    # combineDataSets('../res/expressi_prad.csv', '../res/expressi_brca.csv', 'expressi_combined')
    # combineDataSets('../res/methyl_prad.csv', '../res/methyl_brca.csv', 'methyl_combined')

    # createListOfProbesInData()
    # probeToOneHotAll = probeToOneHotAsCSVcolumns(isSample=False, enhancersOnly=False)
    # probeToOneHotAll.to_csv(Conf.probeToOneHotPrefixAll + '.csv', index=False)
    # createProbePositionsDict()
    # createDistanceMatrx(numProbes=-1, sort_probes=True, preSelectedProbes=False, useInverseDist=True, window_limit=20000)
    # createMatchingDataSets(probeToOneHotFileName=Conf.probeToOneHotPrefixAll+'.csv', perChromosome=False, withDistances=True)
    # transposeE(probeDataFileName=Conf.probeToOneHotPrefixAll+'.csv')

    # all if too large:
    #------------------
    # probeToOneHotAsCSVcolumnsTooLargeForPkl()
    # verifyLargeFileMatches()
    # createMatchingDataSets(probeToOneHotFileName='probeToOneHotLargeAll.csv', perChromosome=False)
    # transposeE(probeDataFileName=Conf.probeToOneHotMtrxFilePrefixAll+'.csv')

    # per chromosome
    #-----------------

    # surroundingSeqToOneHotMtrx(True)
    # for c in [i for i in Conf.chrArr[10:]]:
    #     print(c)
    #
    #     probeToOneHotFileName = Conf.probeToOneHotMtrxFilePrefixChr+str(c)+'.csv'
    #     probeToOneHotChr = probeToOneHotAsCSVcolumns(isSample=False, perChromosome=True, chromosome=c)
    #     probeToOneHotChr.to_csv(Conf.probeToOneHotMtrxFilePrefixChr+str(c)+'.csv', index=False)
    #     createMatchingDataSets(probeToOneHotFileName=probeToOneHotFileName,
    #                            chromosome=c,
    #                            perChromosome=True)
    #     transposeE(filename_CH3='TCGA_CH3_final_Chr_'+str(c),filename_E='TCGA_E_final', probeDataFileName=Conf.probeToOneHotMtrxFilePrefixChr+str(c)+'.csv')

    # Sample
    #-----------
    # createSampleData(withDistances=True)
    # createSingleDataFile()
    # createTrainTestFilesDeepWideReady(isSample=True)

    create_data_for_task1_task2_comparison('/home/alona/data/methyl_new/model_eccb_no_dist_limit_800_with_5bp_rand_100kCpG/res/', '/home/alona/data/methyl_new/model_10k_window_dist_to_closest_gene/res/')