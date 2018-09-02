class Conf():

    dir_hg19 = '../res/hg19/'
    checkpoint_dir = ''
    numSurrounding = 400 #per side of CpG i.e. total is x2
    chrArr = [str(i) for i in range(1,23)]
    chrArr.extend(['X','Y'])
    suffix = ''

    ### YOUR SETTINGS - START ### TODO

    filename_sequence = 'probeToOneHotAll.csv'
    filename_expression = 'TCGA_E_final_transposed.csv'
    filename_dist = 'distances.csv'
    filename_labels = 'TCGA_CH3_final.csv'  # lessDecimals.csv'

    validation_portion_subjects = 0.1
    validation_portion_probes = 0.1
    train_portion_probes = 0.7

    ### YOUR SETTINGS - END ###

    #TODO: below conf files are intended for use ONLY in dataProcessor, not in model code
    probeToSurroundingSeqFilePrefixAll = '../res/probe_to_surroundingSeq_'
    probeToSurroundingSeqFilePrefixChr = '../res/interims/probe_to_surroundingSeq_'  # '../res/interims/probe_to_surroundingSeq_'
    probeToOneHotMtrxFilePrefixChr = '../res/probeToOneHotMtrx_'
    probeToOneHotMtrxFilePrefixAll = '../res/probeToOneHotMtrxAll'+str(suffix)# + str(numSurrounding)
    probeToOneHotPrefixAll = '../res/probeToOneHotAll'+str(suffix)# + str(numSurrounding)
    probeToOneHotPrefixChr = '../res/probeToOneHotChr_'+str(suffix)# + str(numSurrounding)
    numBases = 5
    # dfOriginalMethylCancer = '../res/methyl_brca.csv'#'../res/original_data_from_R/TCGA_CH3.csv'
    # dfOriginalExpressionCancer = '../res/expressi_brca.csv' #original_data_from_R/TCGA_E.csv'
    dfDistances = '../res/distances.csv'
    dfMethylName = 'combined_CH3'
    dfMethyl = '../res/methyl_brca.csv'
    dfExpression = '../res/expressi_brca.csv'
    # dfExpression = '../res/expressi_combined.csv'
    # dfMethyl = '../res/methyl_combined.csv'

    numSampleInputCpgs = 4
    numInputCpgs = 5000

    epochs = 2
    batch_size = 300
    num_steps = 50000


class ConfSample(Conf):

    numSurrounding = 400 #per side
    suffix = ''

    filename_sequence = 'probeToOneHotAll_sample_mini.csv'
    filename_expression = 'e_sample_mini.csv'
    filename_dist = 'd_sample_mini.csv'
    filename_labels = 'ch3_sample_mini.csv'

    # filename_img = 'probeToOneHotAll_sample.csv'
    # filename_other = 'TCGA_E_final_transposed_sample.csv'
    # filename_dist = 'distances_sample.csv'
    # filename_labels = 'TCGA_CH3_final_sample.csv'
    #
    # filename_img = 'probeToOneHotAll_sample_comb.csv'
    # filename_other = 'TCGA_E_final_transposed_sample_comb.csv'
    # filename_dist = 'distances_sample_comb.csv'
    # filename_labels = 'TCGA_CH3_final_sample_comb.csv'

    # filename_sequence = 'probeToOneHotAll_sample_withDist_10k_closest_gene.csv'
    # filename_expression = 'TCGA_E_sample_withDist_10k_closest_gene.csv'
    # filename_dist = 'distances_sample_withDist_10k_closest_gene.csv'
    # filename_labels = 'TCGA_CH3_sample_withDist_10k_closest_gene.csv'
    #
    # filename_sequence = 'probeToOneHotAll_sample_mini.csv'
    # filename_expression = 'TCGA_E_sample_mini.csv'
    # filename_dist = 'distances_sample_mini.csv'
    # filename_labels = 'TCGA_CH3_sample_mini.csv'



    validation_portion_subjects = 0.1
    validation_portion_probes = 0.2
    train_portion_probes = 0.5

    probeToOneHotPrefixAll = '../res/probeToOneHotAll_sample' + str(suffix)  # + str(numSurrounding)
    numBases = 5 #4
    # dfDistances = '../res/distances_sample.csv'
    # dfDistances = '../res/distances_sample_comb.csv'
    dfDistances = '../res/distances_sample_withDist_10k_closest_gene.csv'

    numSampleInputCpgs = 4

    epochs = 50
    batch_size = 10
    # num_steps = 1000
