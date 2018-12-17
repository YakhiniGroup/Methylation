import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from Bio import SeqIO
from helperMethods import *
import random
from sklearn import preprocessing
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import csv
from xlmhg import *
import math

# matplotlib.style.use('ggplot')

dir = '../out/postTrainingAnalysis/'


def read_headerless_csv_into_pandas_df(fname, columns=[], nrows=-1):

    data = []
    row_counter = -1
    with open(fname) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 1:
                continue
            row_counter += 1
            print(row)
            print(len(row))
            try:
                data.append(np.array(row).astype(float))
            except:
                row = [float(s.replace('\'','')) for s in row]
                data.append(np.array(row))
            if nrows != -1 and row_counter >= nrows:
                break

        df = pd.DataFrame(data)
        if len(columns) > 0:
            df.columns = columns
    return df




def saveAsGrayscaleImg(df, name):
    height, width = 50, 33  # in pixels
    spines = 'left', 'right', 'top', 'bottom'

    labels = ['label' + spine for spine in spines]

    tick_params = {spine: False for spine in spines}
    tick_params.update({label: False for label in labels})

    img = np.array(df)

    img *= 255. / img.max()#4096

    desired_width = 8  # in inches
    scale = desired_width / float(width)

    fig, ax = plt.subplots(1, 1, figsize=(desired_width, height * scale))
    img = ax.imshow(img, cmap=cm.Greys_r, interpolation='none')

    # remove spines
    for spine in spines:
        ax.spines[spine].set_visible(False)

    # hide ticks and labels
    ax.tick_params(**tick_params)

    # preview
    # plt.show()

    # save
    fig.savefig(dir+'figures/'+name+'.png', dpi=300)
    print("saved to "+dir+'figures/'+name+'.png')
    plt.close()

def brca_prad_differential_expression(gene_name="PARD3B"):
    from scipy import stats
    e_brca = pd.read_csv('expressi_brca.csv', index_col=False)
    e_prad = pd.read_csv('expressi_prad.csv', index_col=False)
    e_brca = e_brca.transpose()
    e_prad = e_prad.transpose()
    e_brca.columns = e_brca.iloc[0,:]
    e_prad.columns = e_prad.iloc[0, :]
    print(e_prad.columns)
    prad_gene = np.array(e_prad.loc[:, gene_name])[1:].astype(float)
    brca_gene = np.array(e_brca.loc[:, gene_name])[1:].astype(float)
    print(stats.ttest_ind(prad_gene, brca_gene))


# def plot_attention(sort_by, attention_type, attention_file_name='attention.csv', n_top=100, row_limit=1000):
#     # TODO: might need to remove cpGs, so remove those with max value less than some threshold, or intially take random rows so I can see visually what would be interesting
#     d = pd.read_csv('../out/'+attention_file_name, index_col=False, nrows=row_limit)
#     methylation_vals = d.iloc[:,-1]
#     d = d.iloc[:,:-1] # removing the methylation value column
#     columns_df = pd.read_csv('../res/'+"TCGA_E_final_transposed.csv", nrows=1)
#     columns = columns_df.columns[1:]
#     d.columns = columns[:len(d.columns)]
#     d = d.loc[:, (d > 0.1).any(axis=0)]
#
#     # d = d.reindex_axis(d.mean().sort_values(ascending=False).index, axis=1)
#     # d = d.reindex_axis(d.mean().sort_values(ascending=False).index, axis=1)
#
#     # d = d.iloc[:, :20] # take the first 10 genes
#     # d = d.sort_values(by=[c for c in d.columns], ascending=True)
#     new_data_frame = {}
#     if sort_by == "dist":
#         dist_raw = pd.read_csv('../out/dist_attn_raw_data.csv', index_col=False, nrows=row_limit)
#     for i in range(len(d.columns)):
#         if sort_by=="dist":
#             dist_raw_per_gene = abs(dist_raw.iloc[:, i])
#             if sum(dist_raw_per_gene) == 0:
#                 continue
#             attn_per_gene = d.iloc[:, i]
#             zipped = zip(dist_raw_per_gene, attn_per_gene)
#             zipped = sorted(zipped, reverse=True, key=lambda t: t[0]) #desc order
#             zipped_arr = np.array(zipped)
#             dist_attn_sorted_by_dist_raw = zipped_arr[:n_top, 1]
#             dist_raw_sorted = zipped_arr[:, 0]
#             # if sum(dist_raw_sorted) > 0:
#             new_data_frame[d.columns[i]] = dist_attn_sorted_by_dist_raw
#         elif sort_by == "methyl":
#             attn_per_gene = d.iloc[:, i]
#             zipped = zip(methylation_vals, attn_per_gene)
#             zipped = sorted(zipped, reverse=True, key=lambda t: t[0]) #desc order
#             dist_attn_sorted_by_methyl = np.array(zipped)[:n_top, 1]
#             new_data_frame[d.columns[i]] = dist_attn_sorted_by_methyl
#         else:
#             new_data_frame[d.columns[i]] = list(d.nlargest(n_top, d.columns[i]).iloc[:,i]) # get the attention column simply sorted by attention values from high to low
#     new_data_frame = pd.DataFrame(new_data_frame)
#     new_data_frame = new_data_frame.loc[:, (new_data_frame > 0.1).any(axis=0)]
#     new_data_frame.to_csv('../out/'+attention_type+'.csv',index=False)
#         # d.iloc[:,i] = d.nlargest(100, d.columns[i]).iloc[:,i]
#         # d = d.nlargest(100, d.columns[1:])
#         # d = d.nlargest(100, d.columns[2:])
#     # d = d.nlargest(100, d.columns[1])
#     # d = d.nlargest(100, d.columns[2])
#
#
#     # d = d.loc[(d > 0.1).any(axis=1), (d > 0.1).any(axis=0)]
#     # d = d.run_example(100) # run_example 100 CpGs
#     m_data =new_data_frame.values #last column contains the methylation level
#     # m_data = d.iloc[:, :-1].values
#     SIZE = 15
#     matplotlib.rc('font', size=SIZE)
#     matplotlib.rc('axes', titlesize=SIZE)
#     with sns.axes_style("white"):
#         fig, ax = plt.subplots()
#         ax = sns.heatmap(linewidth=0.90, cmap="YlGnBu", data = m_data)
#
#         # the size of A4 paper
#         fig.set_size_inches(11.7, 8.27)
#         # ax = sns.heatmap(m_data)
#         ax.set(xlabel='Genes', ylabel='CpG Loci')
#         # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#         #              ax.get_xticklabels() + ax.get_yticklabels()):
#         #     item.set_fontsize(20)
#         xTicks = tuple(new_data_frame.columns)
#         # xTicks = tuple(["abc" for i in range(len(d.columns))])
#         yTicks = tuple([""])
#         xTicks_positions = list(range(len(xTicks)))
#         xTicks_positions = [v+0.5 for v in xTicks_positions]
#         plt.xticks(xTicks_positions, xTicks, rotation=90)
#         plt.yticks(range(len(yTicks)), yTicks, rotation=0)#, va="center")
#
#         # plt.xlabel(xl)
#         # plt.ylabel(list(d.iloc[:, -1]))
#         plt.savefig('../out/'+attention_type+'.png')
#         plt.show()
#         plt.close()


from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from weblogolib import *
import matplotlib.pyplot as plt


def create_seq_logo(seqs, inname, outname):
    fin = open(inname)
    seqs = read_seq_data(fin)
    data = LogoData.from_seqs(seqs)
    options = LogoOptions()
    # options.resolution = 1600
    # options.title = "A Logo Title 5"
    format = LogoFormat(data, options)
    # format.resolution = 1600

    eps = eps_formatter(data, format)
    # eps = png_print_formatter(data, format)
    # png_print_formatter
    # import ghostscript
    img = Image.open(BytesIO(eps))
    # img.show()
    img.save('../out/'+outname+'.jpg')

def create_seq_logos_for_attn():
    for name1 in ['distattn', 'sequenceattn']:
        if name1 != 'sequenceattn':
            continue
        for name2 in ['attn', 'methyl']:
            if name1 == 'sequenceattn':
                # genes = ['CAMK2A', 'CRB1', 'GLUD1', 'HAND1', 'IMPA1', 'KRTAP21-2', 'NODAL', 'ODF1', 'SSTR4', 'ZNF621']
                genes = ['HAND1']
            else:
                genes = ['ALG8', 'AMZ2', 'BOD1', 'BRCC3', 'C3orf36', 'CACNG6', 'ELAVL1', 'ERAL1', 'ERI2', 'FOXO3B',
                         'GRPEL1', 'LCMT1', 'LMLN', 'MSH6', 'NRG3', 'RAB11FIP1', 'RABGAP1', 'RABIF', 'RFC1', 'SF3B4',
                         'SLC25A3', 'SNW1', 'TMEM106C', 'TTC9B', 'UAP1', 'UTP14A', 'VPS72', 'XRCC5']
            for gene in genes:
                list_CpG_seq_for_gene_as_sorted_from_attn_analysis('gene_to_sorted_seq_%s_sorted_by_%s' %(name1, name2), gene)
                for i in range(80):
                    try:
                        fin = open('../out/' + 'gene_to_sorted_seq_%s_sorted_by_%s_startPos_%d.txt' % (name1, name2, i))

                        seqs = read_seq_data(fin)
                        data = LogoData.from_seqs(seqs)
                        options = LogoOptions()
                        # options.resolution = 1600
                        # options.title = "A Logo Title 5"
                        format = LogoFormat(data, options)
                        # format.resolution = 1600

                        eps = eps_formatter(data, format)
                        # eps = png_print_formatter(data, format)
                        # png_print_formatter
                        # import ghostscript
                        img = Image.open(BytesIO(eps))
                        # img.show()
                        img.save('../out/seqLogos/%s_%s_%s_%d_TOP_5.jpg' %(name1, name2, gene, i))
                    except:
                        continue


def scatter_plot(x, y, name):
    plt.scatter(x, y, c='blue', alpha=0.08, s=8)
    plt.savefig('../out/plots/scatter_%s.png' % name)
    plt.xlim(0, 1)
    # plt.ylim(-2, 2)
    plt.xlabel('Predicted methylation')
    plt.ylabel('Actual methylation')
    plt.show()
    plt.close()


def plot_own_sequence_logo(ALL_SCORES):
    # ALL_SCORES = [[('A', 0),
    #                 ('C', 1),
    #                 ('G', 1),
    #                 ('T', 0.0)],
    #                [('C', 1),
    #                 ('G', 1),
    #                 ('A', 0),
    #                 ('T', 0)]]
    # NOTE: order MATTERS within each row - smallest probability should appear first

    fp = FontProperties(family="Arial", weight="bold")
    globscale = 1.35
    LETTERS = {"T": TextPath((-0.305, 0), "T", size=1, prop=fp),
               "G": TextPath((-0.384, 0), "G", size=1, prop=fp),
               "A": TextPath((-0.35, 0), "A", size=1, prop=fp),
               "C": TextPath((-0.366, 0), "C", size=1, prop=fp)}
    COLOR_SCHEME = {'G': 'orange',
                    'A': 'green',
                    'C': 'blue',
                    'T': 'red'}

    def letterAt(letter, x, y, yscale=1, ax=None):
        text = LETTERS[letter]

        t = mpl.transforms.Affine2D().scale(1 * globscale, yscale * globscale) + \
            mpl.transforms.Affine2D().translate(x, y) + ax.transData
        p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter], transform=t)
        if ax != None:
            ax.add_artist(p)
        return p

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))

    all_scores = ALL_SCORES
    x = 1
    maxi = 0
    for scores in all_scores:
        y = 0
        for base, score in scores:
            letterAt(base, x, y, score, ax)
            y += score
        x += 1
        maxi = max(maxi, y)

    # plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.ylabel("bits",weight='bold')
    plt.xticks(range(1, x),weight='bold')
    plt.yticks([0,1,2], weight='bold')
    plt.xlim((0.5, x-0.5))
    plt.ylim((0, 2))#maxi

    ax.tick_params(length=0.0)

    # rc('axes', linewidth=2)
    # rc('font', weight='bold')


    # ax.spines['bottom'].set_visible(False)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    plt.show()

# plot_own_sequence_logo()

def get_sequences_for_logo(df):

    matrix = df.iloc[:,:-1].values # without N
    # matrix = df.iloc[:, :].values  # with N
    matrix_probabilties = preprocessing.minmax_scale(matrix, [0, 1], axis=1)
    valid_nts = ['A', 'C', 'G', 'T']
    sequence = ''
    for row in matrix_probabilties: #letter
        min_prob = 1
        row = row / float(sum(row))
        prob_to_letters = [(row[0], 'A'), (row[1], 'C'), (row[2], 'G'), (row[3], 'T')]#, (row[4], np.random.choice(valid_nts, 1)[0])]
        prob_to_letters.sort()
        letter = prob_to_letters[0][1]
        rand_val = random.random()
        start_prob = 0
        for j in range(len(prob_to_letters)):
            if rand_val > start_prob and rand_val < start_prob+prob_to_letters[j][0]:
                letter = prob_to_letters[j][1]
                break
            else:
                start_prob = prob_to_letters[j][0]
        sequence+=letter
    print(sequence)

    return sequence


def convert_conv_filter_into_position_weight_matrix_for_own_seq_logo(df):
    matrix = df.iloc[:, :-1].values  # without N
    matrix_probabilties = preprocessing.minmax_scale(matrix, [0,1], axis=1) # scale 0,1
    matrix_probabilties = np.apply_along_axis(lambda row: row / sum(row), 1, matrix_probabilties) # position probability matrix
    func_pos_weight_matrix = lambda i: math.log2(i / float(0.25)) if i != 0 else -np.inf
    vectorized_func_pos_weight_matrix = np.vectorize(func_pos_weight_matrix)
    position_weight_matrix = vectorized_func_pos_weight_matrix(matrix_probabilties)  # turn into position weight matrix (see wiki in topic: https://en.wikipedia.org/wiki/Position_weight_matrix
    # creating the information content score per position:

    # the final height of each letter out of the information content score should be roportional to its frequency:
    # IC is per position overall, not letter (will be in range (0,2) i.e. in bits), then we multiply it by each letter's frequency:
    all_scores = []
    for i in range(len(matrix_probabilties)):
        IC_position = math.log2(4) + sum([x * math.log2(x) if x != 0 else 0 for x in matrix_probabilties[i]]) #according to berkley's: "Sequence logos for DNA sequence alignments" by Oliver Bembom
        # for each nt multiply its frequency by the IC_position to get its fraction of it.
        score_row = [('A', matrix_probabilties[i][0]*IC_position), ('C', matrix_probabilties[i][1]*IC_position), ('G', matrix_probabilties[i][2]*IC_position), ('T', matrix_probabilties[i][3]*IC_position)]
        score_row.sort(key=lambda item: item[1]) # order it as required by plot_own_logo function below
        all_scores.append(score_row)

    return all_scores


def list_CpG_seq_for_gene_as_sorted_from_attn_analysis(fname_geneToSeqDict, gene_name, seq_portion_range=20):
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.Alphabet import IUPAC

    geneToSeqDict = load_obj(fname_geneToSeqDict, '../out/')

    for i in range(int(800/seq_portion_range)):
        sequences = []
        with open('../out/'+fname_geneToSeqDict+'_startPos_'+str(i)+'.txt', 'w') as csvwriter:
            for g in geneToSeqDict[gene_name]:
                seq_portion = g[seq_portion_range*i:seq_portion_range*i+seq_portion_range]
                # seq_portion = g[300:500]
                csvwriter.write(seq_portion)
                csvwriter.write('\n')

                record = SeqRecord(Seq(seq_portion,
                                       IUPAC.Alphabet),
                                   id="", name="",
                                   description="vvv")
                sequences.append(record)
            csvwriter.close()
        with open("../out/"+gene_name+'_startPos_'+str(i)+".fasta", "w") as output_handle:
            SeqIO.write(sequences, output_handle, "fasta")
            # print(100*i)
            # print(100*i+100)
            # seq_portion = list(geneToSeqDict[gene_name])[seq_portion_range*i:seq_portion_range*i+seq_portion_range]
            # csvwriter.writelines('\n'.join(seq_portion))



def plot_attention(sort_by, attention_type_name, get_matching_sequences=True, get_matching_distances=False, attention_file_name='attention.csv', n_top=100, row_limit=1000):
    '''

    :param sort_by:
    :param attention_type_name: either 'sequence', 'dist' or 'gene'
    :param get_matching_sequences:
    :param attention_file_name:
    :param n_top:
    :param row_limit:
    :return:
    '''


    columns_df = pd.read_csv('../res/' +Conf.filename_other, nrows=1)
    gene_cols = columns_df.columns[1:]
    columns = list(gene_cols)
    columns.append('methylVal')
    # row_limit = row_limit*2 # each row is followed by an empty row due to how data is saved during testing
    d = pd.read_csv('../out/'+attention_file_name, nrows=row_limit, header=None)
    d.columns = columns
    d = d.dropna() # in case there are empty rows due to the format in which the file is being saved

    # d = read_headerless_csv_into_pandas_df('../out/'+attention_file_name, columns, nrows=row_limit)

    methylation_vals = d.iloc[:, -1]
    d = d.iloc[:, :-1] # removing the methylation value column


    # for col in d.columns:
    #     # print(col)
    #     d[col] = d[col].astype(float)
    attn_gene_filter_one_larger_than_0_01 = (d > 0.1).any(axis=0) #TODO: further generalize to function call for median
    d = d.loc[:, attn_gene_filter_one_larger_than_0_01]

    df_sorted_attn = {}
    gene_to_sorted_seq = {}
    gene_to_sorted_seq_random = {}
    gene_to_raw_dist_over_CpGs_rand = {}
    gene_to_raw_dist_over_CpGs_top_bottom = {}
    # if sort_by == "dist":

        # dist_raw = read_headerless_csv_into_pandas_df('../out/dist_attn_raw_data.csv', gene_cols, nrows=row_limit)
        # for col in dist_raw.columns:
        #     dist_raw[col] = dist_raw[col].astype(float)
    if get_matching_sequences:
        sequences = pd.read_csv('../out/' + attention_type_name + '_attn_raw_sequence_data.csv', nrows=row_limit,  header=None)
        # should not be filtered as filter is only relevant for genes, and here columns are position in one-hot matrix, nothing to do with genes
        # sequences = read_headerless_csv_into_pandas_df('../out/' + attention_type_name + '_attn_raw_sequence_data'+suffix+'.csv', nrows=row_limit)
        # sequences = sequences.iloc[1:,:]
        sequences = sequences.astype(int)
        sequences = [oneHotMtrxToSeq(np.array(sequences.iloc[i,:])) for i in range(len(sequences.iloc[:,0]))]
    if get_matching_distances:
        dist_raw = pd.read_csv('../out/dist_attn_raw_data.csv', index_col=False, nrows=row_limit, header=None)
        dist_raw = dist_raw.dropna()
        dist_raw = dist_raw.loc[:, list(attn_gene_filter_one_larger_than_0_01)]
    n_genes = 0
    for i in range(len(d.columns)):
        # if "SNW1" in d.columns[i]:
        #     print("SNW1 found")
        attn_per_gene = d.iloc[:, i]
        if get_matching_distances or sort_by=="dist":
            dist_raw_per_gene = abs(dist_raw.iloc[:, i]) # gathering all CpG distances for this gene
        if sort_by=="dist":
            print("here")
            if sum(dist_raw_per_gene) == 0:
                continue
            zipped = zip(dist_raw_per_gene, attn_per_gene, sequences, dist_raw_per_gene)
        elif sort_by == "methyl":
            zipped = zip(methylation_vals, attn_per_gene, sequences, dist_raw_per_gene)
        else:
            zipped = zip(attn_per_gene, attn_per_gene, sequences, dist_raw_per_gene)
            # new_data_frame[d.columns[i]] = list(d.nlargest(n_top, d.columns[i]).iloc[:,i]) # get the attention column simply sorted by attention values from high to low
        n_genes += 1
        zipped = sorted(zipped, reverse=True, key=lambda t: t[0])  # desc order
        zipped = np.array(zipped)
        attn_sorted = zipped[:n_top, 1].astype(float)
        sequences_sorted = zipped[:, 2]
        idx_for_sequences = np.random.choice(range(len(sequences_sorted)), 3000, replace=False)
        idx_for_sequences = sorted(idx_for_sequences)
        sequences_sorted_rand = sequences_sorted[idx_for_sequences]
        sequences_sorted_top = zipped[:20, 2]
        sequences_sorted_bottom = zipped[-100:, 2]
        if 'HAND1' in d.columns[i] or 'NODAL' in d.columns[i]:
            print(d.columns[i])
            print("top attn: \n", attn_sorted[:20])
            print("bottom attn: \n", attn_sorted[-20:])
            print("average attn in selected top: ", np.mean(attn_sorted[:20]))
            print("average attn in selected bottom: ", np.mean(attn_sorted[-20:]))
            print("average attn in bottom 100: ", np.mean(attn_sorted[-100:]))
        sequences_sorted = np.append(sequences_sorted_top, sequences_sorted_bottom)
        # sorting_criteria_sorted = np.array(zipped[:,0]) # e.g. saving sorted distances
        dist_raw_sorted = zipped[:, 3]
        idx_for_raw_dist = np.random.choice(range(len(dist_raw_sorted)), 10000, replace=False)
        idx_for_raw_dist = sorted(idx_for_raw_dist)
        raw_dist_top = zipped[:5000, 3]
        raw_dist_bottom = zipped[-5000:, 3]
        dist_top_bottom_sorted = np.append(raw_dist_top, raw_dist_bottom)

        gene_to_raw_dist_over_CpGs_rand[d.columns[i]] = dist_raw_sorted[idx_for_raw_dist]
        gene_to_raw_dist_over_CpGs_top_bottom[d.columns[i]] = dist_top_bottom_sorted
        df_sorted_attn[d.columns[i]] = attn_sorted
        gene_to_sorted_seq[d.columns[i]] = sequences_sorted
        gene_to_sorted_seq_random[d.columns[i]] = sequences_sorted_rand


    print("n_genes: ", n_genes)
    df_sorted_attn = pd.DataFrame(df_sorted_attn, index=None)
    df_sorted_attn = df_sorted_attn.loc[:, df_sorted_attn.median(axis=0)>0.1] #(df_sorted_attn > 0.1).any(axis=0)  #TODO: !!! median > 0.1 or any > 0.1
    df_sorted_attn.to_csv('../out/' + attention_type_name + 'attn_sorted_by_'+sort_by+'.csv', index=False)

    save_obj(gene_to_sorted_seq,"gene_to_sorted_seq_"+attention_type_name + "attn_sorted_by_"+sort_by, '../out/')
    save_obj(gene_to_sorted_seq_random, "gene_to_sorted_seq_rand_" + attention_type_name + "attn_sorted_by_" + sort_by, '../out/')
    save_obj(gene_to_raw_dist_over_CpGs_rand, "gene_to_sorted_sorting_criteria_"+attention_type_name + "attn_sorted_by_"+sort_by, '../out/')
    save_obj(gene_to_raw_dist_over_CpGs_top_bottom,
             "gene_to_raw_dist_over_CpGs_top_bottom_" + attention_type_name + "attn_sorted_by_" + sort_by, '../out/')
    # print(gene_to_sorted_seq)

    m_data =df_sorted_attn.values #last column contains the methylation level
    # m_data = d.iloc[:, :-1].values
    SIZE = 15
    matplotlib.rc('font', size=SIZE)
    matplotlib.rc('axes', titlesize=SIZE)
    with sns.axes_style("white"):
        fig, ax = plt.subplots()
        ax = sns.heatmap(linewidth=0.90, cmap="YlGnBu", data = m_data)

        # the size of A4 paper
        fig.set_size_inches(11.7, 8.27)
        ax.set(xlabel='Genes', ylabel='CpG Loci')
        # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        #              ax.get_xticklabels() + ax.get_yticklabels()):
        #     item.set_fontsize(20)
        xTicks = tuple(df_sorted_attn.columns)
        yTicks = tuple([""])
        xTicks_positions = list(range(len(xTicks)))
        xTicks_positions = [v+0.5 for v in xTicks_positions]
        plt.xticks(xTicks_positions, xTicks, rotation=90)
        plt.yticks(range(len(yTicks)), yTicks, rotation=0)#, va="center")

        # plt.xlabel(xl)
        # plt.ylabel(list(d.iloc[:, -1]))
        plt.tight_layout()
        plt.savefig('../out/' + attention_type_name + '.png', dpi=1000)
        plt.show()
        plt.close()


def plot_attention_from_file(filename):
    new_data_frame = pd.read_csv('../out/'+filename+'.csv', index_col=False)
    # new_data_frame = new_data_frame.loc[:, (new_data_frame > 0.1).any(axis=0)]
    # new_data_frame = new_data_frame.loc[:, new_data_frame.median(axis=0) > 0.1]
    m_data = new_data_frame.values  # last column contains the methylation level

    SIZE = 12
    matplotlib.rc('font', size=SIZE)
    matplotlib.rc('axes', titlesize=SIZE)
    with sns.axes_style("white"):
        fig, ax = plt.subplots()
        ax = sns.heatmap(linewidth=0.90, cmap="YlGnBu", data=m_data)

        # the size of A4 paper
        fig.set_size_inches(11.7, 8.27)
        # ax = sns.heatmap(m_data)
        ax.set(xlabel='Genes', ylabel='CpG Loci')

        # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        #              ax.get_xticklabels() + ax.get_yticklabels()):
        #     item.set_fontsize(20)
        # ax.get_xticklabels().set_fontsize(5)

        xTicks = tuple(new_data_frame.columns)
        # xTicks = tuple(["abc" for i in range(len(d.columns))])
        yTicks = tuple([""])
        xTicks_positions = list(range(len(xTicks)))
        xTicks_positions = [v + 0.5 for v in xTicks_positions]
        plt.xticks(xTicks_positions, xTicks, rotation=90, fontsize=5)
        plt.yticks(range(len(yTicks)), yTicks, rotation=0)  # , va="center")



        # plt.xlabel(xl)
        # plt.ylabel(list(d.iloc[:, -1]))
        plt.tight_layout()
        plt.savefig('../out/'+filename + '.png', dpi=1000)
        plt.show()
        plt.close()
# def generate_heatmap_from_table(column_names, row_names, )



def get_enrichment_mhg(sorted_list, turn_to_binary):
    sorted_list = np.array(sorted_list).astype(float)
    if turn_to_binary:
        sorted_list = abs(sorted_list)
        sorted_list = np.array([int(math.ceil(i)) for i in sorted_list])
        sorted_list = sorted_list.astype(int)
    return xlmhg_test(sorted_list)




if __name__ == '__main__':
    # ATTN PLOT
    # suffix = ''
    # plot_attention(sort_by="attn", attention_type_name='dist', attention_file_name='dist_attn'+suffix+'.csv', get_matching_distances=True, row_limit=1000)
    # plot_attention_from_file('distattn_sorted_by_attn')
    # plot_attention_from_file('seqAttnByAttnVal')

    # MHG
    # results={}
    # gene_to_sorted_dist_by_dist_attn = load_obj('gene_to_sorted_sorting_criteria_distattn_sorted_by_attn','../out/')
    # for gene in gene_to_sorted_dist_by_dist_attn.keys():
    #     mHG_result = get_enrichment_mhg(gene_to_sorted_dist_by_dist_attn[gene], True)
    #     pval = mHG_result[-1]
    #     cutoff = mHG_result[1]
    #     results[gene] = [pval, cutoff]
    # save_obj(results, "gene_to_mHG_result_for_proximity_enrichment_in_top_dist_attn_by_attn", '../out/')
    # gene_to_mhg = load_obj("gene_to_mHG_result_for_proximity_enrichment_in_top_dist_attn_by_attn", '../out/')
    # # sorting by p-val from lowest to highest:
    # n_genes = 0
    # n_significant_genes = 0
    # for key, value in sorted(gene_to_mhg.items(), key=lambda t: t[1]):
    #     n_genes += 1
    #     # if value[0]
    #     print("%s: %s" % (key, value))
    #     # TODO: if a gene doesn't exist in this list it means that

    # SCATTER PLOT
    # preds = np.array(load_obj("test_preds", '../out/'))
    # actual = np.array(load_obj("test_labels", '../out/'))
    # idx = np.arange(0,len(actual))
    # rand_indices = np.random.choice(idx, 3000, replace=False)
    # scatter_plot(preds[rand_indices], actual[rand_indices], "")

    #CNN WEIGHTS
    for i in range(64):
        print(i)

        # if i != 1:
        #     continue
        # activations
        # df = pd.read_csv(dir+'cnn_activation_%d.csv' % i, index_col=None)
        # saveAsGrayscaleImg(df,"conv_activ_%d" % i)

        # weights

        cnn_filter_type = 'cnn_filter'
        # cnn_filter_type = 'conv_attn_filter'
        # df = pd.read_csv(dir + 'cnn_filter_%d.csv' % i, index_col=None)
        df = pd.read_csv(dir + cnn_filter_type+'_%d.csv' % i, index_col=None)
        sequences = []
        # for k in range(5000):
        #     sequences.append(get_sequences_for_logo(df))   # !! TODO: take the output of this and put in: https://weblogo.berkeley.edu/logo.cgi
        conv_as_weight_matrix = convert_conv_filter_into_position_weight_matrix_for_own_seq_logo(df)
        plot_own_sequence_logo(conv_as_weight_matrix)

        # with open('../out/'+cnn_filter_type+'_sequences_for_logo_%d.csv' %i, 'w') as writer:
        #     writer.writelines('\n'.join(sequences))
        # create_seq_logo(sequences, '../out/'+cnn_filter_type+'_sequences_for_logo_%d.csv' %i, cnn_filter_type+'_%d' %i)


        # saveAsGrayscaleImg(df, "conv_attn_filter_%d" % i)


    # GENE WEIGHTS
    # df = pd.read_csv(dir+'genes_weights.csv', index_col=None)
    # img = np.array(df)
    # img.resize((100,100))
    # img *= 255. / img.max()
    # im = Image.fromarray(img)
    # im.show()
    #
    # saveAsGrayscaleImg(df, "genes_weights")

    # DIST WEIGHTS
    # df = pd.read_csv(dir+'dist_weights.csv', index_col=None)
    # # saveAsGrayscaleImg(df, "dist_weights")
    # img = np.array(df)
    # img.resize((100,100))
    # img *= 255. / img.max()
    # im = Image.fromarray(img)
    # im.show()

    # list_CpG_seq_for_gene_as_sorted_from_attn_analysis('gene_to_sorted_seq_sequenceattn_sorted_by_attn_20_vs_20', 'NODAL', 800)
    # list_CpG_seq_for_gene_as_sorted_from_attn_analysis('gene_to_sorted_seq_distattn_sorted_by_attn',
    #                                                    'FOXO3B', 800)
    # create_seq_logos_for_attn()