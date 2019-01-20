from sklearn import preprocessing
import itertools
from helperMethods import *
import random
import pandas as pd
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed


class Dataset():
    def __init__(self,
                 images,
                 other,
                 dist,
                 probes_idx,
                 subjects_idx,
                 labels,
                 labels_raw,
                 is_prediction=False,
                 expression_sample_names=None,
                 cpg_id_names=None,
                 dtype=dtypes.uint8,
                 reshape=False,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)

        self._num_examples = len(probes_idx)*len(subjects_idx)
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:  # irrelevant for our task (legacy from images with 3 channels)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
        # converting to black and white?
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._other = other
        self._dist = dist
        self._labels_idx = None
        self._labels = labels
        self._labels_raw = labels_raw
        # self._raw_labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._probes_idx = probes_idx
        self._subjects_idx = subjects_idx
        self.tuples_generator = None
        self.is_prediction = is_prediction
        self.expression_sample_names = expression_sample_names
        self.cpg_id_names = cpg_id_names

    @property
    def images(self):
        return self._images

    @property
    def other(self):
        return self._other

    @property
    def dist(self):
        return self._dist

    @property
    def labels_idx(self):
        return self._labels_idx

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def labels(self):
        return self._labels

    @property
    def labels_raw(self):
        return self._labels_raw

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def random_combination(self, iterable):
        "Random selection from itertools.combinations(iterable, r)"
        pool = tuple(iterable)
        n = len(pool)
        indices = sorted(random.sample(range(n), n))
        return tuple(pool[i] for i in indices)

    def create_tuples_generator(self):
        mylist = itertools.product(self._probes_idx, self._subjects_idx)
        pool = tuple(mylist)
        n = len(pool)
        if self.is_prediction:
            indices = range(n)
        else:
            indices = np.random.choice(range(n), n, replace=False)

        for i in indices:
            yield tuple(pool[i])

    @staticmethod
    def next_tuples_idx_batch(labels_generator, batch_size):
        generator_counter = 0
        labels_idx = []
        for i in labels_generator:
            labels_idx.append(i)
            generator_counter += 1
            if generator_counter == batch_size:
                # print(np.array(labels_idx))
                return np.array(labels_idx)


    def next_batch(self, batch_size, shuffle=True, testing=False):
        # print("getting next batch (first one per epoch includes a shuffle in the generator so will take longer")
        """Return the next `batch_size` examples from this data set."""
        if self.is_prediction:
            shuffle = False
            labels_rest_part = None
            labels_rest_part_raw = None
            labels_new_part = None
            labels_new_part_raw = None
            labels_ = None
            labels_raw_ = None

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0: # won't shuffle if is_prediction
            if shuffle:
                np.random.shuffle(self._subjects_idx)
                np.random.shuffle(self._probes_idx)
            # Get the shuffled full data
            self.tuples_generator = self.create_tuples_generator() # required also for predictions as it creates the tuples of (CpG, sample_ID) (e.g. [probe_123, subject_1])
            # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            if rest_num_examples != 0: # we still have some remaining in previous epoch
                # Get the rest in this epoch
                tuples_idx_rest_part = self.next_tuples_idx_batch(self.tuples_generator,
                                                                  batch_size=self._num_examples - start)
                if not self.is_prediction:
                    labels_rest_part = [self.labels[i, j] for [i, j] in tuples_idx_rest_part]
                    labels_rest_part_raw = [self.labels_raw[i, j] for [i, j] in tuples_idx_rest_part]
                cpg_id_names_rest_part = [self.cpg_id_names[i] for [i, _] in tuples_idx_rest_part]
                expression_sample_names_rest_part = [self.expression_sample_names[j] for [_, j] in tuples_idx_rest_part]
                sequences_rest_part = [self.images[i] for [i, _] in tuples_idx_rest_part]
                expression_rest_part = [self.other[j] for [_, j] in tuples_idx_rest_part]
                dist_rest_part = [self.dist[i] for [i, _] in tuples_idx_rest_part]

            # Shuffle the full data (for the new part that we will take)
            if shuffle:
                np.random.shuffle(self._subjects_idx)
                np.random.shuffle(self._probes_idx)
                # Get the shuffled full data
                self.tuples_generator = self.create_tuples_generator()

            if testing or self.is_prediction:
                # don't start a new epoch, we're done testing/predicting
                return sequences_rest_part, expression_rest_part, dist_rest_part, labels_rest_part, labels_rest_part_raw,\
                       cpg_id_names_rest_part, expression_sample_names_rest_part
            # Start next epoch if not in prediction mode:
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            self.tuples_generator = self.create_tuples_generator()
            tuples_idx_new_part = self.next_tuples_idx_batch(self.tuples_generator, batch_size=end - start)
            sequence_new_part = [self.images[i] for [i, _] in tuples_idx_new_part]
            expression_new_part = [self.other[j] for [_, j] in tuples_idx_new_part]
            dist_new_part = [self.dist[i] for [i, _] in tuples_idx_new_part]
            cpg_id_names_new_part = [self.cpg_id_names[i] for [i, _] in tuples_idx_new_part]
            expression_sample_names_new_part = [self.expression_sample_names[j] for [_, j] in tuples_idx_new_part]
            if not self.is_prediction:
                labels_new_part = [self.labels[i, j] for [i, j] in tuples_idx_new_part]
                labels_new_part_raw = [self.labels_raw[i, j] for [i, j] in tuples_idx_new_part]
            if rest_num_examples == 0:
                return sequence_new_part, expression_new_part, dist_new_part, labels_new_part, labels_new_part_raw, cpg_id_names_new_part, expression_sample_names_new_part
            else:
                return np.concatenate((sequences_rest_part, sequence_new_part), axis=0), \
                   np.concatenate((expression_rest_part, expression_new_part), axis=0), \
                       np.concatenate((dist_rest_part, dist_new_part), axis=0), \
                       np.concatenate((labels_rest_part, labels_new_part), axis=0), \
                       np.concatenate((labels_rest_part_raw, labels_new_part_raw), axis=0), \
                       np.concatenate((cpg_id_names_rest_part, cpg_id_names_new_part), axis=0), \
                       np.concatenate((expression_sample_names_rest_part, expression_sample_names_new_part), axis=0)

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            tuples_idx = self.next_tuples_idx_batch(self.tuples_generator, batch_size=end - start)
            sequence_ = [self.images[i] for [i, _] in tuples_idx]
            expression_ = [self.other[j] for [_, j] in tuples_idx]
            dist_ = [self.dist[i] for [i, _] in tuples_idx]
            cpg_id_names_ = [self.cpg_id_names[i] for [i, _] in tuples_idx]
            expression_sample_names_ = [self.expression_sample_names[j] for [_, j] in tuples_idx]
            if not self.is_prediction:
                labels_ = [self.labels[i, j] for [i, j] in tuples_idx]
                labels_raw_ = [self.labels_raw[i, j] for [i, j] in tuples_idx]
            return sequence_, expression_, dist_, labels_, labels_raw_, cpg_id_names_, expression_sample_names_


def _get_bins_avg(bins, n_quant):
    mean_bins_array = []
    for i in range(len(bins)):
        if i != 0:
            mean = (bins[i] + bins[i-1]) / float(2)
            mean_bins_array.append(mean)
    return mean_bins_array


def _get_max_diff_bins(bins, n_quant):
    err_array = []
    for i in range(len(bins)):
        if i == 0:
            err = bins[i+1]-bins[i]
        elif i == len(bins)-1:
            err = bins[i]-bins[i-1]
        else:
            err = max(bins[i]-bins[i-1], bins[i+1]-bins[i])
        err_array.append(err)
    return err_array


def read_data_sets(filename_sequence,
                   filename_expression,
                   filename_dist,
                   filename_labels,
                   directory='../res/',
                   is_prediction=False,
                   # one_hot=False,
                   # dtype=dtypes.uint8,
                   dense_labels_by_quantile=False,
                   train_portion_subjects=0.75,
                   train_portion_probes=0.75,
                   validation_portion_subjects=0.25,
                   validation_portion_probes=0.25,
                   seed=None,
                   load_model_ID=3):


    print("reading data")

    data_sequence_full = pd.read_csv(directory + filename_sequence)
    data_expression_full = pd.read_csv(directory + filename_expression)
    data_dist_full = pd.read_csv(directory + filename_dist)
    expression_sample_names = data_expression_full.iloc[:, 0]
    cpg_id_names = data_sequence_full.iloc[:, 0]
    if is_prediction:
        data_labels = None
    else:
        data_labels = pd.read_csv(directory + filename_labels)

    # z-score scaling "other" & "dist" data (gene expr. E)
    # std_scale = preprocessing.StandardScaler().fit(data_expression_full[data_expression_full.columns[1:]])
    # data_expression_full = std_scale.transform(data_expression_full[data_expression_full.columns[1:]])
    # std_scale_dist = preprocessing.StandardScaler().fit(data_dist_full[data_dist_full.columns[1:]])
    # data_dist_full= std_scale_dist.transform(data_dist_full[data_dist_full.columns[1:]])
    window = 2000

    def dist_window(x):
        if x == 0:
            return x
        elif abs(1 / float(x)) <= window:
            return 1*np.sign(x)  # acts as an amplifier for the gene expr. data - if the gene is within the above window - tf.multiply gets the +/- value of the gene exp (depends if it's before or after CpG). Otherwise - gets 0.
            # return x
        else:
            return 0

    def dist_window_buckets(x):
        # return x
        if x == 0:
            return x
        else:
            dist_in_bp = abs(1 / float(x))
            if dist_in_bp <= 2000:
                return np.sign(x)
                # return x
            elif dist_in_bp <= 4000:
                return 0.5 * np.sign(x)
            elif dist_in_bp <= 6000:
                return 0.5 ** 2 * np.sign(x)
            elif dist_in_bp <= 8000:
                return 0.5 ** 3 * np.sign(x)
            elif dist_in_bp <= 10000:
                return 0.5 ** 4 * np.sign(x)
            elif dist_in_bp <= 12000:
                return 0.5 ** 5 * np.sign(x)
            elif dist_in_bp <= 14000:
                return 0.5 ** 6 * np.sign(x)
            elif dist_in_bp <= 16000:
                return 0.5 ** 7 * np.sign(x)
            elif dist_in_bp <= 18000:
                return 0.5 ** 8 * np.sign(x)
            elif dist_in_bp <= 20000:
                return 0.5 ** 9 * np.sign(x)
            else:
                return 0

    def dist_window_for_regular_dist(x): #in case not inverse dist
        if x == 0:
            return x
        elif abs(x) <= window:
            return 1*np.sign(x)  #acts as an amplifier for the gene expr. data - if the gene is within the above window - tf.multiply gets the +/- value of the gene exp (depends if it's before or after CpG). Otherwise - gets 0.
            # return x
        else:
            return 0

    def drop_probes_with_no_pos_dist_in_window(data_img, data_labels, data_dist):
        probes_to_remove = []
        probes_kept = []
        tot_n_probes = 0
        for index, row in data_dist.iterrows():
            tot_n_probes += 1
            if 1 not in row.values:
                probes_to_remove.append(index)
            else:
                probes_kept.append(index)
        data_img = data_img.drop(probes_to_remove)
        data_dist = data_dist.drop(probes_to_remove)
        if not is_prediction:
            data_labels = data_labels.drop(probes_to_remove)
        save_obj(probes_kept, "probes_kept_from_distance_window")
        return data_img, data_labels, data_dist

    if load_model_ID == 1:
        use_dist_window = True
    else:
        use_dist_window = False

    # scaling:
    data_expression_full = preprocessing.minmax_scale(data_expression_full[data_expression_full.columns[1:]], [10, 100], axis=0)  # this is to give distances more of an effect (even if there's 0 gene expression (will be 0.5 in practice) then if the distance is not 0 will get some value
    data_dist_ix = data_dist_full['Probe']
    if use_dist_window:
        data_dist_full = data_dist_full.iloc[:, 1:].applymap(lambda x: dist_window(x))
    else:
        data_dist_full = data_dist_full.iloc[:, 1:].applymap(lambda x: dist_window_buckets(x))

    data_dist_full.index = data_dist_ix  # adding back the index so we can remove probe with no positive distance in window
    data_sequence_full.index = data_sequence_full['Probe']
    if not is_prediction:
        data_labels.index = data_labels['Probe']

    if use_dist_window:
        data_sequence_full, data_labels, data_dist_full = drop_probes_with_no_pos_dist_in_window(data_sequence_full, data_labels, data_dist_full)

    # The other two are marked out here because the sacling above already returned them in an np.array format
    data_sequence_full = data_sequence_full.as_matrix()  # in the future replace .as_matrix with .values as it will be deprecated in the future
    # data_expression_full = data_expression_full.as_matrix()
    data_dist_full = data_dist_full.as_matrix()
    if not is_prediction:
        data_labels = data_labels.as_matrix()

    features_img = data_sequence_full[:, 1:]
    features_other = data_expression_full[:,:]
    features_dist = data_dist_full[:, :]
    if not is_prediction:
        data_labels = data_labels[:,1:]

    data_labels_raw = data_labels
    if dense_labels_by_quantile != False:
        data_labels, bins = pd.qcut(data_labels, dense_labels_by_quantile, labels=False, retbins=True)
    if is_prediction:
        num_probes = features_img.shape[0]  # num probes
        num_subjects = features_other.shape[0]  # num_subjects (file transposed!)

        # subjects idx train/test_ch3_e_blind/val
        train_subjects_idx = random.sample(range(num_subjects), int(round(num_subjects * train_portion_subjects)))
        test_subjects_idx = [i for i in range(num_subjects) if i not in train_subjects_idx]
        validation_subjects_size = int(round(len(train_subjects_idx) * validation_portion_subjects))
        validation_subjects_idx = train_subjects_idx[0:validation_subjects_size]
        train_subjects_idx = train_subjects_idx[validation_subjects_size:]

        # probes idx train/test_ch3_e_blind/val
        perm_num_probes = np.arange(num_probes)
        np.random.shuffle(perm_num_probes)
        # train_probes_idx = random.run_example(range(num_probes), int(round(num_probes * train_portion)))
        train_probes_idx = perm_num_probes[:int(round(num_probes * train_portion_probes))]
        test_probes_idx = perm_num_probes[int(round(num_probes * train_portion_probes)):]
        validation_probes_size = int(round(len(train_probes_idx) * validation_portion_probes))
        validation_probes_idx = train_probes_idx[0:validation_probes_size]
        train_probes_idx = train_probes_idx[validation_probes_size:]
    else:
        try:
            train_probes_idx = load_obj("train_probes_idx", directory)
            train_subjects_idx = load_obj("train_subjects_idx", directory)
            validation_probes_idx = load_obj("validation_probes_idx", directory)
            validation_subjects_idx = load_obj("validation_subjects_idx", directory)
            test_probes_idx = load_obj("test_probes_idx", directory)
            test_subjects_idx = load_obj("test_subjects_idx", directory)
        except:
            num_probes = features_img.shape[0] #num probes
            num_subjects = features_other.shape[0] # num_subjects (file transposed!)

            print("splitting subjects into train/test_ch3_e_blind/val")
            # subjects idx train/test_ch3_e_blind/val
            train_subjects_idx = random.sample(range(num_subjects), int(round(num_subjects * train_portion_subjects)))
            test_subjects_idx = [i for i in range(num_subjects) if i not in train_subjects_idx]
            validation_subjects_size = int(round(len(train_subjects_idx) * validation_portion_subjects))
            validation_subjects_idx = train_subjects_idx[0:validation_subjects_size]
            train_subjects_idx = train_subjects_idx[validation_subjects_size:]

            print("splitting probes into train/test_ch3_e_blind/val")
            # probes idx train/test_ch3_e_blind/val
            perm_num_probes = np.arange(num_probes)
            np.random.shuffle(perm_num_probes)
            # train_probes_idx = random.run_example(range(num_probes), int(round(num_probes * train_portion)))
            train_probes_idx = perm_num_probes[:int(round(num_probes * train_portion_probes))]
            test_probes_idx = perm_num_probes[int(round(num_probes * train_portion_probes)):]
            validation_probes_size = int(round(len(train_probes_idx) * validation_portion_probes))
            validation_probes_idx = train_probes_idx[0:validation_probes_size]
            train_probes_idx = train_probes_idx[validation_probes_size:]

            save_obj(train_probes_idx, "train_probes_idx", directory)
            save_obj(train_subjects_idx, "train_subjects_idx", directory)
            save_obj(validation_probes_idx, "validation_probes_idx", directory)
            save_obj(validation_subjects_idx, "validation_subjects_idx", directory)
            save_obj(test_probes_idx, "test_probes_idx", directory)
            save_obj(test_subjects_idx, "test_subjects_idx", directory)

    train = Dataset(features_img, features_other, features_dist,
                    train_probes_idx, train_subjects_idx, labels=data_labels, labels_raw=data_labels_raw,
                    is_prediction=is_prediction, expression_sample_names=expression_sample_names, cpg_id_names=cpg_id_names)
    # fully blind val and test:
    validation_ch3_e_blind = Dataset(features_img, features_other, features_dist,
                                     validation_probes_idx, validation_subjects_idx, labels=data_labels, labels_raw=data_labels_raw,
                                     is_prediction=is_prediction, expression_sample_names=expression_sample_names, cpg_id_names=cpg_id_names)
    test_ch3_e_blind = Dataset(features_img, features_other, features_dist,
                    test_probes_idx, test_subjects_idx, labels=data_labels, labels_raw=data_labels_raw,
                               is_prediction=is_prediction, expression_sample_names=expression_sample_names, cpg_id_names=cpg_id_names)
    # blind ch3 (ch3 = val/test idx, e = train idx)
    validation_ch3_blind = Dataset(features_img, features_other, features_dist,
                                     validation_probes_idx, train_subjects_idx, labels=data_labels, labels_raw=data_labels_raw, is_prediction=is_prediction,
                                   expression_sample_names=expression_sample_names, cpg_id_names=cpg_id_names)
    test_ch3_blind = Dataset(features_img, features_other, features_dist,
                            test_probes_idx, train_subjects_idx, labels=data_labels, labels_raw=data_labels_raw, is_prediction=is_prediction,
                             expression_sample_names=expression_sample_names, cpg_id_names=cpg_id_names)
    # blind e (ch3 = train idx, e = val/test idx)
    validation_e_blind = Dataset(features_img, features_other, features_dist,
                                     train_probes_idx, validation_subjects_idx, labels=data_labels, labels_raw=data_labels_raw,
                                 is_prediction=is_prediction, expression_sample_names=expression_sample_names, cpg_id_names=cpg_id_names)
    test_e_blind = Dataset(features_img, features_other, features_dist,
                            train_probes_idx, test_subjects_idx, labels=data_labels, labels_raw=data_labels_raw,
                           is_prediction=is_prediction, expression_sample_names=expression_sample_names, cpg_id_names=cpg_id_names)

    return train, validation_ch3_e_blind, test_ch3_e_blind, validation_ch3_blind, test_ch3_blind, validation_e_blind, test_e_blind


