import os
from helperMethods import *
from scipy.stats import spearmanr
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from conf import Conf, ConfSample
from collections import deque
import pandas as pd
import math
import time


class Model():
    def __init__(self, sample, modelID,
                 lr, multiplyNumUnitsBy, n_quant, neighborAlpha, c,
                 ff_h1, ff_h2, conv_f, conv_p, conv_s, connected_h1, connected_h2, regularization_scale,
                 train, validation, test, validation_ch3_blind, test_ch3_blind, validation_e_blind, test_e_blind,
                 n_genes,
                 ff_n_hidden, connected_n_hidden, optimizer, loss, load_model, is_prediction, test_time, save_models,
                 save_weights):

        # perChromosome = False

        self.Conf = Conf
        if sample:
            self.Conf = ConfSample

        self.load_model = load_model
        self.is_prediction = is_prediction
        self.test_time = test_time
        self.save_models = save_models
        self.save_weights = save_weights
        self.n_genes = n_genes

        self.sample = sample
        self.modelID = modelID
        self.lr = lr
        self.multiplyNumUnitsBy = multiplyNumUnitsBy
        self.n_quant = n_quant
        self.neighborAlpha = neighborAlpha
        self.c = c
        self.ff_h1 = ff_h1
        self.ff_h2 = ff_h2
        self.conv_f = conv_f
        self.conv_p = conv_p
        self.conv_s = conv_s
        self.connected_h1 = connected_h1
        self.connected_h2 = connected_h2
        self.regularization_scale = regularization_scale
        self.optimizer_func = optimizer
        self.loss_func = loss
        self.ff_num_hidden = ff_n_hidden
        self.connected_num_hidden = connected_n_hidden

        self.train = train
        self.validation = validation
        self.test = test
        self.validation_ch3_blind = validation_ch3_blind
        self.test_ch3_blind = test_ch3_blind
        self.validation_e_blind = validation_e_blind
        self.test_e_blind = test_e_blind

    def build(self, with_autoencoder):

        self.regularizer = tf.contrib.layers.l1_regularizer(scale=self.regularization_scale)
        self.init_xavier = tf.contrib.layers.xavier_initializer(uniform=True)
        self.activation = tf.nn.elu

        def save_single_weights(sess, tf_weights, name):
            weights = sess.run(tf_weights)
            weights = pd.DataFrame(weights)
            weights.to_csv('../out/postTrainingAnalysis/' + name + '_weights.csv', index=None, columns=None)

        # -- FC --  # takes gene expression data (~20K)
        units = np.array([int(self.ff_h1 * (0.9 ** i)) for i in range(self.ff_num_hidden)])
        # -- CNN -- # takes ambient seq data (~800)
        dimx = self.Conf.numSurrounding * 2
        dimy = self.Conf.numBases
        kernel_size = [(11, self.Conf.numBases)]
        filters = [self.conv_f]
        numConvs = 1
        pool_size_layer1 = (self.conv_p, 1)
        pool_strides_layer1 = (self.conv_s, 1)
        pool_size_layer2 = (2, 1)
        pool_strides_layer2 = (2, 1)
        pooling = tf.layers.average_pooling2d
        activationCNN = tf.nn.elu
        # -- CONNECTED --
        num_connected_layers = self.connected_num_hidden  # global_num_hidden
        connected_units = np.array([int(self.connected_h1 * (0.9 ** i)) for i in range(num_connected_layers)])

        # TESNROFLOW GRAPH COMPONENTS
        # -----------------
        # -- INPUT --
        ff_trainable = True
        reuse_var = tf.AUTO_REUSE

        with tf.name_scope('input_gene_expression') as scope:
            inputs_ff = tf.placeholder(tf.float32, (None, self.n_genes))
        with tf.name_scope('input_dist') as scope:
            inputs_dist = tf.placeholder(tf.float32, (None, self.n_genes))
        with tf.name_scope('input_ambient_seq') as scope:
            inputs_cnn = tf.placeholder(tf.float32, (None, dimx, dimy, 1))
            paddings = tf.constant([[0, 0], [5, 5], [0, 0], [0, 0]])
            padded_inputs_cnn = tf.pad(inputs_cnn, paddings, "CONSTANT")
        with tf.name_scope('labels_methyl_value') as scope:
            labels_ = tf.placeholder(tf.float32, (None, 1), name='labels')

        # -- FF --
        def encoder(inputs_ff):
            scope_name = 'gene_exp_encoder_decoder/'
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
                # with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                # encoder
                for i in range(self.ff_num_hidden):
                    name = 'dense_%d' % i
                    if i == 0:
                        layer_ff = tf.layers.dense(inputs=inputs_ff, units=units[i], activation=None, name=name,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(
                                                       uniform=True), kernel_regularizer=self.regularizer,
                                                   bias_initializer=self.init_xavier, trainable=ff_trainable,
                                                   reuse=tf.AUTO_REUSE)
                        self.tf_weights_genes = tf.get_default_graph().get_tensor_by_name(
                            scope_name + '/' + name + '/kernel:0')  # TODO: use tf.all_variables() to find kernel name
                    else:
                        layer_ff = tf.layers.dense(inputs=layer_ff__, units=units[i], activation=None, name=name,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(
                                                       uniform=True), kernel_regularizer=self.regularizer,
                                                   bias_initializer=self.init_xavier, trainable=ff_trainable,
                                                   reuse=tf.AUTO_REUSE)

                    layer_ff_ = tf.layers.batch_normalization(layer_ff, trainable=ff_trainable, reuse=reuse_var,
                                                              name="batch_norm_" + name)
                    layer_ff__ = self.activation(layer_ff_)
                    tf.summary.histogram(name, layer_ff_)
                    tf.summary.histogram(name + '/activation', layer_ff__)
            return layer_ff__

        scope_name = 'all_but_gene_exp/'

        def dist(inputs_dist):
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as var_scope:
                # dropout = None
                for i in range(self.ff_num_hidden):
                    name = 'dist_%d' % i
                    if i == 0:
                        layer_dist = tf.layers.dense(inputs=inputs_dist, units=units[i], activation=None, name=name,
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer(
                                                         uniform=True),
                                                     kernel_regularizer=None, bias_initializer=self.init_xavier)
                        # tf_weights_dist = tf.get_default_graph().get_tensor_by_name(scope_name +'/' +  name + '/kernel:0')
                    else:
                        layer_dist = tf.layers.dense(inputs=layer_dist__, units=units[i], activation=None, name=name,
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer(
                                                         uniform=True),
                                                     kernel_regularizer=None, bias_initializer=self.init_xavier)
                    layer_dist_ = tf.layers.batch_normalization(layer_dist, name=name)
                    layer_dist__ = self.activation(layer_dist_)
                    tf.summary.histogram(name, layer_dist_)
                    tf.summary.histogram(name + '/activation', layer_dist__)
            return layer_dist__

        def cnn(padded_inputs_cnn):
            # -- CNN --
            i = 0
            conv = None
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as var_scope:
                for i in range(numConvs):
                    name = 'conv_%d' % i
                    namePool = 'pool_%d' % i
                    if i == 0:
                        conv = tf.layers.conv2d(inputs=padded_inputs_cnn, filters=filters[i],
                                                kernel_size=kernel_size[i],
                                                activation=None, name=name, bias_initializer=self.init_xavier)
                        conv = tf.layers.batch_normalization(conv)
                        conv_activ1 = activationCNN(conv)
                        tf_weights_conv = tf.get_default_graph().get_tensor_by_name(
                            'all_but_gene_exp//all_but_gene_exp//conv_0/kernel:0')
                        pool = pooling(conv_activ1, pool_size=pool_size_layer1, strides=pool_strides_layer1,
                                       name='pool_' + str(i))
                    else:
                        conv = tf.layers.conv2d(inputs=pool, filters=filters[i], kernel_size=kernel_size[i],
                                                activation=None, name=name, bias_initializer=self.init_xavier)
                        conv = tf.layers.batch_normalization(conv, name=name)
                        conv_activ = activationCNN(conv)
                        pool = pooling(conv_activ, pool_size=pool_size_layer2, strides=pool_strides_layer2,
                                       name='pool_' + str(i))
                    tf.summary.histogram(name, conv)
                    tf.summary.histogram(namePool, pool)

                final_conv_flat = tf.reshape(pool, [-1, pool.shape[1] * pool.shape[2] * pool.shape[3]], name='flatten')
            return final_conv_flat, tf_weights_conv

        # # -- CNN Fully Connected --
        def cnn_fc(final_conv_flat):
            # scope_name = 'cnn_FC/'
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as var_scope:
                for i in range(3):
                    name = 'cnn_fc_%d' % i
                    if i == 0:
                        cnn_fc_layer = tf.layers.dense(inputs=final_conv_flat, units=connected_units[i],
                                                       activation=None, name=name,
                                                       kernel_regularizer=None, bias_initializer=self.init_xavier)
                    else:
                        cnn_fc_layer = tf.layers.dense(inputs=cnn_fc_layer__, units=connected_units[i], activation=None,
                                                       name=name,
                                                       kernel_regularizer=None, bias_initializer=self.init_xavier)
                    cnn_fc_layer_ = tf.layers.batch_normalization(cnn_fc_layer, name=name)
                    cnn_fc_layer__ = self.activation(cnn_fc_layer_)
                    tf.summary.histogram(name, cnn_fc_layer)
            return cnn_fc_layer__

        def dist_for_gene_attn(inputs_dist):
            # -- CNN --
            i = 0
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as var_scope:
                name = 'dist_gene_attn_%d' % i

                if self.load_model == 1:
                    inputs_dist_layer = tf.layers.dense(inputs=inputs_dist, units=int(self.n_genes / 2),
                                                        activation=None, name='fully_conn_gene_attn_dist',
                                                        kernel_regularizer=None, bias_initializer=self.init_xavier,
                                                        kernel_initializer=self.init_xavier)
                else:
                    inputs_dist_layer = tf.layers.dense(inputs=inputs_dist, units=int(self.n_genes / 6),
                                                        activation=None, name='fully_conn_gene_attn_dist',
                                                        kernel_regularizer=None, bias_initializer=self.init_xavier,
                                                        kernel_initializer=self.init_xavier)
                inputs_dist_layer = tf.layers.batch_normalization(inputs_dist_layer, name=name)
                if not self.load_model == 1:
                    inputs_dist_layer = self.activation(inputs_dist_layer)
                inputs_dist_layer2 = tf.layers.dense(inputs=inputs_dist_layer, units=self.n_genes, activation=None,
                                                     name='fully_conn_gene_attn_dist_2',
                                                     kernel_regularizer=None, bias_initializer=self.init_xavier,
                                                     kernel_initializer=self.init_xavier)
                inputs_dist_layer2 = tf.layers.batch_normalization(inputs_dist_layer2, name=name + "_2")
                if not self.load_model == 1:
                    inputs_dist_layer2 = self.activation(inputs_dist_layer2)
                inputs_dist_layer2 = tf.nn.softmax(inputs_dist_layer2)

            return inputs_dist_layer2

        def gene_exp_for_gene_attn(inputs_ff):
            i = 0
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as var_scope:
                name = 'geneExp_gene_attn_%d' % i

                inputs_geneExp_layer = tf.layers.dense(inputs=inputs_ff, units=int(self.n_genes / 6), activation=None,
                                                       name='fully_conn_gene_attn_geneExp',
                                                       kernel_regularizer=None, bias_initializer=self.init_xavier,
                                                       kernel_initializer=self.init_xavier)
                inputs_geneExp_layer = tf.layers.batch_normalization(inputs_geneExp_layer, name=name)
                inputs_geneExp_layer = self.activation(inputs_geneExp_layer)

                inputs_geneExp_layer2 = tf.layers.dense(inputs=inputs_geneExp_layer, units=self.n_genes,
                                                        activation=None,
                                                        name='fully_conn_gene_attn_geneExp_2',
                                                        kernel_regularizer=None, bias_initializer=self.init_xavier,
                                                        kernel_initializer=self.init_xavier)
                inputs_geneExp_layer2 = tf.layers.batch_normalization(inputs_geneExp_layer2, name=name + "_2")

                inputs_geneExp_layer2 = self.activation(inputs_geneExp_layer2)
                inputs_geneExp_layer2 = tf.nn.softmax(
                    inputs_geneExp_layer2)  # for probabilities #TODO: have the distance vectpr activate? add to loss?

            return inputs_geneExp_layer2

        def cnn_for_gene_attn(padded_inputs_cnn):
            # -- CNN --
            i = 0
            conv = None
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as var_scope:
                for i in range(numConvs):
                    name = 'conv_gene_attn_%d' % i
                    namePool = 'pool_gene_attn_%d' % i
                    if i == 0:
                        conv = tf.layers.conv2d(inputs=padded_inputs_cnn, filters=filters[i],
                                                kernel_size=kernel_size[i],
                                                activation=None, name=name, bias_initializer=self.init_xavier)
                        conv = tf.layers.batch_normalization(conv)
                        conv_activ1 = activationCNN(conv)
                        tf_weights_conv_attn = tf.get_default_graph().get_tensor_by_name(
                            'all_but_gene_exp//all_but_gene_exp//conv_gene_attn_0/kernel:0')
                        pool = pooling(conv_activ1, pool_size=pool_size_layer1, strides=pool_strides_layer1,
                                       name='pool_gene_attn_' + str(i))
                    else:
                        conv = tf.layers.conv2d(inputs=pool, filters=filters[i], kernel_size=kernel_size[i],
                                                activation=None, name=name, bias_initializer=self.init_xavier)
                        conv = tf.layers.batch_normalization(conv, name=name)
                        conv_activ = activationCNN(conv)
                        pool = pooling(conv_activ, pool_size=pool_size_layer2, strides=pool_strides_layer2,
                                       name='pool_gene_attn_' + str(i))
                    tf.summary.histogram(name, conv)
                    tf.summary.histogram(namePool, pool)

                final_conv_flat = tf.reshape(pool, [-1, pool.shape[1] * pool.shape[2] * pool.shape[3]],
                                             name='flatten_gene_attn_')
                cnn_fc_layer = tf.layers.dense(inputs=final_conv_flat, units=int(20), activation=None,
                                               name='fully_conn_gene_attn_conv',
                                               kernel_regularizer=None, bias_initializer=self.init_xavier,
                                               kernel_initializer=self.init_xavier)
                cnn_fc_layer = tf.layers.batch_normalization(cnn_fc_layer, name=name)
                cnn_fc_layer = self.activation(cnn_fc_layer)
                cnn_fc_layer2 = tf.layers.dense(inputs=cnn_fc_layer, units=self.n_genes, activation=None,
                                                name='fully_conn_gene_attn_conv_2',
                                                kernel_regularizer=None, bias_initializer=self.init_xavier,
                                                kernel_initializer=self.init_xavier)
                cnn_fc_layer2 = tf.layers.batch_normalization(cnn_fc_layer2, name=name + "_2")
                cnn_fc_layer2 = self.activation(cnn_fc_layer2)
                cnn_fc_layer2 = tf.nn.softmax(
                    cnn_fc_layer2)  # for probabilities #TODO: have the distance vectpr activate? add to loss?

            return cnn_fc_layer2, tf_weights_conv_attn

        #  -- CONNECTED --
        def connected(cnn_fc_layer__, layer_ff__, layer_dist__, inputs_ff, inputs_dist, dist_gene_attn_out,
                      gene_gene_attn_out, cnn_gene_attn_out):
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as var_scope:
                for i in range(num_connected_layers):
                    name = 'conn_dense_%d' % i
                    if i == 0:
                        if self.load_model == 1:
                            layer = tf.layers.dense(inputs=tf.concat(
                                [cnn_fc_layer__, tf.multiply(inputs_ff, dist_gene_attn_out)],
                                axis=1), units=connected_units[i], activation=None, name=name)
                        else:
                            layer = tf.layers.dense(
                                inputs=tf.concat([cnn_fc_layer__, tf.multiply(cnn_gene_attn_out, inputs_ff),
                                                  tf.multiply(inputs_ff, gene_gene_attn_out),
                                                  tf.multiply(inputs_ff, dist_gene_attn_out)],
                                                 axis=1), units=connected_units[i], activation=None, name=name)
                    else:
                        layer = tf.layers.dense(inputs=layer__, units=connected_units[i], activation=None, name=name,
                                                kernel_regularizer=None, bias_initializer=self.init_xavier)
                    layer_ = tf.layers.batch_normalization(layer, name=name)
                    layer__ = self.activation(layer_)
                    tf.summary.histogram(name, layer__)
            return layer__

        def pred(padded_inputs_cnn, inputs_ff, inputs_dist, encoder____):
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as var_scope:
                dist_attn = dist_for_gene_attn(inputs_dist)
                cnn_attn, tf_weights_conv_attn = cnn_for_gene_attn(padded_inputs_cnn)
                gene_attn = gene_exp_for_gene_attn(inputs_ff)
                cnn_out, tf_weights_conv = cnn(padded_inputs_cnn)
                pred = tf.layers.dense(
                    inputs=connected(cnn_fc(cnn_out), encoder____, dist(inputs_dist), inputs_ff,
                                     inputs_dist, dist_attn, gene_attn, cnn_attn), units=1, name="pred")
            return pred, dist_attn, cnn_attn, gene_attn, tf_weights_conv, tf_weights_conv_attn

        def save_attn_results_from_batch(batch_out, f_out_name, seen_rows, methylation_level, input_data_batch,
                                         ambient_seq_batch):
            attn_fname = '../out/' + f_out_name + '.csv'
            raw_fname = '../out/' + f_out_name + '_raw_data.csv'
            raw_seq_fname = '../out/' + f_out_name + '_raw_sequence_data.csv'

            with open(attn_fname, 'a') as csvwriter:
                with open(raw_fname, 'a') as csvwriter_raw:
                    with open(raw_seq_fname, 'a') as csvwriter_raw_sequence_data:
                        # csvwriter.write(','.join([str(i) for i in range(self.Conf.numGenes)]))
                        # csvwriter.write('\n')
                        for i in range(len(batch_out)):
                            row = batch_out[i]
                            raw_data_row = input_data_batch[i]
                            ambient_seq_row = ambient_seq_batch[i]
                            if tuple(row) in seen_rows:
                                continue
                            seen_rows.add(tuple(row))
                            row = row.astype(str)
                            raw_data_row = raw_data_row.astype(str)
                            ambient_seq_row = ambient_seq_row.astype(str)
                            row = np.append(row, methylation_level[i])
                            csvwriter.write('\n')
                            csvwriter.write(','.join(row))
                            csvwriter.write('\n')
                            csvwriter_raw.write('\n')
                            csvwriter_raw.write(','.join(raw_data_row))
                            csvwriter_raw.write('\n')
                            csvwriter_raw_sequence_data.write('\n')
                            csvwriter_raw_sequence_data.write(','.join(ambient_seq_row))
                            csvwriter_raw_sequence_data.write('\n')
            return seen_rows

        # putting the graph all together before testing / training:
        # ------------------------------------------------------

        encoded = encoder(inputs_ff)
        full_model, dist_attn, cnn_attn, gene_attn, tf_weights_conv, tf_weights_conv_attn = pred(padded_inputs_cnn,
                                                                                                 inputs_ff, inputs_dist,
                                                                                                 encoded)
        loss_full_model = self.loss_func(labels=labels_, predictions=full_model)
        tf.summary.scalar('loss_full_model', loss_full_model)

        # updates for batch norm
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(extra_update_ops):
            if self.optimizer_func == tf.train.MomentumOptimizer:
                opt_full_model = self.optimizer_func(self.lr, 0.95)  # .minimize(cost)
            else:
                opt_full_model = self.optimizer_func(self.lr)  # .minimize(cost)

            gradients, variables = zip(*opt_full_model.compute_gradients(
                loss_full_model))  # todo: remove the autoencoder's gradients and verify working
            current_gradients_norm = tf.global_norm(gradients)
            train_step_full_model = opt_full_model.apply_gradients(zip(gradients, variables))
            tf.summary.scalar('gradients_norm_full', current_gradients_norm)
        # adding saver op
        saver = tf.train.Saver()

        sess = tf.Session()

        if self.load_model:
            if self.load_model == 1:
                saver.restore(sess, "../out/ch3_e_blind/model_1.ckpt")
            elif self.load_model == 2:
                saver.restore(sess, "../out/ch3_e_blind/model_2.ckpt")
            elif self.load_model == 3:
                saver.restore(sess, "../out/ch3_e_blind/model_3.ckpt")
            else:
                print("Illegal modelID - please choose 1/2/3 or 0 to train from scratch.")
            print("Model restored.")

        if self.test_time:
            if not self.is_prediction:
                for weight_name in ["conv"]:
                    if weight_name == "conv_attn":
                        weights = tf_weights_conv_attn
                    else:
                        weights = tf_weights_conv
                    cnn_weights = sess.run(weights)
                    for i in range(self.conv_f):
                        cnn_filter = np.asarray(cnn_weights[:, :, 0, i])
                        cnn_filter = pd.DataFrame(cnn_filter)
                        cnn_filter.to_csv('../out/postTrainingAnalysis/%s_filter_%d.csv' % (weight_name, i), index=None)

            # TEST EVAL
            batch_size = 800

            sum = 0
            preds_all = []
            labels_all = []
            cpg_id_names_all = []
            expression_sample_names_all = []
            sum_abs_error = 0
            seen_rows_dist = set([])
            seen_rows_seq = set([])
            seen_rows_gene = set([])
            n_batches = math.ceil(self.test.num_examples / batch_size)
            # print("If you're testing a trained model and want the attention output files, make sure to remove previously created attention"
            #       " files, otherwise results will be appended to them.")

            for i in range(n_batches):
                if i % 5 == 1:
                    if self.is_prediction:
                        print("completed %d of %d batches of predictions." % (i, n_batches))
                    else:
                        print("%d of %d" % (i, n_batches))
                        print("MAE so far: ", sum_abs_error / float(len(preds_all)))
                        print("spearman so far: ", spearmanr(preds_all, labels_all))
                        save_obj(labels_all, "labels", directory="../out/")
                    save_obj(preds_all, "preds", directory="../out/")

                ambient_seq_batch, gene_exp_batch, dist_batch, labels_batch, _, \
                cpg_id_names_batch, expression_sample_names_batch = self.test.next_batch(
                    batch_size=batch_size, testing=True)
                labels_batch = np.array(labels_batch).reshape((-1, 1))
                ambient_seq_batch = np.array(ambient_seq_batch).reshape(
                    (-1, self.Conf.numSurrounding * 2, self.Conf.numBases, 1))
                batch_cost, pred = sess.run([loss_full_model, full_model], feed_dict={inputs_ff: gene_exp_batch,
                                                                                      inputs_cnn: ambient_seq_batch,
                                                                                      labels_: labels_batch,
                                                                                      inputs_dist: dist_batch})

                if self.is_prediction:
                    cpg_id_names_all.extend(cpg_id_names_batch)
                    expression_sample_names_all.extend(expression_sample_names_batch)
                else:
                    dn = sess.run(dist_attn, feed_dict={inputs_dist: dist_batch})
                    cn = sess.run(cnn_attn, feed_dict={inputs_cnn: ambient_seq_batch})
                    gn = sess.run(gene_attn, feed_dict={inputs_ff: gene_exp_batch})

                    seen_rows_dist = save_attn_results_from_batch(dn, 'dist_attn', seen_rows_dist, labels_batch,
                                                                  dist_batch, ambient_seq_batch.reshape(-1, 800 * 5))
                    seen_rows_seq = save_attn_results_from_batch(cn, 'sequence_attn', seen_rows_seq, labels_batch,
                                                                 ambient_seq_batch.reshape(-1, 800 * 5),
                                                                 ambient_seq_batch.reshape(-1, 800 * 5))
                    seen_rows_gene = save_attn_results_from_batch(gn, 'gene_attn', seen_rows_gene, labels_batch,
                                                                  gene_exp_batch,
                                                                  ambient_seq_batch.reshape(-1, 800 * 5))
                    labels_all.extend(labels_batch.flatten())
                    sum_abs_error += batch_cost * len(labels_batch)

                preds_all.extend(pred.flatten())
                sum += len(pred)

            if self.is_prediction:
                df = pd.DataFrame(
                    {'cpg_id': cpg_id_names_all, 'sample_id': expression_sample_names_all, 'pred_methyl': preds_all})
                df.to_csv('../out/predictions.csv', index=False)
                print("Completed job! Predictions can be found under the out directory.")
            if not self.is_prediction:
                print("spearman all: ", spearmanr(preds_all, labels_all))
                print("loss all: ", sum_abs_error / float(len(preds_all)))
        else:
            # TRIAINING

            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('../logs/train/model_%s' % self.modelID, sess.graph)
            val_writer = tf.summary.FileWriter('../logs/validation/model_%s' % self.modelID)
            validation_ch3_blind_writer = tf.summary.FileWriter('../logs/val_ch3_blind/model_%s' % self.modelID)
            validation_e_blind_writer = tf.summary.FileWriter('../logs/val_e_blind/model_%s' % self.modelID)
            if not self.load_model:
                sess.run(tf.global_variables_initializer())

            mse_queue_val = deque([])
            cor_queue_val = deque([])
            avg_cor_prev_val = -np.inf
            mse_queue_val_e_blind = deque([])
            loss_queue_val_e_blind = deque([])
            avg_mse_prev_val_e_blind = np.inf
            counter = 0
            stop = False
            max_val_correl = -np.inf
            max_ch3_blind_correl = -np.inf
            max_e_blind_correl = -np.inf
            start = time.time()

            for e in range(self.Conf.epochs):
                print("Epoch {}".format(e))
                for ii in range(self.train.num_examples // self.Conf.batch_size):
                    counter += 1
                    current_time = time.time()
                    if current_time >= start + 60 * 180:  # one hour has passed (60 seconds x 60 minutes gives no. of seconds in 1 hour)
                        # if current_time >= start + 10:
                        if self.save_models:
                            save_path = saver.save(sess,
                                                   "../out/ch3_e_blind/model_180m_elapsed_%s_%d_ch3_e_blind.ckpt" % (
                                                       self.modelID, counter))
                            if self.save_weights:
                                save_single_weights(sess, self.tf_weights_genes, "genes")

                        tf.reset_default_graph()
                        return

                    if ii % 50 == 0:
                        # TRAIN EVAL
                        ambient_seq_batch, gene_exp_batch, dist_batch, labels_batch, _, _, _ = self.train.next_batch(
                            self.Conf.batch_size)
                        labels_batch = np.array(labels_batch).reshape((-1, 1))
                        ambient_seq_batch = np.array(ambient_seq_batch).reshape(
                            (-1, self.Conf.numSurrounding * 2, self.Conf.numBases, 1))
                        batch_cost, _, pred, summary = sess.run(
                            [loss_full_model, train_step_full_model, full_model, merged],
                            feed_dict={inputs_ff: gene_exp_batch,
                                       inputs_cnn: ambient_seq_batch,
                                       labels_: labels_batch,
                                       inputs_dist: dist_batch})
                        # tf.Summary.Value.add(tag='correl', simple_value=spearmanr(pred, labels_batch))
                        self.print_step(False, train_writer, summary, pred, counter, gene_exp_batch, batch_cost,
                                        labels_batch)

                        # VALIDATION EVAL
                        ambient_seq_batch, gene_exp_batch, dist_batch, labels_batch, _, _, _ = self.validation.next_batch(
                            batch_size=self.Conf.batch_size * 2)
                        labels_batch = np.array(labels_batch).reshape((-1, 1))
                        ambient_seq_batch = np.array(ambient_seq_batch).reshape(
                            (-1, self.Conf.numSurrounding * 2, self.Conf.numBases, 1))
                        batch_cost, pred, summary = sess.run([loss_full_model, full_model, merged],
                                                             feed_dict={inputs_ff: gene_exp_batch,
                                                                        inputs_cnn: ambient_seq_batch,
                                                                        labels_: labels_batch,
                                                                        inputs_dist: dist_batch})
                        dn = sess.run(dist_attn, feed_dict={inputs_dist: dist_batch})
                        cn = sess.run(cnn_attn, feed_dict={inputs_cnn: ambient_seq_batch})
                        gn = sess.run(gene_attn, feed_dict={inputs_ff: gene_exp_batch})

                        self.print_step(False, val_writer, summary, pred, counter, gene_exp_batch, batch_cost,
                                        labels_batch, "VAL___________")

                        corr = spearmanr(pred, labels_batch)

                        if corr[0] > 0.86 and corr[0] > max_val_correl:
                            if self.save_models:
                                save_path = saver.save(sess,
                                                       "../out/ch3_e_blind/m_%s_%d_ch3_e_cor_%.2f_mae_%.2f.ckpt" % (
                                                           self.modelID, counter, corr[0], batch_cost))
                            max_val_correl = corr[0]
                        if len(cor_queue_val) < 10:
                            cor_queue_val.append(corr[0])
                        else:
                            avg_cor = np.mean(cor_queue_val)
                            if avg_cor <= avg_cor_prev_val or math.isnan(avg_cor):
                                print("avg_cor", avg_cor)
                                print("cor_queue", cor_queue_val)
                                print("STOP at %d" % counter)  # create_boxplot(pred, labels_batch_raw, counter)
                                if self.save_models:
                                    save_path = saver.save(sess,
                                                           "../out/ch3_e_blind/model_%s_%d_ch3_e_blind_stop.ckpt" % (
                                                               self.modelID, counter))
                                return
                            avg_cor_prev_val = avg_cor
                            cor_queue_val = deque([])

                        # CH3 BLIND VALIDATION (same gene exp, new methylation location (sequence)):
                        ambient_seq_batch, gene_exp_batch, dist_batch, labels_batch, _, _, _ = self.validation_ch3_blind.next_batch(
                            batch_size=self.Conf.batch_size * 2)
                        labels_batch = np.array(labels_batch).reshape((-1, 1))
                        ambient_seq_batch = np.array(ambient_seq_batch).reshape(
                            (-1, self.Conf.numSurrounding * 2, self.Conf.numBases, 1))
                        batch_cost, pred, summary = sess.run([loss_full_model, full_model, merged],
                                                             feed_dict={inputs_ff: gene_exp_batch,
                                                                        inputs_cnn: ambient_seq_batch,
                                                                        labels_: labels_batch,
                                                                        inputs_dist: dist_batch})

                        corr = spearmanr(pred, labels_batch)
                        self.print_step(False, validation_ch3_blind_writer, summary, pred, counter, gene_exp_batch,
                                        batch_cost, labels_batch, "CH3___________")

                        if corr[0] > 0.99 and corr[0] > max_ch3_blind_correl:
                            if self.save_models:
                                save_path = saver.save(sess, "../out/ch3_blind/model_%s_%d_ch3_blind.ckpt" % (
                                    self.modelID, counter))
                            max_ch3_blind_correl = corr[0]

                        # E BLIND VALIDATION (same methylation location, new gene expression):
                        ambient_seq_batch, gene_exp_batch, dist_batch, labels_batch, _, _, _ = self.validation_e_blind.next_batch(
                            batch_size=self.Conf.batch_size * 2)
                        labels_batch = np.array(labels_batch).reshape((-1, 1))
                        ambient_seq_batch = np.array(ambient_seq_batch).reshape(
                            (-1, self.Conf.numSurrounding * 2, self.Conf.numBases, 1))
                        batch_cost, pred, summary = sess.run([loss_full_model, full_model, merged],
                                                             feed_dict={inputs_ff: gene_exp_batch,
                                                                        inputs_cnn: ambient_seq_batch,
                                                                        labels_: labels_batch,
                                                                        inputs_dist: dist_batch})

                        corr = spearmanr(pred, labels_batch)
                        self.print_step(False, validation_e_blind_writer, summary, pred, counter, gene_exp_batch,
                                        batch_cost, labels_batch, "E___________")

                        if corr[0] > 0.99 and corr[0] > max_e_blind_correl:
                            if self.save_models:
                                save_path = saver.save(sess, "../out/e_blind/model_%s_%d_e_blind.ckpt" % (
                                    self.modelID, counter))
                            max_e_blind_correl = corr[0]
                            if len(loss_queue_val_e_blind) < 10:
                                loss_queue_val_e_blind.append(batch_cost)
                            else:
                                avg_mse_e_blind = np.mean(loss_queue_val_e_blind)
                                if avg_mse_e_blind >= avg_mse_prev_val_e_blind:
                                    print(
                                        "STOP at %d" % counter)  # create_boxplot(pred, labels_batch_raw, counter)
                                    if self.save_models:
                                        save_path = saver.save(sess,
                                                               "../out/e_blind/model_%s_%d_e_blind_stop.ckpt" % (
                                                                   self.modelID, counter))
                                        print("Model saved in file: %s" % save_path)
                                    # break
                                avg_mse_prev_val_e_blind = avg_mse_e_blind
                                loss_queue_val_e_blind = deque([])
                    else:
                        ambient_seq_batch, gene_exp_batch, dist_batch, labels_batch, _, _, _ = self.train.next_batch(
                            self.Conf.batch_size)
                        labels_batch = np.array(labels_batch).reshape((-1, 1))
                        ambient_seq_batch = np.array(ambient_seq_batch).reshape(
                            (-1, self.Conf.numSurrounding * 2, self.Conf.numBases, 1))
                        batch_cost, _ = sess.run([loss_full_model, train_step_full_model],
                                                 feed_dict={inputs_ff: gene_exp_batch,
                                                            inputs_cnn: ambient_seq_batch,
                                                            labels_: labels_batch,
                                                            inputs_dist: dist_batch})
                print("Epoch: {}/{}...".format(e + 1, self.Conf.epochs))

    def print_step(self, is_autoencoder, writer, summary, pred, counter, gene_exp_batch, batch_cost, labels_batch,
                   validation_type=''):

        if is_autoencoder:
            rand_range = np.arange(0, self.Conf.batch_size * 2)
            random_batch_samples_idx = np.random.choice(rand_range, 10)
            corr = spearmanr(pred[random_batch_samples_idx[0]], gene_exp_batch[random_batch_samples_idx[0]])
            print(validation_type + " corr: ", corr[0], "loss", batch_cost)
            if corr[0] > 0.3:
                samples_corr = []
                for s in random_batch_samples_idx:
                    samples_corr.append(spearmanr(pred[s], gene_exp_batch[s])[0])
                avg_cor_sampled = np.mean(samples_corr)
                if avg_cor_sampled > 0.3:
                    print("autoencoder avg passes %.2f" % avg_cor_sampled)

        else:
            corr = spearmanr(pred, labels_batch)
            if validation_type != '':
                print(validation_type + " corr: ", corr[0], "loss", batch_cost)
            extSummary = tf.Summary()
            extSummary.value.add(tag='correl', simple_value=corr[0])
            extSummary.value.add(tag='correl_pval', simple_value=corr[1])
            writer.add_summary(extSummary, counter)
            writer.add_summary(summary, counter)

    def dense_to_one_hot(self, labels_dense, num_classes, neighbourLossAlpha=None, disregardCenter=False):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        if neighbourLossAlpha:
            num_classes += 2  # for padding, one
            index_offset = np.arange(num_labels) * num_classes
            labels_one_hot = np.zeros((num_labels, num_classes))
            labels_one_hot.flat[
                index_offset + labels_dense.ravel() + 1] = 1  # +1 due to padding induced by numclasses+=2 (we want padding on two ends and without this it starts with 0), #0 becomes 1 and 9 becomes 10 out of ix 11
            # adding alpha before actual label and after
            labels_one_hot.flat[index_offset + labels_dense.ravel() + 2] = neighbourLossAlpha
            labels_one_hot.flat[index_offset + labels_dense.ravel()] = neighbourLossAlpha
            # shrinking it back to real num_class (ignoring edges of each sub array)
            labels_one_hot = labels_one_hot[:, 1:-1]
        else:
            index_offset = np.arange(num_labels) * num_classes
            labels_one_hot = np.zeros((num_labels, num_classes))
            labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    def scatter_plot(self, x, y, counter, name):
        plt.scatter(x, y, c='blue', alpha=0.1, s=20)
        plt.savefig('../out/plots/model_%s_%s_step_%d.png' % (self.modelID, name, counter))
        plt.xlim(0, 1)
        # plt.ylim(-2, 2)
        plt.xlabel('Predicted methylation')
        plt.ylabel('Actual methylation')
        plt.close()
