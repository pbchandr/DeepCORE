#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Pramod Bharadwaj Chandrashekar
@email: pchandrashe3@wisc.edu
"""

import argparse
import os
import time
import pickle as pk
import numpy as np
import sklearn.metrics as skm
import tensorflow as tf
import DeepCORE_data_util as ddu
from DeepCORE_model import CnnBilstm

def get_binary_perfromance(y_true, y_pred):
    """Function to return the precision, recall, fscore, and accuracy"""
    precision, recall, fscore, _ = skm.precision_recall_fscore_support(y_true, y_pred,
                                                                       average="micro")
    acc = skm.accuracy_score(y_true, y_pred)
    return precision, recall, fscore, acc

def get_auc(y_true, y_pred_scores, method='max'):
    """Function to return the AUC score for the classifier"""

    # Converting multi-class probability into binary class
    num_class = np.size(y_pred_scores, 1)
    if num_class > 2:
        mid_class = int(num_class/2)
        if method == 'max':
            pred_scores = np.array([np.amax(y_pred_scores[:, 0:mid_class], axis=1),
                                    np.amax(y_pred_scores[:, mid_class:num_class], axis=1)])
        elif method == 'mean':
            pred_scores = np.array([np.mean(y_pred_scores[:, 0:mid_class], axis=1),
                                    np.mean(y_pred_scores[:, mid_class:num_class], axis=1)])

        pred_scores = np.transpose(pred_scores)
        pos_idx = np.where(pred_scores[:, 1] >= 0.5)[0]
        neg_idx = np.where(pred_scores[:, 1] < 0.5)[0]
        pred_scores[pos_idx, 0] = 1 - pred_scores[pos_idx, 1]
        pred_scores[neg_idx, 1] = 1 - pred_scores[neg_idx, 0]
    else:
        pred_scores = y_pred_scores

    if len(y_true) != len(pred_scores):
        y_true = y_true[0:len(pred_scores)]

    return skm.roc_auc_score(y_true, pred_scores[:, 1])

def train_step(sess, model, data, labels, args):
    """Training step involving optimizing the weights based on the data"""
    avg_cost, error, wt_cost, attn_cost = 0.0, 0.0, 0.0, 0.0
    keep_prob = 1 - args.dropout_rate
    total_batch = int(len(data)/args.batch_size)

    if args.normalize:
        max_intensity = ddu.get_max_count(data.copy(), args.add_sequence_info,
                                          args.add_epigenetic_info, args.flanking_region,
                                          args.flanking_width)
    else:
        max_intensity = 1.0

    for ptr in range(0, len(data), args.batch_size):
        # Run backprop and cost during training
        features = ddu.get_feature_data(data[ptr:ptr+args.batch_size].copy(),
                                        args.add_sequence_info, args.add_epigenetic_info,
                                        args.flanking_region, args.flanking_width)
        features = features/max_intensity
        features = features.transpose(0, 2, 1)
        #print('features before ',features.shape)

        #decoder_ip = np.ones([features.shape[0], 1])
        if args.non_zero:
            features = features + 0.5
        if args.add_epigenetic_info:
            if args.epigenetic_index != 'all':
                epi_idx = [int(x) for x in args.epigenetic_index.split(',')]
                if args.add_sequence_info:
                    feat_ind = range(0, 4) + epi_idx
                else:
                    feat_ind = epi_idx
                features = features[:, :, feat_ind]
        #print('features ',features.shape)
        cost, err, wls, als, _ = sess.run([model.loss, model.error, model.out_wt_loss,
                                           model.attn_loss, model.optimizer],
                                          feed_dict={model.input_x: features,
                                                     model.input_y: labels[ptr:ptr+args.batch_size],
                                                     model.dropout_keep_prob: 1-keep_prob})
        # Compute average loss across batches
        avg_cost += cost/total_batch
        error += np.mean(err)/total_batch
        wt_cost += wls/total_batch
        attn_cost += als/total_batch

    print "\nError: %.3f, Weight Cost: %.3f, Attn Cost: %.3f"%(error, wt_cost, attn_cost)
    return avg_cost, max_intensity

def predict(sess, model, eval_data, eval_labels, max_intensity, args, verbose='train'):
    """Function to get the predictions based on the built model"""
    predictions = []
    attentions = []
    #cv_out = []

    for ptr in range(0, len(eval_data), args.batch_size):
        features = ddu.get_feature_data(eval_data[ptr:ptr+args.batch_size].copy(),
                                        args.add_sequence_info, args.add_epigenetic_info,
                                        args.flanking_region, args.flanking_width)
        features = features/max_intensity
        features = features.transpose(0, 2, 1)

        #decoder_ip = np.ones([features.shape[0], 1])
        if args.non_zero:
            features = features + 0.5
        if args.add_epigenetic_info:
            if args.epigenetic_index != 'all':
                epi_idx = [int(x) for x in args.epigenetic_index.split(',')]
                if args.add_sequence_info:
                    feat_ind = range(0, 4) + epi_idx
                else:
                    feat_ind = epi_idx
                features = features[:, :, feat_ind]

        pred, attn, cv = sess.run([model.scores, model.attention, model.cont_vec],
                                  feed_dict={model.input_x: features,
                                             model.dropout_keep_prob: 1.0})
        predictions.extend(pred)
        attentions.extend(attn)
        #cv_out.extend(cv)

    predictions = np.asarray(predictions)
    #cv_out = np.asarray(cv_out)

#    print "ConVec - Max: %.4f, Min: %d, Mean: %d, Median: %d"%(np.max(cv_out), np.min(cv_out),
#                                                               np.mean(cv_out), np.median(cv_out))
    print "Max Pred: %.4f, Min Pred: %.4f"%(np.max(predictions), np.min(predictions))
    print "Max Truth: %.4f, Min Truth: %.4f"%(np.max(eval_labels), np.min(eval_labels))

    te_dat = eval_data.copy()
    te_dat = te_dat[['gene_id', 'chromosome_name', 'transcript_start',
                     'transcript_end', 'strand', 'TPM', 'region', 'tissue']]
    te_dat = te_dat.reset_index(drop=True)

    pred_info = [te_dat, attentions, predictions, eval_labels]
    return predictions, pred_info

def evaluate_regression(y_true, y_pred_scores, cutoff, verbose):
    """ Function to compute performance of a regression model """
    print "----------%s----------"%(verbose)
    # Check if truth and pred labels have the same size
    if len(y_true) != len(y_pred_scores):
        y_true = y_true[0:len(y_pred_scores)]

    # Compute the MSE score
    mse = skm.mean_squared_error(y_true, y_pred_scores)

    y_true_bin = ddu.discretize_labels(np.copy(y_true), cutoff)
    y_pred_bin = ddu.discretize_labels(np.copy(y_pred_scores), cutoff)

    precision, recall, fscore, acc = get_binary_perfromance(y_true_bin, y_pred_bin)
    print "MSE: %.5f, P: %.5f, R: %.5f, F1: %.5f, ACC: %.5f"%(mse, precision, recall,
                                                              fscore, acc)
    print skm.confusion_matrix(y_true_bin, y_pred_bin)
    return mse, fscore, acc

def train_regression(args):
    """ Training method"""
    tot_st_time = time.time()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # Step 1: Load and preprocess data
    st_time = time.time()

    # Read and fetch data and labels
    gene_info, labels = ddu.process_data(data_file=args.gex_gepi_data_file,
                                         chromosome=args.chromosome_name,
                                         num_classes=args.num_classes,
                                         categorize=args.grouping_method)

    if args.num_classes > 1:
        labels = np.argmax(labels, 1)
    labels = np.reshape(labels, (-1, 1))
    cutoff = np.median(labels)

    tr_info, trr, val_info, valr, te_info, ter = ddu.split_data(data=gene_info,
                                                                labels=labels,
                                                                save_location=args.save,
                                                                split_percent=args.split_percent,
                                                                need_balance=args.balanced_train)

    print("Sample length - Train : %d, Valid : %d, Test : %d"%(len(tr_info), len(val_info),
                                                               len(val_info)))

    with open('genes_84.txt', 'r') as gene_loc_file:
        keep_genes = gene_loc_file.readlines()
    keep_genes = [s.replace('\n', '') for s in keep_genes]

    te_84_info = gene_info[gene_info['gene_id'].isin(keep_genes)]
    ter_84 = labels[te_84_info.index, :]
    te_84_info = te_84_info.reset_index(drop=True)

    del gene_info, labels
    tot_time = round((time.time() - st_time)/60, 2)
    print "Step 1: Load and preprocess data complete in %.2f min\n"%(tot_time)

    # Step 2: choose the model and create an object of the model
    st_time = time.time()
    args.input_width, args.seq_len, seq_h = args.flanking_width, 4, 0
    if "5hm" in args.gex_gepi_data_file:
        args.epi_len = 5
    else:
        args.epi_len = 6

    if args.epigenetic_index != 'all':
        num_epi = [int(x) for x in args.epigenetic_index.split(',')]
        args.epi_len = len(num_epi)

    if args.add_epigenetic_info:
        seq_h += args.epi_len
    if args.add_sequence_info:
        seq_h += args.seq_len
    args.input_height = seq_h
    args.num_classes = 1

    model = CRAN(args)
    tot_time = round((time.time() - st_time)/60, 2)
    print "Step 2: Model object create completed in %.2f min\n"%(tot_time)

    # Step 3: Build the graph and train the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # Train cycle
        max_tr_mse, max_val_mse, count_eval = 0, 0, 0

        for epoch in range(args.train_epochs):
            output_str = "Epoch: %02d, "%(epoch+1)
            model_save = False

            st_time = time.time()
            avg_cost, max_int = train_step(sess, model, tr_info, trr, args)
            tot_time = round((time.time() - st_time)/60, 2)
            output_str += "Total Train Time: %.2f"%(tot_time)

            print "%s, cost: %.5f"%(output_str, avg_cost)
            st_time = time.time()
            tr_pred, tr_pred_info = predict(sess, model, tr_info, trr, max_int, args, 'train')
            val_pred, val_pred_info = predict(sess, model, val_info, valr, max_int, args, 'valid')

            tr_mse, _, _ = evaluate_regression(trr, tr_pred, cutoff, verbose='Training')
            val_mse, _, _ = evaluate_regression(valr, val_pred, cutoff, verbose='Validation')

            if epoch == 0 or (tr_mse < max_tr_mse and val_mse < max_val_mse):
                model_save = True
                max_tr_mse, max_val_mse = tr_mse, val_mse
                count_eval = 0
            elif tr_mse >= max_tr_mse or val_mse >= max_val_mse:
                count_eval += 1
                print "Total Evaluation time: %.2f\n"%(round((time.time() - st_time)/60, 2))

            if model_save is True and args.save is not None:
                te_pred, te_pred_info = predict(sess, model, te_info, ter, max_int, args, 'test')
                te_mse, _, _ = evaluate_regression(ter, te_pred, cutoff, verbose='Testing')

                te_84_pred, te_84_pred_info = predict(sess, model, te_84_info, ter_84, max_int,
                                                      args, 'test_84')
                
                ddu.dump_data(tr_pred_info, args.save + 'train_pred_info.pkl')
                ddu.dump_data(val_pred_info, args.save + 'val_pred_info.pkl')
                ddu.dump_data(te_pred_info, args.save + 'test_pred_info.pkl')
                ddu.dump_data(te_84_pred_info, args.save + 'test_84_pred_info.pkl')

                np.savetxt(args.save+'train_truth.out', trr, delimiter="\t")
                np.savetxt(args.save+'train_pred.out', tr_pred, delimiter="\t")

                np.savetxt(args.save+'test_truth.out', valr, delimiter="\t")
                np.savetxt(args.save+'test_pred.out', val_pred, delimiter="\t")

                np.savetxt(args.save+'test_truth.out', ter, delimiter="\t")
                np.savetxt(args.save+'test_pred.out', te_pred, delimiter="\t")



                print "Total Evaluation time: %.2f"%(round((time.time() - st_time)/60, 2))
                print "Saving model to {}\n".format(args.save)
                saver.save(sess, args.save)

            if epoch > 15 and count_eval > 4:
                break
    print "Optimization Finished!"
    tot_end_time = round((time.time() - tot_st_time)/60, 2)
    print "Step 2: Model object create completed in %.2f min\n"%(tot_end_time)


def main():
    """ Main: This method is used to parse all the arguements and call train function """
    parser = argparse.ArgumentParser()

    # Input data file
    parser.add_argument('--gex_gepi_data_file', type=str, default='data/lung_chr_all.csv',
                        help='Gene Expression with epigenetic data input file location')
    parser.add_argument('--add_epigenetic_info', type=bool, default=False,
                        help='Epigenetic details to be included or not')
    parser.add_argument('--add_sequence_info', type=bool, default=False,
                        help='Should the data be flattened')
    parser.add_argument('--non_zero', type=bool, default=False,
                        help='If true, 0.5 is added to the input data')
    parser.add_argument('--epigenetic_index', type=str, default="all",
                        help='Choose between 1, 2, 3, 4, 5, all. Also choose multiple.')
    parser.add_argument('--normalize', type=bool, default=False,
                        help='Use this option to normalize inputs')

    parser.add_argument('--flanking_region', type=str, default='none',
                        help='Choose between upstream, downstream, both, none')
    parser.add_argument('--flanking_width', type=int, default=10000,
                        help='Choose the width of flanking around TSS')

    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--task', type=str, default='regression', help='Choose task type')
    parser.add_argument('--grouping_method', type=str, default='histogram',
                        help='Choose from rank(binary), percentile/histogram(multi-class)')

    parser.add_argument('--split_percent', type=float, default=0.8,
                        help='Choose how the tain and test split to occur.')
    parser.add_argument('--balanced_train', type=bool, default=False,
                        help='Specify this parameter if all the training set is to be balanced.')
    parser.add_argument('--chromosome_name', type=str, default='All',
                        help='Chromosomes that should be included for analysis, comma delimited')
    parser.add_argument('--bin_cutoff', type=float, default=0.5,
                        help='Probability cutoff to decide class labels. Must be between 0 and 1.')

    # Hyperparameters
    # 1 - CNN
    parser.add_argument('--cnn_num_layers', type=str, default='5',
                        help='Number of CNN layers')
    parser.add_argument('--cnn_filter_sizes', type=str, default='5',
                        help='filter sizes, comma delimited. Must be equal to the no of layers')
    parser.add_argument('--cnn_num_filters', type=str, default='128',
                        help='No of filters, comma delimited. i.e. dim for wts in each filters.')
    parser.add_argument('--cnn_stride_length', type=str, default='1',
                        help='Filter strides, comma delimited. i.e steps for sliding window.')
    parser.add_argument('--cnn_pool_sizes', type=str, default='500', help='Pooling size')

    # 2 - RNN Encoder
    parser.add_argument('--add_encoder', type=bool, default=False,
                        help='Flag to add encoder layer')
    parser.add_argument('--encoder_model_type', type=str, default='None',
                        help='Choose between None, BIRNN, BIURNN, BIGRU, BILSTM, BILSTMP')
    parser.add_argument('--encoder_num_layers', type=int, default=1,
                        help='Number of rnn layers.')
    parser.add_argument('--encoder_hid_dims', type=str, default='32',
                        help='Number of hidden dimensions for encoder rnn layer, comma delimited.')

    # 3 - Attention Unit
    parser.add_argument('--attn_hid_dims', type=str, default='32',
                        help='Number of hidden dimensions for attention fc layer, comma delimited.')
    parser.add_argument('--attn_wt_randomize', type=bool, default=False,
                        help='Flag to set the init attn weights to random')
    parser.add_argument('--add_enc_last_state', type=bool, default=False,
                        help='Flag to include last state of encoder to attention computation')
    parser.add_argument('--attn_score_activator', type=str, default='tanh',
                        help='Choose between relu and tanh')
    parser.add_argument('--attn_estimator', type=str, default='softmax',
                        help='Choose btwn softmax, softmax+temp, gumbell softmax, sparsemax')
    parser.add_argument('--attn_temp', type=float, default=1.0,
                        help='Choose the sharpening temperature')
#    parser.add_argument('--attn_dropout', type=str, default='None',
#                        help='Give the dropout probability or None for attention layer')
#    parser.add_argument('--add_attn_bias', type=bool, default=False,
#                        help='Flag to add bias term to the attention mechanism')

    parser.add_argument('--attn_regularizer', type=str, default='l2',
                        help='Choose btwn l0, l2, entropy, deviation')
    parser.add_argument('--attn_reg_lambda', type=float, default=0.005, help='l2_reg for attn')
    parser.add_argument('--attn_type', type=str, default='soft', help='self vs soft')

    # 4 - Decoder Unit
    parser.add_argument('--decoder_multi_view', type=bool, default=False,
                        help='Flag to set the RNN for seq and epi separate or together')
    parser.add_argument('--decoder_model_type', type=str, default='None',
                        help='Choose between None, BILSTM, BIURNN, BIGRU, BILSTM, BILSTMP, FCN')
    parser.add_argument('--decoder_hid_dims', type=str, default='32',
                        help='Number of hidden dimensions for decoder, comma delimited.')

    # 4 - FCN
    parser.add_argument('--num_fc_layers', type=int, default=1,
                        help='Number of fully connected layers to be used after convolution layer')
    parser.add_argument('--num_fc_neurons', type=str, default='32',
                        help='Number of kernels for fully connected layers, comma delimited.')

    parser.add_argument('--dropout_rate', type=float, default=1.0,
                        help='Droupout % for handling overfitting. 1 to keep all & 0 to keep none')
    parser.add_argument('--randomize_weights', type=bool, default=False,
                        help='Is true, then the fnn weights are random else they are zeros')
    parser.add_argument('--seq_random_weights', type=bool, default=False,
                        help='Is true, then the seq cnn weights are random else they are zeros')
    parser.add_argument('--epi_random_weights', type=bool, default=False,
                        help='Is true, then the epi cnn weights are random else they are zeros')

    # Settings
    parser.add_argument('--train_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--eval_interval', type=int, default=1, help='Evaluate once in _ epochs')
    parser.add_argument('--batch_size', type=int, default=150, help='Batch size of training')
    parser.add_argument('--learn_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--out_reg_lambda', type=float, default=0.005, help='l2_reg_lambda')

    # Model save paths
    parser.add_argument('--save', type=str, default="model/samp", help="path to save model")
    parser.add_argument('--out_file', type=str, default="results.txt", help="path to save perf")

    args = parser.parse_args()
    print args
    if args.task == 'regression':
        train_regression(args)

if __name__ == '__main__':
    main()
