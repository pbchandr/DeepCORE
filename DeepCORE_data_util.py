#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: pbchandr
"""
import copy
import sys
import os
import random
import pickle as pk
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as ss
from sklearn.model_selection import train_test_split


def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims

def get_max_count(data, add_sequence, add_epigenetics, flanking_region, flanking_length):
    """Function to fetch the max intensities among the histone marks"""
    max_intensity = 0.0
    for ptr in range(0, len(data), 150):
        features = get_feature_data(data[ptr:ptr+150].copy(), add_sequence,
                                    add_epigenetics, flanking_region, flanking_length)
        if np.amax(features) > max_intensity:
            max_intensity = np.amax(features)
    return max_intensity

def get_balanced_indexes(data):
    bin_data = np.sum(data, 1)
    unique, counts = np.unique(np.sum(data, 1), return_counts=True)

    new_data_idx = []
    for index, val in enumerate(unique):
        lbl_idx = np.where(bin_data == val)[0]
        if index == 0:
            new_data_idx = np.random.choice(lbl_idx, min(counts))
        else:
            new_data_idx = np.concatenate([new_data_idx, np.random.choice(lbl_idx, min(counts),
                                                                          replace=False)])
    return new_data_idx

def saturated_mutageneis(seq, nucleotides=['A', 'C', 'T', 'G']):
    """
    Function to generate Saturated Mutagenesis
    (https://en.wikipedia.org/wiki/Saturation_mutagenesis)
    - Gen all possible SNP mutations for each base pair in the sequence
    - Each sample has 5000 bps with nucleotide inforation.
    - Since we are dealing with nucleotide sequence, only 4 nucleotides at each position
        -As the base sequence is given, we can have only 3 other mutations for each position.
        -This gives a total of 5000*3 + 1 number of mutated sequences
    """
    data = np.array([list(seq),]*(len(seq)*(len(nucleotides)-1)+1))
    i = 0
    j = 0
    while i <= (len(data)-len(nucleotides)+1):
        new_nucleotide = copy.copy(nucleotides)
        new_nucleotide.remove(data[i, j])
        data[(i+1):(i + len(new_nucleotide)+1), j] = new_nucleotide
        i = i + len(nucleotides)-1
        j = j + 1
    return data

def perform_one_hot_encoding(data):
    """
    Function to perform one hot encoding of the sequence which is the input to the system
        - Sequences are converted into 4*10000 dim vector
        - 4 dimensions:- Order of dimension is A-1, G-2, C-3, T-4
        - The value in each cell is either 0 or 1 representing corresponding nucleotide.
    """
    info = np.zeros(shape=(len(data), 4, 10000))
    for i, _ in enumerate(data):
        indices = [j for j, a in enumerate(data[i]) if a == 'A']
        info[i, 0, indices] = 1

        indices = [j for j, a in enumerate(data[i]) if a == 'G']
        info[i, 1, indices] = 1

        indices = [j for j, a in enumerate(data[i]) if a == 'C']
        info[i, 2, indices] = 1

        indices = [j for j, a in enumerate(data[i]) if a == 'T']
        info[i, 3, indices] = 1
    return info

def perform_rank_hot_encoding(data):
    """
    Function to perform rank encoding for outputs: 0 - 0000, 1 - 1000, 2 - 1100, etc
    """
    max_rank = max(data)
    info = np.zeros(shape=(len(data), max_rank))
    for i, _ in enumerate(data):
        info[i, 0:data[i]] = 1
    return info

def load_data(gex_gepi_fname, chromosome="All", tissue=None, remove_zero_hm=False):
    """Function to read the data. Filter the chromose if required"""

    print "Load data entered: %s, %s, %s"%(gex_gepi_fname, remove_zero_hm, tissue)
    # Read the gene expression with epigenetic information
    dat = pd.read_csv(gex_gepi_fname, sep='\t')

    # Filter the data based on chromosome
    geq_data = pd.DataFrame()
    if chromosome == "All":
        geq_data = dat
    else:
        chrom = [x for x in chromosome.split(',')]
        for ind, chrm in enumerate(chrom):
            print 'chr'+chrm
            if ind == 0:
                geq_data = dat.loc[dat['chromosome_name'] == ('chr'+chrm)]
            else:
                geq_data = geq_data.append(dat.loc[dat['chromosome_name'] == ('chr'+chrm)])

    #Remove genes with 0 histone marks
    print 'geq_data before removal: %d'%(len(geq_data))
    if remove_zero_hm:
        fname = tissue+'_zero_hm_genes.txt'
#        fname = './data/'+tissue+'_zero_hm_genes.txt'
        zero_hm_file = open(fname, 'r')
#        zero_hm_file = open('./data/lung_zero_hm_genes.txt', 'r')

        rem_genes = [s.replace('\n', '') for s in zero_hm_file.readlines()]
        zero_hm_file.close()
        print 'Zero HM info: Filename: %s, Length: %d'%(fname, len(rem_genes))

        geq_data = geq_data[~geq_data.gene_id.isin(rem_genes)]

    print('Dat shape', geq_data.shape)
    return geq_data

def get_bin_indexes(labels, intervals):
    indx_list = []
    prev = 0
    for index, val in enumerate(intervals):
        if index == 0:
            prev = val
        elif index == len(intervals):
            break
        else:
            match_idx = np.where(np.logical_and(labels >= prev, labels < val))[0]
            indx_list.append(match_idx)
        prev = val
    return indx_list

def get_class_labels(data, num_classes=2, method='rank', is_ordinal=False):
    """
    Function to get class labels using the methods:
        1. Rank(binary): Top 1/3rd and bottom 1/3rd are used as class 1 & 0 respectively.
        2. Percentile(muti-class): Uses percentile cut-offs to bin the labels
        3. Histogram(muti-class): Use the histogram cut-offs to bin the labels

    Once this is done, we perfrom one-hot encoding for binary/multi-class
    and rank-hot encoding for multi-class ordinal

    """
    print "get_class_labels entered: %d, %s"%(num_classes, method)
    # Categorizing the labels
    if method == 'rank':
        data = ss.rankdata([-1 * i for i in data])
        max_rank = max(data)
        keep_indx = np.where((data > max_rank*2/3) | (data < max_rank/3))

        labels = data[keep_indx]
        labels[labels > max_rank*2/3] = 0
        labels[labels != 0] = 1
    elif method == 'percentile':
        keep_indx = None
        labels = ss.rankdata(data)/len(data)
        labels = np.int_(np.multiply(labels, num_classes)) # Convert to ints
        labels[labels > num_classes-1] = num_classes - 1

    unique, counts = np.unique(labels, return_counts=True)
    print("label split ---", dict(zip(unique, counts)))
    labels = np.eye(num_classes)[np.int32(labels)]

    # Rank-hot encoding for ordinal labels
    if is_ordinal:
        labels = perform_rank_hot_encoding(np.argmax(labels, axis=1))

    return labels, keep_indx

def discretize_labels(data, cutoff):
    """This method converts the continuous data into discrete values based on the cutoff"""
    data[data < cutoff] = 0
    data[data >= cutoff] = 1
    return data


def process_data(data_file, chromosome="All", num_classes=2,
                 categorize='percentile', is_ordinal=False):
    """ This method reads and fetches the input data """
    print "Process data entered: %s, %d, %s"%(data_file, num_classes, categorize)

    # Check if path is a file or folder. If folder, read all the files in that folder.
    if os.path.isfile(data_file):
        tissue = data_file.replace('.csv', '')
        print data_file, tissue

        data = load_data(data_file, chromosome=chromosome, tissue=tissue)
        if num_classes > 1:
            labels, keep_indx = get_class_labels(data['TPM'].values, num_classes=num_classes,
                                                 method=categorize, is_ordinal=is_ordinal)
        else:
            labels, keep_indx = np.log(data['TPM'].values+0.01), None
        if keep_indx is not None:
            data = data.ix[keep_indx]

        data = data.reset_index(drop=True)
        data['region'] = 1
        data['tissue'] = tissue
    else:
        files = os.listdir(data_file)
        data, labels = [], []
        for indx, fls in enumerate(files):
            tissue = fls.replace('.csv', '')
            print(indx, data_file+fls, tissue)
            dat = load_data(data_file+fls, chromosome=chromosome, tissue=tissue)

            if num_classes > 1:
                lbls, keep_indx = get_class_labels(dat['TPM'].values, num_classes=num_classes,
                                                   method=categorize, is_ordinal=is_ordinal)
            else:
                lbls, keep_indx = np.log(dat['TPM'].values+0.01), None

            if keep_indx is not None:
                dat = dat.ix[keep_indx]
            dat = dat.reset_index(drop=True)
            dat['region'] = indx
            dat['tissue'] = tissue
            if indx == 0:
                data = dat
                labels = lbls
            else:
                data = pd.concat([data, dat])
                labels = np.append(labels, lbls, axis=0)

    data = data.reset_index(drop=True)
    return data, labels

def split_data(data, labels, save_location=None, split_percent=0.8, need_balance=False):
    """ Function to split the data into test, train, and validation set """
    unq_genes = np.unique(data[['gene_id']])
    tr_genes, vr_genes = train_test_split(unq_genes, test_size=split_percent,
                                          random_state=random.randint(0, 10000)) #666)
    val_genes, te_genes = train_test_split(vr_genes, test_size=0.5, random_state=random.randint(0, 10000)) #6667

    if save_location is not None:
        with open(save_location+'/train_genes.txt', 'w') as tr_file:
            for item in tr_genes:
                tr_file.write("%s\n" % item)

        with open(save_location+'/val_genes.txt', 'w') as val_file:
            for item in val_genes:
                val_file.write("%s\n" % item)

        with open(save_location+'/test_genes.txt', 'w') as te_file:
            for item in te_genes:
                te_file.write("%s\n" % item)

    print("Sample length - Train : %d, Valid : %d, Test : %d"%(len(tr_genes), len(val_genes),
                                                               len(te_genes)))

    trr = labels[data.index[data['gene_id'].isin(tr_genes)]]
    if need_balance:
        trr_idx = get_balanced_indexes(trr)
        trr = trr[trr_idx]
        tr_info = data[data['gene_id'].isin(tr_genes[trr_idx])]
    else:
        tr_info = data[data['gene_id'].isin(tr_genes)]

    valr = labels[data.index[data['gene_id'].isin(val_genes)]]
    val_info = data[data['gene_id'].isin(val_genes)]

    ter = labels[data.index[data['gene_id'].isin(te_genes)]]
    te_info = data[data['gene_id'].isin(te_genes)]

#    if trr.shape[1] == 2:
#        tru, trc = np.unique(np.argmax(trr, 1), return_counts=True)
#        valu, valc = np.unique(np.argmax(valr, 1), return_counts=True)
#        teu, tec = np.unique(np.argmax(ter, 1), return_counts=True)
#    else:
#        tru, trc = np.unique(np.sum(trr, 1), return_counts=True)
#        valu, valc = np.unique(np.sum(valr, 1), return_counts=True)
#        teu, tec = np.unique(np.sum(ter, 1), return_counts=True)
#
#    print("Train label split ---", dict(zip(tru, trc)))
#    print("Validation label split ---", dict(zip(valu, valc)))
#    print("Test label split ---", dict(zip(teu, tec)))

    return tr_info, trr, val_info, valr, te_info, ter

def get_feature_data(data, add_sequence, add_epigenetics, flanking_region, flanking_length):
    """ Function to fetch the features which is given as input to the model """
    # Check if atleast one type of data is selected: sequence and epigenetics
    if (not add_sequence) and (not add_epigenetics):
        print 'No data for training. Plase choose atleast one: add_sequence and add_epigenetics'
        sys.exit(1)


    if add_sequence:
        data.sequence = data.sequence.str.slice(0, len(data['sequence'].values[0])-1)
        seq_info = perform_one_hot_encoding(data["sequence"].values)
        #print "seq_info", seq_info.shape

    if add_epigenetics:
        epigen_data = get_epigenetic_data(data)
        #print "epi_info", epigen_data.shape

    if add_sequence and add_epigenetics:
        gex_gepi_data = np.concatenate((seq_info, epigen_data), 1)
    elif add_sequence and not add_epigenetics:
        gex_gepi_data = seq_info
    else:
        gex_gepi_data = epigen_data

    #print("gex_gepi_data", gex_gepi_data.shape)
    lower_limit, upper_limit = 0, 10000
    if flanking_region == 'upstream':
        lower_limit = gex_gepi_data.shape[2]/2-flanking_length
        upper_limit = gex_gepi_data.shape[2]/2
    elif flanking_region == 'downstream':
        lower_limit = gex_gepi_data.shape[2]/2-1
        upper_limit = gex_gepi_data.shape[2]/2 + flanking_length - 1
    elif flanking_region == 'both':
        lower_limit = gex_gepi_data.shape[2]/2-(flanking_length/2+1)
        upper_limit = gex_gepi_data.shape[2]/2+flanking_length/2-1
    
    #print lower_limit, upper_limit
    gex_gepi_data = gex_gepi_data[:, :, lower_limit:upper_limit]
    return gex_gepi_data

def get_epigenetic_data(data):
    """ Function to extract histone modification data """
    epigen_cols = [col for col in data.columns if 'chipseq' in col]
    epigen_data = np.zeros([len(data), len(epigen_cols), 10000])
    for indx, col in enumerate(epigen_cols):
        epigen_info = data[col].values
        for i in range(0, len(data)):
            temp = np.reshape(np.fromstring(epigen_info[i], dtype=int, sep=","), [1, -1])
            if temp.shape[1] > 10000:
                epigen_data[i, indx, :] = temp[0, 0:10000]
            else:
                epigen_data[i, indx, 0:temp.shape[1]] = temp
#            epi_split = epigen_info[i].split(',')
#            temp = np.reshape(np.asarray(epi_split), [1, -1])
#            if len(epi_split) > 10000:
#                epigen_data[i, indx, 0:10000] = temp[0, (len(epi_split)-10000):len(epi_split)+1]
#            else:
#                epigen_data[i, indx, 0:len(epi_split)] = temp
    return epigen_data

def filter_data(data, labels, gene_loc, task):
    """ Function to filter the data based on gene list from the file"""
    with open(gene_loc, 'r') as gene_loc_file:
        keep_genes = gene_loc_file.readlines()
    keep_genes = [s.replace('\n', '') for s in keep_genes]

    print keep_genes[0:10]
    data = data[data['gene_id'].isin(keep_genes)]
    if task == 'regression':
        labels = labels[data.index]
    else:
        labels = labels[data.index, :]
    print len(data), labels.shape

    data = data.reset_index(drop=True)
    return data, labels

def dump_data(data, file_name):
    with open(file_name, 'w') as mdl_file:
        pk.dump(data, mdl_file)
