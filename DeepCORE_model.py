#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: pbchandr
"""
import sys
import tensorflow as tf
from tensorflow.contrib import rnn

class CnnBilstm(object):
    """
    A CNN for gene expression prediction.
    Uses a convolution layer, max-pooling, and softmax layer with dropouts
    """
    @classmethod
    def get_pooled_indexes(cls, data, pooling_size):
        """Function to get pooled indexes"""
        pooled_index = []
        for ptr in range(0, data.shape[1], pooling_size):
            pooled_index.append(tf.argmax(data[:, ptr: (ptr+pooling_size-1), :, :], axis=1))
        return tf.concat(pooled_index, 1)

    @classmethod
    def weight_variable(cls, shape, random, var_name):
        """Function to create a weights"""
        if random:
            initial = tf.truncated_normal(shape, stddev=0.1)
        else:
            #initial = tf.zeros(shape)
            initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=var_name+'_wt')

    @classmethod
    def bias_variable(cls, shape, random, var_name):
        """Function to create a bias based on the shape"""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=var_name+'_bias')

    @classmethod
    def conv2d(cls, inp, conv_wt, stride_len):
        """Function to call tensorflow convolution"""
        return tf.nn.conv2d(inp, conv_wt, strides=[1, stride_len, 1, 1], padding='VALID')

    @classmethod
    def conv_layer(cls, inp, shape, stride_len, random, var_name):
        """Function to create convolution layer"""
        conv_wt = cls.weight_variable(shape, random, var_name)
        conv_bias = cls.bias_variable([shape[3]], random, var_name)
        return tf.nn.relu(cls.conv2d(inp, conv_wt, stride_len) + conv_bias)

    @classmethod
    def max_pool(cls, inp, pooling_size):
        """Function for max pooling after convolution"""
        pool_output = tf.nn.max_pool(inp, ksize=[1, pooling_size, 1, 1],
                                     strides=[1, pooling_size, 1, 1], padding='SAME')
        return pool_output

    @classmethod
    def fc_layer(cls, inp, shape, random, var_name):
        """Function for creating fully connected network"""
        fc_weight = cls.weight_variable(shape, random, var_name)
        fc_bias = cls.bias_variable([shape[1]], random, var_name)
        fc_mul_wb = tf.add(tf.matmul(inp, fc_weight), fc_bias)
        return fc_mul_wb, fc_weight

    @classmethod
    def linear_transform(cls, inp, shape, random, var_name):
        """Function for creating fully connected network"""
        value_wt = cls.weight_variable(shape, random, '%s-wt'%var_name)
        value_bias = cls.bias_variable([shape[1]], random, '%s-bias'%var_name)
        values_out = tf.add(tf.tensordot(inp, value_wt, [[2], [0]]), value_bias)
        return values_out

    @classmethod
    def rnn_cell(cls, dims, model_type, keep_prob):
        '''Function to define the type of RNN cell to be used in RNN layer'''
        # Simple RNN Cell
        if model_type.lower() == 'birnn':
            cell = rnn.BasicRNNCell(dims)
        # Update gated RNN Cell
        elif model_type.lower() == 'biurnn':
            cell = rnn.UGRNNCell(dims)
        # GRU Celll
        elif model_type.lower() == 'bigru':
            cell = rnn.GRUCell(dims)
        # Simple LSTM Cell
        elif model_type.lower() == 'bilstm':
            cell = rnn.BasicLSTMCell(dims, state_is_tuple=True)
        # LSTM Cell with peepholes
        elif model_type.lower() == 'bilstmp':
            cell = rnn.BasicLSTMCell(dims, state_is_tuple=True, use_peepholes=True)
        return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    @classmethod
    def sigmoid(cls, logits, temperature):
        """Function to compute softmax with temperature"""
        return 2*tf.math.sigmoid(logits)

    @classmethod
    def softmax_with_temp(cls, logits, temperature):
        """Function to compute softmax with temperature"""
        return tf.nn.softmax(logits/temperature, axis=1)

    @classmethod
    def sample_gumbel(cls, shape, eps=1e-20):
        """ Function to fetch gumbel distributed samples"""
        gumbel_samples = tf.random_uniform(shape, minval=0, maxval=1)
        return -tf.log(-tf.log(gumbel_samples + eps) + eps)

    @classmethod
    def gumbel_softmax(cls, logits, temperature):
        """Function to compute gumbel softmax"""
        data_gumbel = logits + cls.sample_gumbel(tf.shape(logits))
        return tf.nn.softmax(data_gumbel/temperature, axis=1)

    @classmethod
    def sparsemax(cls, logits):
        """Function to compute sparsemax transformation of logits"""
        spmax_data = tf.reshape(logits, [-1, logits.shape[1]])
        spmax_out = tf.contrib.sparsemax.sparsemax(spmax_data)
        return tf.expand_dims(spmax_out, -1)

    @classmethod
    def attn_layer(cls, values, query, attn_dims, activator, random, add_attn_query):
        """Function for creating attention layer"""

        value_wt_shape = [int(values.shape[2]), attn_dims]
        value_wt = cls.weight_variable(value_wt_shape, random, 'attn_values')
        value_bias = cls.bias_variable([attn_dims], random, 'attn_values')

        values_out = tf.add(tf.tensordot(values, value_wt, [[2], [0]]), value_bias)
        print "attn values_out", values_out

        if add_attn_query:
            query_expanded = tf.expand_dims(query, 1)

            query_wt_shape = [int(query_expanded.shape[2]), attn_dims]
            query_wt = cls.weight_variable(query_wt_shape, random, 'attn_query')
            query_bias = cls.bias_variable([attn_dims], random, 'attn_query')

            query_out = tf.add(tf.tensordot(query_expanded, query_wt, [[2], [0]]), query_bias)
            print "attn query_out", query_out

            attn_wt_out = values_out + query_out
        else:
            attn_wt_out = values_out

        if activator == 'tanh':
            wt_actvn = tf.nn.tanh(attn_wt_out)
        else:
            wt_actvn = tf.nn.relu(attn_wt_out)

        attn_v_shape = [int(wt_actvn.shape[2]), 1]
        attn_v_wt = cls.weight_variable(attn_v_shape, random, 'attn_v')
        attn_v_bias = cls.bias_variable([attn_v_shape[1]], random, 'attn_v')
        attn_v_out = tf.tensordot(wt_actvn, attn_v_wt, [[2], [0]]) + attn_v_bias

        return attn_v_out

    @classmethod
    def self_attn_layer(cls, ip, attn_dims, random):
        """Function for self-attention layer"""
        wt_shape = [int(ip.shape[2]), attn_dims]
        q_mat = cls.linear_transform(ip, wt_shape, random, 'attn_q')
        k_mat = cls.linear_transform(ip, wt_shape, random, 'attn_k')
        v_mat = cls.linear_transform(ip, wt_shape, random, 'attn_v')

        print 'q_mat', q_mat
        print 'k_mat', k_mat
        print 'v_mat', v_mat

        # Compute dot product b/w Q and K
        qk_mat = tf.matmul(q_mat, k_mat, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale qk_mat
        dk = tf.cast(tf.shape(k_mat)[-1], tf.float32)
        scaled_attn_logits = qk_mat / tf.math.sqrt(dk)
        print "scaled_attn_logits", scaled_attn_logits

        # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
        #attn_scores = tf.nn.softmax(scaled_attn_logits, axis=-1) # (..., seq_len_q, seq_len_k)
        #attn_scores = tfa.activations.sparsemax(scaled_attn_logits, axis=-1) # (..., seq_len_q, seq_len_k

        attn_scores = []
        for i in range(0, 200):
            if i == 0:
                attn_scores = cls.sparsemax(scaled_attn_logits[:, :, i])
            else:
                attn_scores = tf.concat([attn_scores,
                                         cls.sparsemax(scaled_attn_logits[:, :, i])], axis=2)
        print "attn_scores", attn_scores

        attn_v_out = tf.matmul(attn_scores, v_mat)  # (..., seq_len_q, depth_v)
        return attn_v_out, attn_scores

    @classmethod
    def attn_probability(cls, score, estimator, temperature):
        """ Function to estimate probability of attention from scores"""
        if estimator == 'softmax':
            return tf.nn.softmax(score, axis=1)
        elif estimator == 'softmax_temp':
            return cls.softmax_with_temp(score, temperature)
        elif estimator == 'gumbel_softmax':
            return cls.gumbel_softmax(score, temperature)
        elif estimator == 'sparsemax':
            return cls.sparsemax(score)

    @classmethod
    def regularize_attention(cls, attention, regularizer='l2'):
        """Function to fetch regularization loss"""
        if regularizer == 'l0':
            return tf.cast(tf.count_nonzero(attention), tf.float32)
        elif regularizer == 'l2':
            return tf.nn.l2_loss(attention)
        elif regularizer == 'entropy':
            attn_log = tf.log(attention + 0.00001)
            return tf.reduce_sum(tf.multiply(attention, attn_log))
        elif regularizer == 'deviation':
            return tf.reduce_sum(tf.math.reduce_std(attention, axis=1))
        else:
            return attention

    @classmethod
    def parameter_checks(cls, args):
        """Function to check parameter dimension checks"""

        # CNN dimension check
        cnn_num_layers = int(args.cnn_num_layers)
        cnn_num_filters = [int(x) for x in args.cnn_num_filters.split(',')]
        cnn_filter_sizes = [int(x) for x in args.cnn_filter_sizes.split(',')]
        cnn_stride_length = [int(x) for x in args.cnn_stride_length.split(',')]
        cnn_pool_sizes = [int(x) for x in args.cnn_pool_sizes.split(',')]

        if any([cnn_num_layers != len(cnn_num_filters), cnn_num_layers != len(cnn_filter_sizes),
                cnn_num_layers != len(cnn_stride_length), cnn_num_layers != len(cnn_pool_sizes)]):
            print "CONV ERROR: Cnn parameter sizes do not match"
            print "%s, %s, %s, %s, %s"%(cnn_num_layers, len(cnn_num_filters),
                                        len(cnn_filter_sizes), len(cnn_stride_length),
                                        len(cnn_pool_sizes))
            sys.exit(1)
        for idx, stride in enumerate(cnn_stride_length):
            if args.input_width/stride < cnn_pool_sizes[idx]:
                print "ERROR: Pooling size should be less than the convolution dimension. \
                      Choose pooling size less than %s"%int(args.input_width/stride)
                sys.exit(1)
        print "CNN dimenston check passed"

        # RNN dimenstion check
        if args.encoder_model_type != 'None':
            rnn_num_layers = int(args.encoder_num_layers)
            rnn_hid_dims = [int(x) for x in args.encoder_hid_dims.split(',')]

            if len(rnn_hid_dims) != rnn_num_layers:
                print "RNN ERROR: Number of layers and number of hidden dimensions do not match"
                sys.exit(1)
            print "RNN dimenston check passed"

        # FCN dimenstion check
        fc_num_layers = int(args.num_fc_layers)
        fc_num_neurons = [int(x) for x in args.num_fc_neurons.split(',')]

        if len(fc_num_neurons) != fc_num_layers:
            print "FC ERROR: Number of layers and number of neurons do not match"
            sys.exit(1)
        print "FFN dimenston check passed"

        return True

    def __init__(self, args):
        """Main CNN model function"""
        # Parse hyper-parameters
        if self.parameter_checks(args):
            cnn_num_layers = int(args.cnn_num_layers)
            cnn_num_filters = [int(x) for x in args.cnn_num_filters.split(',')]
            cnn_filter_sizes = [int(x) for x in args.cnn_filter_sizes.split(',')]
            cnn_stride_length = [int(x) for x in args.cnn_stride_length.split(',')]
            cnn_pool_sizes = [int(x) for x in args.cnn_pool_sizes.split(',')]

            fc_num_layers = int(args.num_fc_layers)
            fc_num_neurons = [int(x) for x in args.num_fc_neurons.split(',')]
        else:
            sys.exit(1)

        # Input layer: Input data & droupouts
        self.input_x = tf.placeholder(tf.int32, [None, args.input_width,
                                                 args.input_height], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, args.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        #self.attn_dropout_keep_prob = tf.placeholder(tf.float32, name="attn_dropout_keep_prob")

        self.input_x_expanded = tf.cast(tf.expand_dims(self.input_x, -1), tf.float32)
        print "self.input_x", self.input_x
        print "self.input_x_expanded", self.input_x_expanded

        # Convolution layer
        ip_layer = self.input_x_expanded
        seq_fw, epi_fw, width = args.seq_len, args.epi_len, 0
        for idx in range(cnn_num_layers):
            with tf.variable_scope("conv-layer-%d"%idx):
                print "conv-layer-%d"%idx, ip_layer.shape

                # Convolution layer for sequences & epigenetics
                seq_filter_shape = [cnn_filter_sizes[idx], seq_fw, 1, cnn_num_filters[idx]]
                epi_filter_shape = [cnn_filter_sizes[idx], epi_fw, 1, cnn_num_filters[idx]]

                if idx == 0:
                    seq_conv_relu = self.conv_layer(ip_layer[:, :, 0:seq_fw, :],
                                                    seq_filter_shape, cnn_stride_length[idx],
                                                    args.seq_random_weights, 'seq-conv-%d'%idx)
                    epi_conv_relu = self.conv_layer(ip_layer[:, :, seq_fw:epi_fw+seq_fw, :],
                                                    epi_filter_shape, cnn_stride_length[idx],
                                                    args.epi_random_weights, 'epi-conv-%d'%idx)

                else:
                    seq_conv_relu = self.conv_layer(ip_layer[:, 0:width, :, :],
                                                    seq_filter_shape, cnn_stride_length[idx],
                                                    args.seq_random_weights, 'seq-conv-%d'%idx)
                    epi_conv_relu = self.conv_layer(ip_layer[:, width:(width+width), :, :],
                                                    epi_filter_shape, cnn_stride_length[idx],
                                                    args.epi_random_weights, 'epi-conv-%d'%idx)

                print "Seq conv filter_shape", seq_filter_shape
                print "seq_conv_relu", seq_conv_relu
                print "Epi conv filter_shape", epi_filter_shape
                print "epi_conv_relu", epi_conv_relu

                # Pooling - Max pooling over the given pooling size for sequence
                seq_conv_pool = self.max_pool(seq_conv_relu, pooling_size=cnn_pool_sizes[idx])
                print 'seq_conv_pool', seq_conv_pool

                # Pooling - Max pooling over the given pooling size for epigenetic data
                epi_conv_pool = self.max_pool(epi_conv_relu, pooling_size=cnn_pool_sizes[idx])
                print 'epi_conv_pool', epi_conv_pool

                # Setting inputs for next layer for the next layer
                conv_pool = tf.concat([seq_conv_pool, epi_conv_pool], axis=1)
                print "conv_pool", conv_pool

                ip_layer = tf.transpose(conv_pool, [0, 1, 3, 2])
                seq_fw, epi_fw = cnn_num_filters[idx], cnn_num_filters[idx]
                width = ip_layer.shape[1]/2

        conv_output = tf.squeeze(conv_pool, 2)
        seq_conv_output = tf.squeeze(seq_conv_pool, 2)
        print "seq conv output", seq_conv_output
        epi_conv_output = tf.squeeze(epi_conv_pool, 2)
        print "epi conv output", epi_conv_output
        self.conv_output = conv_output
        print "conv output", conv_output

        # Encoder unit
        if args.add_encoder:            
            print "Encoder Unit"
            enc_dims = [int(x) for x in args.encoder_hid_dims.split(',')]
            enc_layers = len(enc_dims)

            with tf.name_scope("LstmEncoder"):
                # RNN Cell declaration
                seq_fw_cell = rnn.MultiRNNCell([self.rnn_cell(enc_dims[i], args.encoder_model_type,
                                                              self.dropout_keep_prob)
                                                for i in range(enc_layers)], state_is_tuple=True)
                print "seq_fw_cell", seq_fw_cell

                seq_bw_cell = rnn.MultiRNNCell([self.rnn_cell(enc_dims[i], args.encoder_model_type,
                                                              self.dropout_keep_prob)
                                                for i in range(enc_layers)], state_is_tuple=True)
                print "seq_bw_cell", seq_bw_cell

                epi_fw_cell = rnn.MultiRNNCell([self.rnn_cell(enc_dims[i], args.encoder_model_type,
                                                              self.dropout_keep_prob)
                                                for i in range(enc_layers)], state_is_tuple=True)
                print "epi_fw_cell", epi_fw_cell

                epi_bw_cell = rnn.MultiRNNCell([self.rnn_cell(enc_dims[i], args.encoder_model_type,
                                                              self.dropout_keep_prob)
                                                for i in range(enc_layers)], state_is_tuple=True)
                print "epi_bw_cell", epi_bw_cell

                seq_rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(seq_fw_cell, seq_bw_cell,
                                                                 seq_conv_output, dtype=tf.float32,
                                                                 scope="seq_enclstm")

                epi_rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(epi_fw_cell, epi_bw_cell,
                                                                 epi_conv_output, dtype=tf.float32,
                                                                 scope="epi_enclstm")

                self.seq_enc_out = tf.concat(seq_rnn_out, 2)
                self.seq_enc_fin_st = self.seq_enc_out[:, -1, :]
                print "seq_enc_out", self.seq_enc_out

                self.epi_enc_out = tf.concat(epi_rnn_out, 2)
                self.epi_enc_fin_st = self.epi_enc_out[:, -1, :]
                print "epi_enc_out", self.epi_enc_out

                self.enc_out = tf.concat([self.seq_enc_out, self.epi_enc_out], axis=2)
                print "enc_out", self.enc_out

                self.enc_fin_st = self.enc_out[:, -1, :]
                print "enc_fin_st", self.enc_fin_st

        # Attention Unit
        print "Attention Unit"
        attn_dims = [int(x) for x in args.attn_hid_dims.split(',')]
        print 'attn_dims', attn_dims

        with tf.name_scope("Attention"):
            self.seq_attn_score = self.attn_layer(self.seq_enc_out, self.seq_enc_fin_st,
                                                  attn_dims[0], args.attn_score_activator,
                                                  args.attn_wt_randomize, args.add_enc_last_state)
            self.seq_attention = self.attn_probability(self.seq_attn_score, args.attn_estimator,
                                                       args.attn_temp)
            self.seq_cont_vec = self.seq_attention * self.seq_enc_out
            self.seq_cont_vec = tf.reduce_sum(self.seq_cont_vec, axis=1)

            self.epi_attn_score = self.attn_layer(self.epi_enc_out, self.epi_enc_fin_st,
                                                  attn_dims[0], args.attn_score_activator,
                                                  args.attn_wt_randomize, args.add_enc_last_state)
            self.epi_attention = self.attn_probability(self.epi_attn_score, args.attn_estimator,
                                                       args.attn_temp)
            self.epi_cont_vec = self.epi_attention * self.epi_enc_out
            self.epi_cont_vec = tf.reduce_sum(self.epi_cont_vec, axis=1)

            self.attn_score = self.attn_layer(self.enc_out, self.enc_fin_st, attn_dims[0],
                                              args.attn_score_activator,
                                              args.attn_wt_randomize, args.add_enc_last_state)
            self.attention = self.attn_probability(self.attn_score, args.attn_estimator,
                                                   args.attn_temp)
            #self.attention = tf.nn.softmax(self.attn_score, axis=1)
            self.cont_vec = self.attention * self.enc_out
            self.cont_vec = tf.reduce_sum(self.cont_vec, axis=1)

            print "attn_score", self.attn_score
            print "attention", self.attention
            print "cont_vec", self.cont_vec

        # Attention Loss
        self.attn_loss = self.regularize_attention(self.attention, args.attn_regularizer)
        self.attn_loss = args.attn_reg_lambda * self.attn_loss

        # 4: Decoder Unit
        dec_dims = [int(x) for x in args.decoder_hid_dims.split(',')]
        dec_layers = len(dec_dims)
        with tf.variable_scope('Decoder'):
            if args.decoder_model_type == 'fcn':
                dec_out_shape = [int(self.cont_vec.shape[1]), dec_dims[0]]
                self.dec_out, self.dec_wt = self.fc_layer(self.cont_vec, dec_out_shape,
                                                          args.randomize_weights, 'dec_out')
            else:
                temp = tf.expand_dims(self.cont_vec, 1)
                fw_cell = rnn.MultiRNNCell([self.rnn_cell(dec_dims[i], args.decoder_model_type,
                                                          self.dropout_keep_prob)
                                            for i in range(dec_layers)], state_is_tuple=True)
                print "fw_cell", fw_cell
                bw_cell = rnn.MultiRNNCell([self.rnn_cell(dec_dims[i], args.decoder_model_type,
                                                          self.dropout_keep_prob)
                                            for i in range(dec_layers)], state_is_tuple=True)
                print "bw_cell", bw_cell
                rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, temp,
                                                             dtype=tf.float32, scope="declstm")
                self.dec_out = tf.concat(rnn_out, 2)
            print "conv_dec", self.dec_out

        flat_output = tf.contrib.layers.flatten(self.dec_out)
        print "flat_output", flat_output
        flat_output_drop = tf.nn.dropout(flat_output, keep_prob=self.dropout_keep_prob,
                                         name='dropout')
        print "flat_output_drop", flat_output_drop

        # Dense fully connected layer
        self.out_wt_loss = 0.0
        fc_out = []
        for i in range(fc_num_layers):
            with tf.variable_scope('fc-layer-%d'%i):
                if i == 0:
                    n_1 = int(flat_output_drop.shape[1])
                    n_2 = fc_num_neurons[i]
                    fcn_ip_layer = flat_output_drop
                else:
                    n_1 = fc_num_neurons[(i-1)]
                    n_2 = fc_num_neurons[i]
                    fcn_ip_layer = fc_out
                fc_shape = [n_1, n_2]
                print 'fc_shape-%d'%(i), fc_shape
                fc_out, fc_weight = self.fc_layer(fcn_ip_layer, fc_shape,
                                                  args.randomize_weights, 'fc-%d'%i)
                self.out_wt_loss += tf.nn.l2_loss(fc_weight)
                print 'fc_layer', fc_out
                print 'fc_weight_shape', fc_weight.shape

                if fc_num_layers >= 1 and i < (fc_num_layers - 1):
                    fc_out = tf.nn.relu(fc_out)
        self.fc_out_layer = fc_out

        # Final prediction scores
        with tf.variable_scope("output"):
            output_weight = tf.get_variable(shape=[fc_num_neurons[-1], args.num_classes],
                                            initializer=tf.contrib.layers.xavier_initializer(),
                                            name="out_wt")
            output_bias = tf.Variable(tf.constant(0.1, shape=[args.num_classes]), name="out_b")
            matmul = tf.nn.xw_plus_b(fc_out, output_weight, output_bias, name="matmul")
            self.scores = matmul

            print 'output_weight_shape', output_weight
            print self.scores.shape
        self.out_wt_loss += tf.nn.l2_loss(output_weight)
        self.out_wt_loss = args.out_reg_lambda * self.out_wt_loss

        # Calculate losses - cross-entropy loss for classification/ mse for regression
        with tf.variable_scope("loss"):
            if args.task == 'regression':
                self.error = tf.losses.mean_squared_error(self.input_y, self.scores)
            else:
                self.error = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                                     labels=self.input_y)
        self.loss = tf.reduce_mean(self.error) + self.out_wt_loss + self.attn_loss

        # Optimization
        with tf.variable_scope("optimize"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learn_rate).minimize(self.loss)
        print "optimizer", self.optimizer
