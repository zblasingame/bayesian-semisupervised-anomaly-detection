"""
A module for anomalous Bayesian prediction

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import tensorflow as tf
import numpy as np
import models.models
from sklearn.cluster import KMeans


def gen_kde(n_inputs, special_ops, reg_param=0.01, n_kernels=100):
    """Creates function that generates KDE model.

    Parameters
    ----------
    n_inputs : int
        Number of inputs.
    special_ops : dict
        Dictionary where special operations can be added.
    reg_param : float, optional
        L2 regularization parameter, default 0.01.
    n_kernels : int optional
        Number of kernels used in KDE, default 100.

    Returns
    -------
    kde : function
        Function that generates KDE model.

        Parameters
        ----------
        X : tf.Tensors
            Input tensor of shape [batch_size, n_inputs].

        Returns
        -------
        loss, pmf : tf.Tensor
            Tensors holding the loss and probability mass function.
    """
    def _kde(X):
        k_shape = tf.get_variable(
            'k_shape',
            n_kernels,
            initializer=tf.random_normal_initializer(mean=1, stddev=0.1),
            regularizer=tf.contrib.layers.l2_regularizer(reg_param),
            constraint=lambda x: tf.clip_by_value(x, 1e-9, np.infty)
        )

        # k_weights = tf.get_variable(
        #     'k_weights',
        #     n_kernels,
        #     initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
        #     regularizer=tf.contrib.layers.l2_regularizer(reg_param)
        # )

        k_locs = tf.get_variable(
            'k_locs',
            [n_kernels, n_inputs],
            trainable=False
            # initializer=tf.random_normal_initializer(mean=0.5, stddev=0.25)
        )

        # Add cluster assignment to special ops
        def op(sess, x):
            kmeans = KMeans(n_clusters=n_kernels, n_jobs=-2)
            kmeans.fit(x)
            sess.run(k_locs.assign(kmeans.cluster_centers_))

        special_ops['train_pmf/k_locs_assign'] = op

        dist = tf.transpose(
            tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(
                X,
                tf.expand_dims(k_locs, 1)
            )), axis=2)),
            perm=[1,0]
        )

        kernel = tf.div(
            tf.exp(-tf.square(dist) / k_shape),
            tf.sqrt(np.pi * k_shape)
        )

        # Kernel shape [n_kernels, batch_size, n_inputs]
        # Weighted kernel
        # pmf = tf.reduce_sum(
        #     tf.multiply(tf.nn.softmax(k_weights), kernel),
        #     axis=1
        # )
        pmf = tf.reduce_mean(kernel, axis=1)
        eps = 1e-15
        pmf = tf.clip_by_value(pmf, eps, 1-eps)
        # pmf = tf.Print(pmf, [k_locs])

        # print('Shapes')
        # print(kernel)
        # print(X)
        # print(dist)
        # print(pmf)
        # print('-'*79)

        return -tf.reduce_mean(tf.log(pmf)), pmf

    return _kde


def gen_neural_net_pmf(layers, activations, reg_param=0.01, drop_prob=False):
    """Creates function that generates neural net to estimate a pmf.

    Parameters
    ----------
    layers: list
        List of sizes for each layer including input layer.
        Note the size of the input and output layer must match.
    activations : list
        List of activations, note `len(activations) == len(layers) - 1`.
    reg_param : float, optional
        L2 regularization parameter, default 0.01.
    drop_prob : bool or tf.Variable, optional
        Probability of dropout layer. If False the
        dropout layer is not created. Otherwise, the
        value must be of type tf.Variable, default False.

    Returns
    -------
    ae : function
        Function that generates an auto encoder.

        Parameters
        ----------
        X : tf.Tensors
            Input tensor of shape [batch_size, n_inputs].

        Returns
        -------
        loss, scores : tf.Tensor
            Tensors holding the output of the neural net and the loss.
    """
    def _pmf(X):
        net = models.models.gen_fc_neural_net(
            layers,
            activations,
            reg_param,
            drop_prob
        )

        pmf = net(X)
        eps = 1e-15
        pmf = tf.clip_by_value(pmf, eps, 1-eps)

        pmf = tf.Print(pmf, [pmf, tf.reduce_sum(pmf)])
        return -tf.reduce_mean(tf.log(pmf)), pmf

    return _pmf
