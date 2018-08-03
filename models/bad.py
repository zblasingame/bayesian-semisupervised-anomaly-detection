"""General Bayesian Anomaly Detection Module

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import tensorflow as tf
import numpy as np
import utils.datasets as ds


class BAD:
    """Bayesian Anomaly Detection.

    Attributes
    ----------
    X : tf.Tensor
        Tensor representing input features.
    Y : tf.Tensor
        Tensor representing labels.
    model : function
        Function that generates tensorflow model.
        Takes in tensor X and outputs a score for a sample.
        Additionally, the model returns a loss function for training
        if the loss is False then the training procedure for the model
        is skipped.
    model_name : str
        Name of the `model`.
    pmf_model : function
        Function that generates tensorflow model.
        Takes output of `model` and outputs its probability distribution.
    pmf_name : str
        Name of `pmf_model`.
    model_params : dict, optional
        Dictionary with parameters specific to `model`.
        n_epochs : int, optional
            Number of epochs, default 100.
        display_step : int, optional
            How many epochs until output from debugger, default 10.
        batch_size : int, optional
            Size of mini-batches, default 100.
        l_rate : float, optional
            Learning rate, default .001.
    pmf_params : dict, optional
        Dictionary with parameters specific to `pmf_model`.
        n_epochs : int, optional
            Number of epochs, default 10.
        display_step : int, optional
            How many epochs until output from debugger, default 1.
        batch_size : int, optional
            Size of mini-batches, default 100.
        l_rate : float, optional
            Learning rate, default 0.001.
    p_isnorm : float, optional
        Probability sample is normal, default 0.5.
    normalize : bool, optional
        Flag to normalize input features, default True.
    debug : bool, optional
        Flag to display debugging information, default False.
    always_train_pmf : bool, optional
        Flag to always train pmf after training model, default True.
    save_path : str, optional
        Path to directory where models are saved, default '.models'
    """

    def __init__(self, X, Y, model, model_name, pmf_model, pmf_name, **kwargs):
        defaults = {
            'p_isnorm': 0.5,
            'normalize': True,
            'debug': False,
            'always_train_pmf': True,
            'save_path': '.models'
        }

        vars(self).update({p: kwargs.get(p, d) for p, d in defaults.items()})

        self.model_params = {
            'n_epochs': 100,
            'display_step': 10,
            'batch_size': 100,
            'l_rate': .001
        }

        self.pmf_params = {
            'n_epochs': 10,
            'display_step': 1,
            'batch_size': 100,
            'l_rate': .001
        }

        if 'model_params' in kwargs:
            dt = kwargs['model_params']
            self.model_params.update(
                {p: dt.get(p, d) for p, d in self.model_params.items()}
            )

        if 'pmf_params' in kwargs:
            dt = kwargs['pmf_params']
            self.pmf_params.update(
                {p: dt.get(p, d) for p, d in self.pmf_params.items()}
            )

        self.model_name = model_name
        self.pmf_name = pmf_name

        self.X = X
        self.Y = Y

        self.feature_min = tf.Variable(
            np.zeros(X.get_shape()[1:]),
            dtype=tf.float32
        )

        self.feature_max = tf.Variable(
            np.zeros(X.get_shape()[1:]),
            dtype=tf.float32
        )

        self.p_uniform = tf.Variable(0, dtype=tf.float32)

        with tf.variable_scope(model_name):
            self.model_loss, self.scores = model(X)

        with tf.variable_scope(pmf_name):
            # _in = tf.concat([tf.expand_dims(self.scores,1), X], 1)
            _in = tf.expand_dims(self.scores, 1)
            self.bayes_loss, pmf = pmf_model(_in)

        # Find posterior
        uniform = tf.ones_like(pmf) * self.p_uniform

        marginal = tf.add(
            self.p_isnorm * pmf,
            (1 - self.p_isnorm) * uniform
        )

        self.posterior = self.p_isnorm * pmf / marginal

        if self.model_loss is not False:
            self.model_loss += tf.add_n(tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES,
                scope=model_name
            ))
            self.model_opt = tf.train.AdamOptimizer(
                self.model_params['l_rate']
            ).minimize(
                self.model_loss,
                var_list=tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope=model_name
                )
            )

        self.bayes_loss += tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES,
            scope=pmf_name
        ))

        self.bayes_opt = tf.train.AdamOptimizer(
            self.pmf_params['l_rate']
        ).minimize(
            self.bayes_loss,
            var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=pmf_name
            )
        )

        pred_labels = tf.cast(tf.greater_equal(
            self.posterior,
            0.5
        ), tf.int32)

        self.confusion_matrix = tf.confusion_matrix(
            self.Y,
            pred_labels,
            num_classes=2
        )

        self.accuracy = tf.reduce_mean(
            tf.to_float(tf.equal(pred_labels, self.Y))
        )

        # Variable ops
        self.init_op = tf.global_variables_initializer()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.saver = tf.train.Saver()
        self.save_path += '/{}.ckpt'

    def train_model(self, X):
        """Train the model.

        Parameters
        ----------
        X : np.ndarray
            Input features, with shape like `self.X`.
        """

        if self.model_loss is False:
            if self.always_train_pmf:
                self.train_pmf(X, ignore_norm=False, restore_model=False)
            return

        training_size = X.shape[0]

        assert self.model_params['batch_size'] < training_size, (
            'batch size is larger than number of samples'
        )

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)

            if self.normalize:
                _min = X.min(axis=0)
                _max = X.max(axis=0)
                X = ds.rescale(X, _min, _max, -1, 1)

                sess.run(self.feature_min.assign(_min))
                sess.run(self.feature_max.assign(_max))

            batch = ds.random_batcher([X], self.model_params['batch_size'])

            self.print('Training {}'.format(self.model_name))
            self.print('Epoch | Loss')

            for epoch in range(self.model_params['n_epochs']):
                batch_x, = next(batch)

                _, l = sess.run(
                    [self.model_opt, self.model_loss],
                    feed_dict={
                        self.X: batch_x
                    }
                )

                if epoch % self.model_params['display_step'] == 0:
                    self.print('{0:05} | {1:7.5f}'.format(epoch+1, l))

            self.print('Finished training {}'.format(self.model_name))

            # save model
            save_path = self.saver.save(
                sess,
                self.save_path.format(self.model_name)
            )
            self.print('Model saved in file: {}'.format(save_path))

        if self.always_train_pmf:
            self.train_pmf(X, ignore_norm=True)

    def train_pmf(self, X, ignore_norm=False, restore_model=True):
        """Train the density estimator.

        Parameters
        ----------
        X : np.ndarray
            Input features, with shape like `self.X`.
        ignore_norm : bool, optional
            Ignore normalization, default is False.
        restore_model : bool, optional
            Restore scoring model, default is True.
        """

        training_size = X.shape[0]

        assert self.pmf_params['batch_size'] < training_size, (
            'Batch size is larger than number of samples'
        )

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)

            if restore_model:
                self.saver.restore(
                    sess,
                    self.save_path.format(self.model_name)
                )

            if self.normalize and not ignore_norm:
                _min = X.min(axis=0)
                _max = X.max(axis=0)
                X = ds.rescale(X, _min, _max, -1, 1)

                sess.run(self.feature_min.assign(_min))
                sess.run(self.feature_max.assign(_max))

            # Find p_uniform
            scores = sess.run(
                self.scores,
                feed_dict={
                    self.X: X
                }
            )

            sess.run(self.p_uniform.assign(1/np.unique(scores).shape[0]))

            batch = ds.random_batcher([X], self.pmf_params['batch_size'])

            self.print('Training {}'.format(self.pmf_name))
            self.print('Epoch | Loss')

            for epoch in range(self.pmf_params['n_epochs']):
                batch_x, = next(batch)

                _, l = sess.run(
                    [self.bayes_opt, self.bayes_loss],
                    feed_dict={
                        self.X: batch_x
                    }
                )

                if epoch % self.pmf_params['display_step'] == 0:
                    self.print('{0:05} | {1:7.5f}'.format(epoch+1, l))

            self.print('Finished training density estimator')

            # save model
            save_path = self.saver.save(
                sess,
                self.save_path.format(self.pmf_name)
            )
            self.print('Model saved in file: {}'.format(save_path))

    def test(self, X, Y, restore_model=True):
        """Evaluate model performance.

        Parameters
        ----------
        X : np.ndarray
            Input features, with shape like `self.X`.
        Y : np.ndarray
            Labels for each sample.
        restore_model : bool, optional
            Restore scoring model, default is True.

        Returns
        -------
        accuracy : float
            Classification accuracy of model.
        c_mat : np.ndarray
            Confusion matrix.
        """

        with tf.Session(config=self.config) as sess:
            if restore_model:
                self.saver.restore(
                    sess,
                    self.save_path.format(self.model_name)
                )

            self.saver.restore(sess, self.save_path.format(self.pmf_name))

            # normalize data
            if self.normalize:
                _min = self.feature_min.eval()
                _max = self.feature_max.eval()
                X = ds.rescale(X, _min, _max, -1, 1)

            acc, mat = sess.run(
                [self.accuracy, self.confusion_matrix],
                feed_dict={
                    self.X: X,
                    self.Y: Y
                }
            )

            self.print('Accuracy = {:.3f}%'.format(acc * 100))
            self.print(mat)

            return acc * 100, mat

    def print(self, val):
        if self.debug:
            print(val)
