import argparse
import csv
import os
from time import time

import numpy as np
import scipy.io as scio
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import plot_model
from sklearn import metrics
from sklearn.cluster import KMeans

from Data.pre_training import pre_train
from Data.synthetic_plots import load_synthetic, plot_points
from utils import target_distribution, ClusteringLayer, cluster_accuracy, get_acc_nmi_ari

# hyper parameters
N_CLUSTERS = 10
UPDATE_INTERVAL = 100
FINAL_WEIGHT_DIR = '../final_weights/dec'
MAX_ITER = 200


def load_mnist():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    # converting to (784, ) shape
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.0)  # normalize pixel value between 0 and 1
    print('MNIST samples shape : {}'.format(x.shape))
    return x, y


def load_har():
    data = scio.loadmat('../Data/har/HAR.mat')
    x = data['X']
    x = x.astype('float32')
    y = data['Y'] - 1
    x = x[:10200]
    y = y[:10200]
    y = y.reshape((10200,))
    print('HHAR samples shape : {}'.format(x.shape))
    return x, y


def autoencoder_model(input_dim, act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    return:
        Model of autoencoder
    """
    x = Input(shape=(input_dim,), name='input')

    h = x

    # internal layers in encoder
    h = Dense(500, activation=act, name='encoder_0')(h)

    h = Dense(500, activation=act, name='encoder_1')(h)

    h = Dense(2000, activation=act, name='encoder_2')(h)

    # hidden layer
    h = Dense(10, name='encoder_3')(h)  # hidden layer, features are extracted from here

    # internal layers in decoder

    h = Dense(2000, activation=act, name='decoder_3')(h)

    h = Dense(500, activation=act, name='decoder_2')(h)

    h = Dense(500, activation=act, name='decoder_1')(h)

    # output
    h = Dense(input_dim, name='decoder_0')(h)

    return Model(inputs=x, outputs=h)


def autoencoder_model_syn(input_dim, act='relu'):
    """
    Fully connected auto-encoder model for synthetic data, symmetric.
    return:
        Model of autoencoder
    """
    x = Input(shape=(input_dim,), name='input')

    h = x

    # internal layers in encoder
    h = Dense(50, activation=act, name='encoder_0')(h)

    h = Dense(10, activation=act, name='encoder_1')(h)

    # hidden layer
    h = Dense(2, activation=act, name='encoder_2')(h)

    # internal layers in decoder

    h = Dense(10, activation=act, name='decoder_2')(h)

    h = Dense(50, activation=act, name='decoder_1')(h)

    # output
    h = Dense(input_dim, name='decoder_0')(h)

    return Model(inputs=x, outputs=h)


class DEC(object):
    def __init__(self,
                 input_dim,
                 n_clusters=10,
                 alpha=1.0,
                 data_set=None):

        super(DEC, self).__init__()
        self.input_dim = input_dim

        self.n_clusters = n_clusters
        self.alpha = alpha
        if data_set == 'synthetic':
            self.autoencoder = autoencoder_model_syn(self.input_dim)
            plot_model(self.autoencoder, to_file='../auto_encoder_' + args.dataset + '.png', show_shapes=True)
        else:
            self.autoencoder = autoencoder_model(self.input_dim)
            plot_model(self.autoencoder, to_file='../auto_encoder_' + args.dataset + '.png', show_shapes=True)

        self.data_set = data_set

    def initialize(self, optimizer, ae_weights=None):
        # load pretrained weights of autoencoder
        self.autoencoder.load_weights(ae_weights)
        print('Pretrained AE weights are loaded successfully from {}'.format(ae_weights))

        if self.data_set == 'synthetic':
            hidden = self.autoencoder.get_layer(name='encoder_2').output
            self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)
        else:
            hidden = self.autoencoder.get_layer(name='encoder_3').output
            self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)

        # prepare DEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
        self.model = Model(inputs=self.autoencoder.input, outputs=[clustering_layer])
        self.model.compile(loss='kld', optimizer=optimizer)

    def extract_feature(self, x):  # extract features from before clustering layer
        if self.data_set == 'synthetic':
            encoder = Model(self.model.input, self.model.get_layer('encoder_2').output)
        else:
            encoder = Model(self.model.input, self.model.get_layer('encoder_3').output)
        return encoder.predict(x)

    def clustering(self, x, y=None,
                   update_interval=100,
                   maxiter=200,
                   save_dir='./results/dec'):

        print('Updating auxilliary distribution after %d iterations' % update_interval)
        save_interval = 10  # 10 epochs
        print('Saving models after %d iterations' % save_interval)

        # initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        k_means = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        y_pred = k_means.fit_predict(self.encoder.predict(x))
        self.model.get_layer(name='clustering').set_weights([k_means.cluster_centers_])

        # logging file
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        log_file = open(save_dir + '/dec_log.csv', 'w')
        log_writer = csv.DictWriter(log_file, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L'])
        log_writer.writeheader()

        loss = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)

                acc, nmi, ari = get_acc_nmi_ari(y, y_pred)
                loss = np.round(loss, 5)
                log_dict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss)
                log_writer.writerow(log_dict)
                print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

            # training on whole data
            loss = self.model.train_on_batch(x=x,
                                             y=p)

            # save intermediate model
            if ite % save_interval == 0:
                # save DEC model checkpoints
                print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
                self.model.save(save_dir + '/DEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        log_file.close()
        print('saving model to:', save_dir + '/DEC_model_final.h5')
        self.model.save(save_dir + '/DEC_model_final.h5')

        return y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', default='mnist', choices=['mnist', 'har', 'synthetic'])
    parser.add_argument('--ae_weights', default=None, help='This argument must be given')
    parser.add_argument('--update_interval', default=1, type=int)
    args = parser.parse_args()
    print(args)

    # loading dataset
    optimizer = SGD(lr=0.1, momentum=0.99)
    if args.dataset == 'mnist':  # recommends: n_clusters=10, update_interval=100
        x, y = load_mnist()
        optimizer = 'adam'
        N_CLUSTERS = 10
    elif args.dataset == 'har':  # recommends: n_clusters=6, update_interval=100
        x, y = load_har()
        N_CLUSTERS = 6
    elif args.dataset == 'synthetic':  # recommends: n_clusters=4, update_interval=100
        x, y = load_synthetic()
        N_CLUSTERS = 4

    # prepare the DEC model
    dec = DEC(input_dim=x.shape[-1], n_clusters=N_CLUSTERS, data_set=args.dataset)

    # if pre_training not done , then doing pre_training
    if not os.path.isfile(args.ae_weights):
        pre_train(x, args.dataset, x.shape[-1], args.ae_weights)

    # initializing with auto_encoder with pre_training weights
    dec.initialize(optimizer=optimizer,
                   ae_weights=args.ae_weights)

    # plotting model description
    plot_model(dec.model, to_file='../dec_model_' + args.dataset + '.png', show_shapes=True)
    dec.model.summary()
    t0 = time()
    final_weights_dir = FINAL_WEIGHT_DIR + '/' + args.dataset

    # if training is already done , initialize the model from saved weights
    if os.path.isfile(final_weights_dir + '/DEC_model_final.h5'):
        dec.model.load_weights(final_weights_dir + '/DEC_model_final.h5')
        kmeans = KMeans(n_clusters=dec.n_clusters, n_init=20, random_state=42)
        y_pred = kmeans.fit_predict(dec.encoder.predict(x))

        if args.dataset == 'synthetic':
            intermediate_output = dec.extract_feature(x)
            plot_points(intermediate_output, y)

        print('Final accuracy is : {}'.format(cluster_accuracy(y, y_pred)))
        exit(0)

    y_pred = dec.clustering(x, y=y, maxiter=MAX_ITER,
                            update_interval=args.update_interval, save_dir=final_weights_dir)
    print('Final accuracy is : {}'.format(cluster_accuracy(y, y_pred)))
    print('Total clustering time: {} minutes'.format((time() - t0) / 60))
