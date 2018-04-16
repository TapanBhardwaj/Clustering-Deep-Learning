import argparse
import csv
import os
from time import time

import numpy as np
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import plot_model
from sklearn.cluster import KMeans

from DEC.dec import autoencoder_model, cluster_accuracy, plot_points
from DEC.dec import load_synthetic, load_mnist, load_har, autoencoder_model_syn
from Data.pre_training import pre_train
from utils import target_distribution, ClusteringLayer, get_acc_nmi_ari

# hyper parameters
N_CLUSTERS = 10
UPDATE_INTERVAL = 100
FINAL_WEIGHT_DIR = '../final_weights/idec'
MAX_ITER = 200


class IDEC(object):
    def __init__(self,
                 input_dim,
                 n_clusters=10,
                 alpha=1.0,
                 data_set=None):

        super(IDEC, self).__init__()
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.data_set = data_set
        if data_set == 'synthetic':
            self.autoencoder = autoencoder_model_syn(self.input_dim)
        else:
            self.autoencoder = autoencoder_model(self.input_dim)

    def initialize(self, ae_weights=None, gamma=0.1, optimizer='adam'):
        # load pretrained weights of autoencoder
        self.autoencoder.load_weights(ae_weights)
        print('Pretrained AE weights are loaded successfully from {}'.format(ae_weights))

        if self.data_set == 'synthetic':
            hidden = self.autoencoder.get_layer(name='encoder_2').output
            self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)
        else:
            hidden = self.autoencoder.get_layer(name='encoder_3').output
            self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)

        # prepare IDEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[clustering_layer, self.autoencoder.output])
        self.model.compile(loss={'clustering': 'kld', 'decoder_0': 'mse'},
                           loss_weights=[gamma, 1],
                           optimizer=optimizer)

    def extract_feature(self, x):  # extract features from before clustering layer
        if self.data_set == 'synthetic':
            encoder = Model(self.model.input, self.model.get_layer('encoder_2').output)
        else:
            encoder = Model(self.model.input, self.model.get_layer('encoder_3').output)
        return encoder.predict(x)

    def clustering(self, x, y=None,
                   update_interval=140,
                   maxiter=200,
                   save_dir='./results/idec'):

        print('Updating auxilliary distribution after %d iterations' % update_interval)
        save_interval = 10  # 10 epochs
        print('Saving models after %d iterations' % save_interval)

        # initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        k_means = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = k_means.fit_predict(self.encoder.predict(x))
        self.model.get_layer(name='clustering').set_weights([k_means.cluster_centers_])

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        log_file = open(save_dir + '/idec_log.csv', 'w')
        log_writer = csv.DictWriter(log_file, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        log_writer.writeheader()

        loss = [0, 0, 0]
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)

                acc, nmi, ari = get_acc_nmi_ari(y, y_pred)
                loss = np.round(loss, 5)
                log_dict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                log_writer.writerow(log_dict)
                print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

            # training on whole data
            loss = self.model.train_on_batch(x=x,
                                             y=[p, x])

            # save intermediate model
            if ite % save_interval == 0:
                # save IDEC model checkpoints
                print('saving model to:', save_dir + '/IDEC_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/IDEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        log_file.close()
        print('saving model to:', save_dir + '/IDEC_model_final.h5')
        self.model.save_weights(save_dir + '/IDEC_model_final.h5')

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

    # prepare the IDEC model
    idec = IDEC(input_dim=x.shape[-1], n_clusters=N_CLUSTERS, data_set=args.dataset)

    # if pre_training not done , then doing pre_training
    if not os.path.isfile(args.ae_weights):
        pre_train(x, args.dataset, x.shape[-1], args.ae_weights)

    # initializing with auto_encoder with pre_training weights
    idec.initialize(optimizer=optimizer,
                    ae_weights=args.ae_weights)

    # plotting model description
    plot_model(idec.model, to_file='../idec_model_' + args.dataset + '.png', show_shapes=True)
    idec.model.summary()
    t0 = time()
    final_weights_dir = FINAL_WEIGHT_DIR + '/' + args.dataset + '3'

    # if training is already done , initialize the model from saved weights
    if os.path.isfile(final_weights_dir + '/IDEC_model_final.h5'):
        idec.model.load_weights(final_weights_dir + '/IDEC_model_final.h5')

        k_means = KMeans(n_clusters=idec.n_clusters, n_init=20, random_state=42)
        y_pred = k_means.fit_predict(idec.encoder.predict(x))

        if args.dataset == 'synthetic':
            intermediate_output = idec.extract_feature(x)
            plot_points(intermediate_output, y)

        print('Final accuracy is : {}'.format(cluster_accuracy(y, y_pred)))
        exit(0)

    y_pred = idec.clustering(x, y=y, maxiter=MAX_ITER,
                             update_interval=args.update_interval, save_dir=final_weights_dir)
    print('Final accuracy is : {}'.format(cluster_accuracy(y, y_pred)))
    print('Total clustering time: {} minutes'.format((time() - t0) / 60))
