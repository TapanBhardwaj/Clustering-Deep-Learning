from time import time
import numpy as np

import tensorflow as tf
from keras.engine.topology import Layer,InputSpec
from keras.layers import Dense,Input
from keras.models import Model
from keras.optimizers import SGD
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import keras.backend as K
from sklearn.cluster import KMeans
from sklearn import metrics
import pickle
np.random.seed(0)

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def autoencoder(dims, act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        Model of autoencoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        h = Dense(dims[i], activation=act, name='decoder_%d' % i)(h)

    # output
    h = Dense(dims[0], name='decoder_0')(h)

    return Model(inputs=x, outputs=h)

class DCN(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=1000):


        super(DCN, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.autoencoder = autoencoder(self.dims)

    def initialize_model(self, ae_weights=None, gamma=0.1, optimizer='adam'):
        if ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('Pretrained AE weights are loaded successfully.')
        else:
            print('ae_weights must be given. E.g.')
            print('python DCN_mnist.py mnist --ae_weights weights.h5')
            exit()

        hidden = self.autoencoder.get_layer(name='encoder_%d' % (self.n_stacks - 1)).output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)

        # prepare DCN model
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[self.encoder.output, self.autoencoder.output])
        self.model.compile(loss={'encoder_3': 'mse', 'decoder_0': 'mse'},
                           loss_weights=[gamma, 1],
                           optimizer=optimizer)


    def load_weights(self, weights_path):  # load weights of IDEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        encoder = Model(self.model.input, self.model.get_layer('encoder_%d' % (self.n_stacks - 1)).output)
        return encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    def clustering(self, x, y=None,
                   tol=1e-3,
                   update_interval=70,
                   maxiter=200,
                   save_dir='./results/DCN_har'):

        print('Update interval', update_interval)
        save_interval = 50
        print('Save interval', save_interval)

        # initialize cluster centers using k-means
        # print('Initializing cluster centers with k-means.')
        # kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        # y_pred = kmeans.fit_predict(self.encoder.predict(x))

        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/dcn_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            q, _ = self.model.predict(x, verbose=0)#actual representation in k-means friendly space
            km=KMeans(n_clusters=self.n_clusters).fit(q)
            centers=km.cluster_centers_
            labels=km.labels_
            for i in range(q.shape[0]):
                q[i,:]=centers[labels[i],:]

            if y is not None:
                acc = np.round(cluster_acc(y, labels), 5)
                nmi = np.round(metrics.normalized_mutual_info_score(y, labels), 5)
                ari = np.round(metrics.adjusted_rand_score(y, labels), 5)
                loss = np.round(loss, 5)
                logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                logwriter.writerow(logdict)
                print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

            # loss=self.model.fit(x=x,y=[q,x],verbose=1)
            # train on batch
            loss=self.model.train_on_batch(x=x,y=[q,x])


            # save intermediate model
            if ite % save_interval == 0:
                # save DCN model checkpoints
                print('saving model to:', save_dir + '/DCN_model_mnist_' + str(ite) + '.h5')
                self.model.save(save_dir + '/DCN_model_mnist' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/DCN_model_final.h5')
        self.model.save(save_dir + '/DCN_model_final.h5')

        return labels


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('dataset', default='har')
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=1000, type=int)
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=70, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default='ae_weights/har_ae_weights.h5')
    parser.add_argument('--save_dir', default='results_har/DCN_har')
    args = parser.parse_args()
    print(args)

    # load dataset
    optimizer = SGD(lr=0.1, momentum=0.99)


    with open('har_data.pkl', 'rb') as f:
        x, y = pickle.load(f)
    print('Data loaded successfully')
    print(x.shape)
    print(y.shape)

    # prepare the IDEC model
    dcn = DCN(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=args.n_clusters, batch_size=args.batch_size)
    dcn.initialize_model(ae_weights=args.ae_weights, gamma=args.gamma, optimizer=optimizer)
    # plot_model(idec.model, to_file='dcn_model.png', show_shapes=True)

    dcn.model.summary()

    # begin clustering, time not include pretraining part.
    t0 = time()
    y_pred = dcn.clustering(x=x, y=y, tol=args.tol, maxiter=args.maxiter,
                             update_interval=args.update_interval, save_dir=args.save_dir)
    print('acc:', cluster_acc(y, y_pred))
    print('nmi',metrics.normalized_mutual_info_score(y, y_pred))
    print('rand',metrics.adjusted_rand_score(y, y_pred))
    print('clustering time: ', (time() - t0))
