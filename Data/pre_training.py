import os

from keras.layers import Dense, Input
from keras.models import Model

from Data.synthetic_plots import load_mnist, load_har
from Data.synthetic_plots import load_synthetic


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


def pre_train(x, data_set, input_dim, file_path, batch_size=256, epochs=200, optimizer='adam'):
    if data_set == 'synthetic':
        autoencoder = autoencoder_model_syn(input_dim)
    else:
        autoencoder = autoencoder_model(input_dim)
    print('...Pre_training...')
    autoencoder.compile(loss='mse', optimizer=optimizer)  # SGD(lr=0.01, momentum=0.9),
    autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, verbose=2)

    autoencoder.save_weights(file_path)
    print('Pre_trained weights are saved to ' + file_path)


if __name__ == '__main__':
    if not os.path.exists('../ae_weights'):
        os.makedirs('../ae_weights')
    for data_set in ['har', 'synthetic', 'mnist']:
        file_path = '../ae_weights/' + data_set + '_ae_weights.h5'

        if data_set == 'mnist':
            x, _ = load_mnist()
        elif data_set == 'har':
            x, _ = load_har()
        elif data_set == 'synthetic':
            x, _ = load_synthetic()

        pre_train(x, data_set, x.shape[-1], file_path)
