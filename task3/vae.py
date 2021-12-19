"""
adapted from: (19.12.2021)
"""
import keras
import tensorflow as tf
from keras import layers
from keras.layers import Input, Flatten, Dense, Lambda, Reshape
from keras.models import Model
from keras import backend as K
import numpy as np


class Sampling(layers.Layer):
    """
    Use (mu, sigma) to sample the encoding vector z
    """

    def call(self, inputs):
        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        eps = K.random_normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * sigma) * eps

class Encoder(layers.Layer):
    """
    Encodes input of any shape to triplet (mu, sigma, z)
    """

    def __init__(self, latent_dim=2, hidden_size=256, input_shape=(28, 28, 1), name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.input = Input(shape=input_shape, name='encoder_input')
        self.flatten = Flatten()
        self.dense_1 = Dense(hidden_size, activation="relu")
        self.dense_2 = Dense(hidden_size, activation="relu")
        self.mu = Dense(latent_dim, name='latent_mu')
        self.sigma = Dense(latent_dim, name='latent_sigma')
        self.sampling = Sampling()

    def call(self, inputs):
        i = self.input(inputs)
        x = self.flatten(i)
        x = self.dense_1(x)
        x = self.dense_2(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        z = self.sampling((mu, sigma))
        return mu, sigma, z

class Decoder(layers.Layer):
    """
    Converts the encoded vector z back to reasonable data
    """

    def __init__(self, latent_dim=2, hidden_size=256, input_shape=(28,28,1), name="decoder", **kwargs ):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.input = Input(shape=(latent_dim,), name='decoder_input')
        self.dense_1 = Dense(hidden_size, activation="relu")
        self.dense_2 = Dense(hidden_size, activation="relu")
        self.dense_3 = Dense(np.prod(list(input_shape), activation="relu"))
        self.reshape = Reshape(input_shape)

    def call(self, inputs):
        i = self.input(inputs)
        x = self.dense_1(i)
        x = self.dense_2(x)
        x = self.dense_3(x)
        reconstruction = self.reshape(x)
        return reconstruction

class VariationalAutoEncoder(keras.Model):
    """
    Combines the encoder and decoder into an end-to-end model for training
    """

    def __init__(
            self,
            input_shape=(28,28,1),
            latent_dim=2,
            hidden_size=256,
            name="vae",
            **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.input_shape = input_shape
        self.encoder = Encoder(latent_dim=latent_dim, hidden_size=hidden_size, input_shape=input_shape)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_size=hidden_size, input_shape=input_shape)

    def call(self, inputs):
        mu, sigma, z = self.encoder
        reconstruction = self.decoder(z)
        return reconstruction
