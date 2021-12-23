"""
Module to implement a Variational Autoencoder.
adapted from: https://keras.io/guides/making_new_layers_and_models_via_subclassing/ (19.12.2021)
"""
import keras
import tensorflow as tf
from keras import layers
from keras.layers import Flatten, Dense, Reshape
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

class Encoder(keras.Model):
    """
    Encodes input of any input_shape to triplet (mu, sigma, z)
    """

    def __init__(self, latent_dim=2, hidden_size=256, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.flatten = Flatten()
        self.dense_1 = Dense(hidden_size, activation="relu")
        self.dense_2 = Dense(hidden_size, activation="relu")
        self.dense_mu = Dense(latent_dim, name='latent_mu')
        self.dense_sigma = Dense(latent_dim, name='latent_sigma')
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        mu = self.dense_mu(x)
        sigma = self.dense_sigma(x)
        z = self.sampling((mu, sigma))
        return mu, sigma, z

class Decoder(keras.Model):
    """
    Converts the encoded vector z back to reasonable data
    """

    def __init__(self, hidden_size=256, input_shape=(28, 28, 1), name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_1 = Dense(hidden_size, activation="relu")
        self.dense_2 = Dense(hidden_size, activation="relu")
        self.dense_3 = Dense(units=np.prod(list(input_shape)), activation="relu")
        self.reshape = Reshape(target_shape=input_shape)

    def call(self, inputs):
        x = self.dense_1(inputs)
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
            input_shape=(28, 28, 1),
            latent_dim=2,
            hidden_size=256,
            name="vae",
            **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = input_shape
        self.encoder = Encoder(latent_dim=latent_dim, hidden_size=hidden_size)
        self.decoder = Decoder(hidden_size=hidden_size, input_shape=input_shape)

    def call(self, inputs):
        mu, sigma, z = self.encoder(inputs)
        reconstruction = self.decoder(z)

        reconstruction_loss = 28 * 28 * keras.losses.mse(K.flatten(inputs), K.flatten(reconstruction))
        kl_loss = -0.5 * K.sum(1 + sigma - K.square(mu) - K.exp(sigma), axis=-1)
        elbo_loss = K.mean(reconstruction_loss + kl_loss)
        self.add_loss(elbo_loss)

        return reconstruction
