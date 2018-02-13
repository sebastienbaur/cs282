"""
Define some useful functions for sampling in the latent space
"""

import keras.backend as K


def sample_eps(batch_size, latent_dim, epsilon_std):
    """Create a function to sample N(0, epsilon_std) vectors"""
    return lambda args: K.random_normal(shape=(batch_size, latent_dim),
                                        mean=0.,
                                        stddev=epsilon_std)


def sample_z0(args):
    """Sample from N(mu, sigma) where sigma is the stddev !!!"""
    z_mean, z_std, epsilon = args
    # generate z0 according to N(z_mean, z_std)
    z0 = z_mean + z_std * epsilon
    return z0


def sampling(batch_size, latent_dim, epsilon_std):
    """
    Useful when you want to save a model
    It allows not to use global variables in the sampling function, so that the Lambda layer can be serialized and deserialized as well
    :param batch_size:
    :param latent_dim:
    :param epsilon_std:
    :return:
    """
    return lambda args: _sampling(args, batch_size, latent_dim, epsilon_std)


def _sampling(args, batch_size, latent_dim, epsilon_std):
    """An auxiliary function used by `sampling`"""
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon
