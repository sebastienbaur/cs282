"""Some log densities"""

import math
import numpy as np
import keras.backend as K


def log_stdnormal(x):
    """log density d'une loi normale centree"""
    c = - 0.5 * math.log(2*math.pi)
    result = c - K.square(x) / 2
    return result


def log_normal(e, log_var):
    """
    log density of a diagonal gaussian
    :param e: (x-mu)/var
    :param log_var: log(var)
    e and log_var are two tensors of shape (batch_size, dim).
    :return: a tensor of shape (batch_size, dim). The sum on the last axis is the true log-density
    """
    c = - math.log(2 * math.pi)
    result = 0.5*(c - log_var - K.square(e))
    return result


def log_normal2(x, mean, log_var):
    """log density d'une loi normale de moyenne mean et de variance exp(log_var)"""
    c = - 0.5 * math.log(2*math.pi)
    result = c - log_var/2 - K.square(x - mean) / (2 * K.exp(log_var) + 1e-8)
    return result


def log_normal2_np(x, mean, log_var):
    """log density d'une loi normale de moyenne mean et de variance exp(log_var)"""
    c = - 0.5 * math.log(2*math.pi)
    result = c - log_var/2 - (x - mean)**2 / (2 * np.exp(log_var) + 1e-8)
#     print(result)
    return result
