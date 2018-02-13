# -*- coding: utf-8 -*-
"""
Script for training an ICNN with the sepsis dataset

@author: Camilo
"""

import pandas as pd
import models_keras


#def ICNN_block(n_units_u, n_units_z, l, activation='selu'):
#    """
#
#    Let's denote `s` the states and `a` the actions
#
#    The ICNN is convex in `a` but not convex in `s`. We define:
#    u_0 = s,
#    u_i+1 = g(u_i),
#    z_i+1 = f(z_i, u_i, a),
#    output = z_L
#
#    This function is just a block of the neural network
#    """
#
#    def f(u, z, a):
#        u_ = Dense(n_units_u, activation=activation)(u)
#
#        z_1 = Dense(n_units_z, activation=None)(u)
#
#        a_shape = a._keras_shape[1]
#        z_2 = Multiply()([a, Dense(a_shape, activation=None)(u)])
#        z_2 = Dense(n_units_z, activation=None, use_bias=False)(z_2)
#
#        z_shape = z._keras_shape[1]
#        z_3 = Multiply()([Dense(z_shape, activation='relu')(u), z])
#        z_3 = Dense(n_units_z, activation=None, use_bias=False, name='w'+str(l))(z_3)  # these weights should stay positive
#
#        z_ = _activation(activation, BN=False)(Add()([z_1, z_2, z_3]))
#
#        return u_, z_
#
#    return f



# General idea of this file: 1. get some dummy inputs from the train dataset. 2. Build a very simple, shallow ICNN. 
# 3. Train it with the inputs. 4. Observe and plot the results. 5. Test with part of test set. 6. Observe output Q values. 7. Check concavity.

# 1. Get inputs
sepsis_df = pd.read_csv('../sepsis_imp.csv')



# 2. Build basic ICNN
n_units_u = 1
n_units_z = 1
l = 1
icnn = models_keras.ICNN_block(n_units_u, n_units_z, l)

icnn(u,z,a)

# 3. Train with inputs

# 4. Observe and plot output of net

# 5. Test with part of test set

# 6. Observe and plot output Q values

# 7. Check concavity wrt actions

