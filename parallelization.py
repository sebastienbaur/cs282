"""
Allows to paralellize a model, so that it can be executed on several GPU
It allows to use a bigger batch size, for instance 128 with 4 GPU instead of 32 with one
"""
from keras.layers import Concatenate
from keras.layers.core import Lambda
from keras.models import Model
import keras.backend as K
import tensorflow as tf


def make_parallel(model, n_gpus):
    def get_slice(x, n_gpus, part):
        """
        Divide the input batch into `n_gpus` slices, and obtain slice no. `part`.
        i.e. if len(x)=10, then slice_batch(x, 2, 1) will return x[5:].
        """
        sh = K.shape(x)
        L = sh[0] / n_gpus
        if part == n_gpus - 1:
            return x[part * L:]
        return x[part * L:(part + 1) * L]

    # Let L be the number of outputs. Each is computed by slicing the number of elements in the input batches.
    # e.g., if there are 1 input and 2 outputs, each containing 128 elements, and 2 gpus, the model will be applied on the first half of the input and then on
    # the second part, thus generating first the first half of the two outputs and then the second half of the two outputs
    # These halves are then merged together
    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    print('output all:', outputs_all)

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(n_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'part': i, 'n_gpus': n_gpus})(x)  # take the i-th part of each input
                    inputs.append(slice_n)

                outputs = model(inputs)  # compute the i-th part of each output

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    print(outputs[l])
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged_outputs = []
        for outputs in outputs_all:
            merged_outputs.append(Concatenate(0)(outputs))

        return Model(input=model.inputs, output=merged_outputs)
