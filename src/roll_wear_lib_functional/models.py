import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Dense
from tqdm.notebook import trange


def my_range(stop: int, verbose: int, **kwargs):
    """ Returns a 'range' object adapted to the needs :
    either a classic range() or a tqdm.notebook.trange() if verbose wanted

    :param stop: Final value of the range
    :param verbose: 0 - Nothing;
            1 - Progress bar hidden at the end of progression;
            2 - Progress bar which stays at the end of progression;
    :param kwargs: additional arguments for the tqdm.notebook.trange() function

    :return: range or trange
    """
    if verbose == 0:
        return range(stop)
    elif verbose == 1:
        return trange(stop, leave=False, **kwargs)
    else:
        return trange(stop, leave=True, **kwargs)


# Neural Network

class DiagonalWeight(Constraint):
    """ Constrains the weights to be diagonal.
    source: https://stackoverflow.com/a/53756678/9531617 by @pitfall https://stackoverflow.com/users/1626564/pitfall """

    def __call__(self, w):
        n = tf.keras.backend.int_shape(w)[-1]
        m = tf.eye(n)
        w.assign(w * m)
        return w


def create_neural_net(output_length: int, hidden_layers_sizes: tuple, layers_activ: tuple):
    """ Initialisation of the Neural Net used to compute k from a strip

    :param output_length:
    :param hidden_layers_sizes: Example [16, 16, 8, 4, 4, 4, 2]
    :param layers_activ: Example ['r', 'sigmoid', 'r', 'softsign', 'sigmoid', 'sigmoid'] - if it has the same
        length than h_l_sizes, the activation of the output layer will be 'tanh'. To change the activation of the output
        layer, an additional activation can be given, so the length of l_activ is the length of h_l_sizes  + 1

    :returns: neural_net: keras Neural Network
    """

    # We add the output layer in the list of layers
    hidden_layers_sizes = hidden_layers_sizes + (output_length,)
    if len(layers_activ) == len(hidden_layers_sizes) - 1:
        layers_activ += ('sigmoid',)

    # We assert the activations are the same size than the dense layers
    assert (len(layers_activ) == len(hidden_layers_sizes)), \
        'The activation list must be the same size (or +1) than the layer list'

    neural_net = Sequential()
    # We add the dense layers, followed by activations
    for units, activation in zip(hidden_layers_sizes, layers_activ):
        neural_net.add(Dense(units=units, activation=activation))

    # Normalisation layer - Adding a second Dense 1 for learned normalization
    # A relu here create a dying ReLu. However, a elu makes the convergence very slow
    neural_net.add(Dense(units=output_length, activation='linear', kernel_constraint=DiagonalWeight()))

    return neural_net


def get_model(inputs_sample: pd.DataFrame, outputs_sample: pd.DataFrame,
              hidden_layers_sizes: tuple = (20, 8), layers_activ: tuple = ('selu', 'softsign', 'sigmoid')):
    """

    :param inputs_sample: Sample of an input, used to get Model dimensions.
                The easiest is to give the training inputs
    :param outputs_sample:Sample of an output, used to get Model dimensions.
                The easiest is to give the training outputs
    :param hidden_layers_sizes: Example [16, 16, 8, 4, 4, 4, 2]
    :param layers_activ: Example ['r', 'sigmoid', 'r', 'softsign', 'sigmoid', 'sigmoid'] - if it has the same
        length than h_l_sizes, the activation of the output layer will be 'tanh'. To change the activation of the output
        layer, an additional activation can be given, so the length of l_activ is the length of h_l_sizes  + 1

    :return: tuple (model_full, model_nn)
            - model_full: Full model, with fwl multiplication
            - model_nn: Core Neural Net, computing individual k coefficient
    """
    input_length, output_length = len(inputs_sample.columns), len(outputs_sample.columns)

    model_input = Input(shape=(input_length,), name='input')
    fwl_input = Input(shape=(output_length,), name='fwl_input')

    neural_net: Sequential = create_neural_net(output_length, hidden_layers_sizes, layers_activ)
    nn_output = neural_net(model_input)
    model_output = tf.keras.layers.multiply([nn_output, fwl_input])

    model_nn = Model(model_input, nn_output)
    model_full = Model([model_input, fwl_input], model_output)

    return model_full, model_nn
