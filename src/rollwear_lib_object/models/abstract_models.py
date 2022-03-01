import json
import os
import pickle
from abc import ABC

# noinspection PyUnresolvedReferences
import ipykernel  # Used to fix Pycharm keras ProgressBar bug https://stackoverflow.com/a/57475559/9531617
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import boolean_mask
from tensorflow.keras import Model, Sequential
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Input, Dense, Activation, Lambda
from tqdm import tqdm

from .metrics import mae_denorm
from ..data import abstact_ds
from ..data import datasets
from ..data.abstact_ds import StripsInputDB

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


# noinspection PyUnresolvedReferences
class MyModel(Model, ABC):
    save_folder = 'Data/Outputs/Models/'

    def __init__(self, *args, **kwargs):
        super(MyModel, self).__init__(*args, **kwargs)
        self.default_compile()

    def default_compile(self):
        # Default compilation, calling self.compile with adapted arguments
        self.compile(optimizer='adam', loss='mae', metrics=[])

    def reset_training(self):
        # We reset the weights, then the optimizer by recompiling
        self._reset_weights()
        self.default_compile()

    def _reset_weights(self):
        # We reinitialize all the weights
        for weight_kernel in self.weights:
            weight_kernel.initializer.run(session=k.get_session())

    def summary(self, line_length=None, positions=None, print_fn=None):
        """ Modified definition of the printed summary, as the classical definition is not adapted
        to multiple parallel-layers

        :param line_length: Not used anymore
        :param positions: Not used anymore
        :param print_fn: Not used anymore
        :return:
        """
        in_shape = self.input_shape
        in_shape = (in_shape[0][0], len(in_shape), in_shape[0][1])
        try:
            trainable_count = int(
                np.sum([k.count_params(p) for p in set(self.trainable_weights)]))
        except TypeError:
            trainable_count = None
        non_trainable_count = int(
            np.sum([k.count_params(p) for p in set(self.non_trainable_weights)]))

        print('_______________________________________________________________\n'
              'Model Summary\n'
              '===============================================================\n'
              '\tInput Shape\t\t\tOutput Shape\n'
              '_______________________________________________________________\n'
              '\t%r\t\t%r\n'
              '===============================================================\n'
              '\tTotal Weights\t\tTrainable W.\t\tNon-Trainable W.\n'
              '_______________________________________________________________\n'
              '\t%r\t\t\t\t\t%r\t\t\t\t\t%r\n'
              '_______________________________________________________________\n'
              % (in_shape, self.output_shape, self.count_params(), trainable_count, non_trainable_count))

    @staticmethod
    def _get_input_keras(dataset: abstact_ds.DataSet):
        x = dataset.get_x()

        # We get the keras Inputs of the neural networks: one for parameters, one for FwL
        seq_length, m = x[0].shape

        # The first inputs are all the parameters
        input_list = [Input(shape=(m,)) for _ in tqdm(range(seq_length), 'NN - Creating inputs (1/3)')]
        # The second inputs are the list of FwL so they can be used as variables later in the NN
        if dataset.input.all_rolls:
            # [fwl_f6, fwl_f6, fwl_f7, fwl_f7] in case of 4 rolls
            # fwl_layer = Lambda(lambda a: a[:, [-2, -2, -1, -1]])
            fwl_layer = Lambda(lambda a: tf.transpose(tf.convert_to_tensor([a[:, -2], a[:, -2], a[:, -1], a[:, -1]])))
        else:
            fwl_layer = Lambda(lambda a: a[:, -1])
        fwl_layer_list = [fwl_layer(input_x) for input_x in tqdm(input_list, 'NN - Creating inputs (2/3)')]

        return input_list, fwl_layer_list

    @staticmethod
    def reshape_inputs(x):
        """ Reshape inputs to correspond to the NN structure """
        return x

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, **kwargs):
        # We call the mother method but reshaping a before
        reshaped_x = self.reshape_inputs(x)
        if validation_data is not None:
            x_val, y_val = validation_data
            validation_data = (self.reshape_inputs(x_val), y_val.to_numpy())

        return super(MyModel, self).fit(reshaped_x, y.to_numpy(), batch_size, epochs, verbose, callbacks,
                                        validation_split, validation_data, shuffle, class_weight, sample_weight,
                                        initial_epoch, steps_per_epoch, validation_steps, **kwargs)

    def predict(self, x, batch_size=None, verbose=0, steps=None, *args):
        # We call the mother method but reshaping a before
        reshaped_x = self.reshape_inputs(x)
        return super(MyModel, self).predict(reshaped_x, batch_size, verbose, steps)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, *args):
        # We call the mother method but reshaping a before
        reshaped_x = self.reshape_inputs(x)
        return super(MyModel, self).evaluate(reshaped_x, y, batch_size, verbose, sample_weight, steps)

    @classmethod
    def _get_save_folder(cls, filename):
        save_folder = cls.save_folder + filename + '/'
        try:
            os.makedirs(save_folder)
        except FileExistsError:
            pass

        return save_folder

    def save(self, filename, overwrite=True, include_optimizer=True, **kwargs):
        save_folder = self._get_save_folder(filename)

        # Saving architecture
        with open(save_folder + 'save.json', "w") as outfile:
            json.dump(self.to_json(), outfile)

        # Saving weights
        name_weights = save_folder + 'weights.h5'
        self.save_weights(name_weights, overwrite=overwrite)

    def plot_history(self):
        """ Plot the evolution of the accuracy and loss of the model fit """
        try:
            history = self.history
            plt.figure()
            for i in range(2):
                plt.subplot(1, 2, i + 1)
                for key in history.history.keys():
                    plt.plot(history.history[key], label=key)

                plt.title('Model metrics during fitting')
                plt.ylabel('Losses')
                plt.xlabel('Epochs')
                plt.legend()

            # On the second graph, we apply a zoom on the last details of convergence
            plt.ylim([0, 0.10])
        except AttributeError:
            print("Can't plot history")


class RollWearModel(MyModel, ABC):

    def __init__(self, dataset: datasets.DataSet, *args, **kwargs):
        # We finally add an additional metric
        def mae_raw(y_true, y_pred):
            return mae_denorm(dataset.output, y_true, y_pred, True)

        self.mae_raw = mae_raw

        super(RollWearModel, self).__init__(*args, **kwargs)

    def default_compile(self):
        self.compile(optimizer='adam',
                     loss=self.mae_raw,
                     metrics=['mae'])


class Mask(object):

    def __init__(self, f6: bool, top: bool,
                 fw_and_l: bool,
                 fwl: bool,
                 strips_param: bool,
                 roll_param: bool,
                 hardness_indic: bool,
                 family: bool,
                 suppliers: bool,
                 cum_length: bool,
                 all_rolls: bool = False, **kwargs):
        """ Create a Mask for the Neural Network. This one selects the inputs to give to the Neural Network.

        :param f6: Is the roll the f6 or f7 
        :param top: Is the Roll top or bottom
        :param fw_and_l: True to keep 'F/w' and 'L' in inputs
        :param fwl: True to keep 'F/w * L' in inputs
        :param strips_param: True to keep the strips parameters in inputs
        :param roll_param: True to keep the roll parameters in inputs
        :param hardness_indic: True to keep the strips hardness indicator in inputs
        :param family: True to keep the strips families in inputs
        :param suppliers: True to keep the roll suppliers in inputs
        :param cum_length: True to keep the cumulative length in inputs
        """

        # We create the dataset that will be used to select columns
        dataset = datasets.StripsWearCenter(f6, top, all_rolls=all_rolls)

        self.fw_and_l: bool = fw_and_l
        self.fwl: bool = fwl
        self.strips_param: bool = strips_param
        self.roll_param: bool = roll_param
        self.hardness_indic: bool = hardness_indic
        self.family: bool = family
        self.suppliers: bool = suppliers
        self.cum_length: bool = cum_length

        col_to_keep = []
        # F/w & L
        if self.fw_and_l:
            col_to_keep += ['STAND FORCE / WIDTH F6*', 'STRIP LENGTH F5 EXIT*',
                            'STAND FORCE / WIDTH F7*', 'STRIP LENGTH F6 EXIT*']
        # F/w * L
        if self.fwl:
            col_to_keep += ['F/w L', 'F/w L F6', 'F/w L F7']
        # Other strips parameters (temperature...)
        if self.strips_param:
            col_to_keep += ['BENDING FORCE F6', 'TEMPERATURE F6 EXIT', 'LEAD  SPEED F6', 'TRAIL SPEED F6',
                            'REDUCTION F6*', 'CONTACT LENGTH F6 TOP*', 'CONTACT LENGTH F6  BOT*',
                            'F6 Oil Flow Rate, on/off']
            col_to_keep += ['BENDING FORCE F7', 'TEMPERATURE F7 EXIT', 'LEAD SPEED F7',
                            'TRAIL SPEED F7', 'REDUCTION F7*', 'CONTACT LENGTH F7 TOP*', 'CONTACT LENGTH F7 BOT*']
        # Roll parameters
        if self.roll_param:
            col_to_keep += ['F6 TOP DIAMETER', 'F6 TOP HARDNESS', 'F6 BOT DIAMETER', 'F6 BOT HARDNESS']
            col_to_keep += ['F7 TOP DIAMETER', 'F7 TOP HARDNESS', 'F7 BOT DIAMETER', 'F7 BOT HARDNESS']
        # Hardness Indicator (linked to family)
        if self.hardness_indic:
            col_to_keep += ['STRIP HARDNESS INDICATOR']
        # Strip families
        if self.family:
            col_to_keep += StripsInputDB.columns_family
        # Roll suppliers
        if self.suppliers:
            # We remove the first value, which is not a supplier
            col_to_keep += StripsInputDB.columns_f6b[1:] + StripsInputDB.columns_f6t[1:] + \
                           StripsInputDB.columns_f7b[1:] + StripsInputDB.columns_f7t[1:]
        # Campaign Line Up
        if self.cum_length:
            col_to_keep += ['CUMULATIVE ROLLING LENGTH F6*', 'CUMULATIVE ROLLING LENGTH F7*']

        # We create the mask to correspond to the data
        inp: StripsInputDB = dataset.input
        self.mask: np.array = np.in1d(inp.columns_name, col_to_keep)

        super(Mask, self).__init__(**kwargs)

    def apply(self, x: np.array):
        assert x.ndim == 3, 'X must have 3 dimensions'
        return x[:, :, self.mask]

    def __str__(self):
        bold = '\033[1m'
        normal = '\033[0m'
        return \
            bold + "Mask data:" + normal + \
            "\n\tF/w & L:\t\t\t" + bold + str(self.fw_and_l) + normal + \
            "\n\tF/w * L:\t\t\t" + bold + str(self.fwl) + normal + \
            "\n\tStrip parameters:\t" + bold + str(self.strips_param) + normal + \
            "\n\tRoll parameters:\t" + bold + str(self.roll_param) + normal + \
            "\n\tHardness Indicator:\t" + bold + str(self.hardness_indic) + normal + \
            "\n\tStrip Family:\t\t" + bold + str(self.family) + normal + \
            "\n\tRoll Supplier:\t\t" + bold + str(self.suppliers) + normal + \
            "\n\tCumulative Length:\t" + bold + str(self.cum_length) + normal


class StripsModel(RollWearModel, ABC):

    def __init__(self, dataset: abstact_ds.DataSet, hidden_layer_sizes: tuple,
                 hidden_layer_activ: tuple = None, mask: Mask = None, *args, **kwargs):
        self.ignore_fwl_position = False

        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_activ = hidden_layer_activ
        self.mask: Mask = mask
        super(StripsModel, self).__init__(dataset=dataset, *args, **kwargs)

    def assert_fwl(self, x: np.array):
        # As a is normalized column-wise, we have a[-1] == ratio * a[0] * a[1]
        #   To make sure we have a[0]*a[1] != 0, we take the maximum of all vectors, as it should be at the same pos.
        ratio_1roll = x[:, :, -1].max() / (x[:, :, 0] * x[:, :, 1]).max()

        ratio_4rolls_f6 = x[:, :, -2].max() / (x[:, :, 0] * x[:, :, 1]).max()
        ratio_4rolls_f7 = x[:, :, -1].max() / (x[:, :, 2] * x[:, :, 3]).max()

        # Due to the 0s, we cannot divide. Due to computations errors (~= 1e-16) we cannot just check a - r*b = 0.
        #   So we check if a[-1] - (r*a[0]*a[1]) < comp_error.
        epsilon = 1e-10

        # We assert either:
        #   No verification
        #   In the 1-roll case, that a[-1] - (r*a[0]*a[1]) < comp_error
        #   In the 4-rolls case, that
        #       a[-2] - (r*a[0]*a[1]) < comp_error, for F6
        #       a[-1] - (r*a[2]*a[3]) < comp_error, for F7
        assert \
            self.ignore_fwl_position \
            or np.max(x[:, :, -1] - ratio_1roll * x[:, :, 0] * x[:, :, 1]) < epsilon \
            or (np.max(x[:, :, -2] - ratio_4rolls_f6 * x[:, :, 0] * x[:, :, 1]) < epsilon
                and np.max(x[:, :, -1] - ratio_4rolls_f7 * x[:, :, 2] * x[:, :, 3]) < epsilon), \
            'The last column of X may not be F/w * L, which implies this network may not work.' \
            'To ignore this message, set StripsModel.ignore_fwl_position to True'

    def _get_input_keras(self, dataset: abstact_ds.DataSet):
        input_list, fwl_layer_list = super(StripsModel, self)._get_input_keras(dataset)

        # We integrate the mask layer to the inputs
        if self.mask is not None:
            mask_array = np.copy(self.mask.mask)  # Needed so the Lambda Layer won't try to save 'self', creating a loop

            # todo: output_shape was originally (np.sum(self.mask.mask), ) but it had to be changed for debugging.
            #  Maybe could bug if conditions changed
            mask_layer = Lambda(lambda a: boolean_mask(a, mask_array, axis=1),
                                output_shape=(np.sum(self.mask.mask), np.sum(self.mask.mask)))
            masked_input_list = [mask_layer(inp) for inp in tqdm(input_list, 'NN - Creating inputs (3/3)')]
            for maked_inp in masked_input_list:
                maked_inp.set_shape((np.sum(self.mask.mask), np.sum(self.mask.mask)))

        else:
            masked_input_list = input_list

        return input_list, fwl_layer_list, masked_input_list

    def reshape_inputs(self, x: np.array):
        """ We check that FwL is the last one, based on relation, then create inputs """
        self.assert_fwl(x)
        x_new = [x[:, i, :] for i in range(x.shape[1])]

        return x_new

    def summary(self, line_length=None, positions=None, print_fn=None):
        print(self.mask)
        super(StripsModel, self).summary(line_length, positions, print_fn)

    def save(self, filename, overwrite=True, include_optimizer=True, **kwargs):

        # We use the mother save method and then we save the structure of the shared layer
        super(StripsModel, self).save(filename, overwrite, include_optimizer, )

        # We save the structure
        dir_path = self._get_save_folder(filename)
        with open(dir_path + 'struct.pkl', 'wb') as f:
            pickle.dump([self.hidden_layer_sizes, self.hidden_layer_activ, self.mask], f)

    @classmethod
    def load(cls, filename, dataset):

        # We load the structure to recreate the network, and then load the weights
        dir_path = cls._get_save_folder(filename)
        with open(dir_path + 'struct.pkl', 'rb') as f:
            load = pickle.load(f)
            try:
                hidden_layer_sizes, hidden_layer_activ, mask = load
            except ValueError:
                hidden_layer_sizes, hidden_layer_activ = load
                mask = None

        fcnn = cls(dataset, hidden_layer_sizes, hidden_layer_activ, mask)

        # Saving weights
        name_weights = dir_path + 'weights.h5'
        fcnn.load_weights(name_weights)

        return fcnn


class SharedLayer(Sequential):
    """ A Sequential Layer, used as a base for both models """

    def __init__(self, hidden_layer_sizes: tuple, hidden_layer_activ: tuple = None, output_size: int = 1):
        """ Initialisation of the SharedLayer

        :param hidden_layer_sizes: Example (16, 16, 8, 4, 4, 4, 2)
        :param hidden_layer_activ: None or ('r', 'sigmoid', 'r', 'softsign', 'sigmoid', 'sigmoid')
        """
        super(SharedLayer, self).__init__()
        if hidden_layer_activ is None:
            hidden_layer_activ = ['r'] * len(hidden_layer_sizes)

        # We assert the activations are the same size than the dense layers
        assert (len(hidden_layer_activ) == len(hidden_layer_sizes)) or (
                len(hidden_layer_activ) == len(hidden_layer_sizes) + 1), \
            'The activation list must be the same size (or +1) than the layer list'

        # We add the dense layers, followed by activations
        for i, n in enumerate(hidden_layer_sizes):
            self.add(Dense(n))
            activ = 'relu' if hidden_layer_activ[i] == 'r' else hidden_layer_activ[i]
            self.add(Activation(activ))

        # If their is one more activation given, it is used for the final layer
        if len(hidden_layer_activ) != len(hidden_layer_sizes) + 1:
            # Default
            activ = 'tanh'
        else:
            activ = 'relu' if hidden_layer_activ[-1] == 'r' else hidden_layer_activ[-1]

        # Final layer
        self.add(Dense(output_size))
        self.add(Activation(activ))
        # Normalisation layer - Adding a second Dense 1 for learned normalization
        self.add(Dense(output_size, kernel_constraint=DiagonalWeight()))  # todo: check the constraint is working
        # A relu here create a dying ReLu. However, a elu makes the convergence very slow
        self.add(Activation('linear'))


class DiagonalWeight(Constraint):
    """ Constrains the weights to be diagonal.
    source: https://stackoverflow.com/a/53756678/9531617 by @pitfall https://stackoverflow.com/users/1626564/pitfall """

    def __call__(self, w):
        n = tf.keras.backend.int_shape(w)[-1]
        m = tf.eye(n)
        w.assign(w * m)
        return w
