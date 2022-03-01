import numpy as np
import tensorflow.keras.layers as layers
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.layers import Lambda
from tqdm import tqdm

from .abstract_models import SharedLayer, StripsModel, Mask, MyModel
from ..data import abstact_ds


class StripsNN(StripsModel):
    """ Neural Network based model to predict Wear at the Centre from list of strips of a campaign.

    The structure of the model is precised in the reports. To resume, it is based on the equation
    Wc = sum_strips(k * F/w * L)
    Where only k is computed with a Neural Network, while F/w*L is got from the inputs.
    """

    def __init__(self, dataset: abstact_ds.DataSet, hidden_layer_sizes: tuple,
                 hidden_layer_activ: tuple = None, mask: Mask = None, *args, **kwargs):
        """ Initialisation of the Fully Connected Neural Network

        :param dataset:
        :param hidden_layer_sizes: Example (16, 16, 8, 4, 4, 4, 2)
        :param hidden_layer_activ: None or ('r', 'sigmoid', 'r', 'softsign', 'sigmoid', 'sigmoid')
        :param mask: Mask applied to the inputs
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_activ = hidden_layer_activ
        self.mask: Mask = mask

        seq_length = dataset.get_x().shape[1]
        output_shape = dataset.get_y().shape
        output_length = 1 if np.ndim(dataset.get_y()) == 1 else output_shape[-1]

        # We load the inputs
        input_list, fwl_layers, masked_inputs = self._get_input_keras(dataset)

        # We reuse the same layer instance multiple times, the weights also being reused
        # (it is effectively *the same* layer)
        shared_layer = SharedLayer(hidden_layer_sizes, hidden_layer_activ, output_length)

        # The list of multiplication layers, for all inputs
        multiplication_layers = [layers.multiply([shared_layer(masked_inputs[i]), fwl_layers[i]])
                                 for i in tqdm(range(seq_length), 'FCNN - Creating Neural Net')]

        output = layers.add(multiplication_layers)

        # We call the super function to compile the model
        super(StripsNN, self).__init__(dataset, hidden_layer_sizes, hidden_layer_activ, mask,
                                       inputs=input_list, outputs=output, *args, **kwargs)


# todo: is this function interesting for them ?
class StripsDT(MyModel):
    """ Copies the behaviour and results of a given Strips NN """

    def __init__(self, neural_net: StripsNN, decision_tree: DecisionTreeRegressor):
        """ Model initialised from a Strips neural_net to copy """
        super(StripsDT, self).__init__()

        # todo: check this is still valid
        self.layer_to_copy: SharedLayer = neural_net.layers[306]
        self.dt = decision_tree

    @staticmethod
    def reshape_inputs(x):
        # We flatten, from (x, 306, 20) to (306 * x, 20)
        return x, x[:, :, -1]

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, **kwargs):

        if validation_data is not None:
            x_val, y_val = validation_data
        else:
            x, x_val, y, y_val = train_test_split(x, y, test_size=validation_split)

        # We reshape x, get the y to predict, and validate our results
        x_fit = self.reshape_inputs(x)
        y_fit = self.layer_to_copy.predict(x)
        self.dt.fit(x_fit, y_fit)

        # Printing results
        print(
            'MAE train: %.1f µm - MAE dev: %.1f µm' % (1000 * self.evaluate(x, y), 1000 * self.evaluate(x_val, y_val)))

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, *args):
        y_pred = self.predict(x)

        return mean_absolute_error(y, y_pred)

    def predict(self, x, batch_size=None, verbose=0, steps=None, *args):
        # We transpose, from (306, x, 20) to (x, 306, 20)
        fwl = x[:, :, -1]

        y = np.zeros((x.shape[0], 1))
        for id_cp, cpgn in enumerate(x):
            y[id_cp] = (self.dt.predict(cpgn) * fwl[id_cp]).sum()

        return y

    def grid_search(self, x):
        tuned_parameters = [{'max_depth': [2, 4, 8, 16],
                             'min_samples_split': [8, 16, 32, 64, 128, 256, 512],
                             'min_samples_leaf': [2, 4, 8, 16, 32, 64, 128, 256],
                             'max_features': [2, 4],
                             'max_leaf_nodes': [None],
                             'presort': [False]}]
        clf = GridSearchCV(DecisionTreeRegressor(), tuned_parameters, cv=4,
                           verbose=2, scoring='neg_mean_absolute_error')

        # We transpose, from (x, 306, 20) to (306 * x, 20)
        x = x.reshape((-1, x.shape[2]))
        y = self.layer_to_copy.predict(x)

        clf.fit(x, y)

        print("Best parameters set found on Train set:")
        print('\t%s' % clf.best_params_)

        return clf.best_params_


class RecurrentStripsNN(StripsModel):
    """ Recurrent Neural Network based model to predict Wear at the Centre from list of strips of a campaign.

    The structure of the model is precised in the reports. To resume, it is based on the equation
    Wc = sum_strips(k * F/w * L)
    Where only k is computed with a Neural Network, while F/w*L is got from the inputs.
    The network is recurrent because the NN for k takes as input the sum of the previous estimated wears,
        hence the current wear of the roll
    """

    def __init__(self, dataset: abstact_ds.DataSet, hidden_layer_sizes: tuple,
                 hidden_layer_activ: tuple = None, mask: Mask = None, *args, **kwargs):
        """ Initialisation of the Recurrent Neural Network

        :param dataset:
        :param hidden_layer_sizes: Example (16, 16, 8, 4, 4, 4, 2)
        :param hidden_layer_activ: None or ('r', 'sigmoid', 'r', 'softsign', 'sigmoid', 'sigmoid')
        :param mask: Mask applied to the inputs
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_activ = hidden_layer_activ
        self.mask: Mask = mask

        seq_length = dataset.get_x().shape[1]
        output_shape = dataset.get_y().shape
        output_length = 1 if np.ndim(dataset.get_y()) == 1 else output_shape[-1]

        # We load the inputs
        input_list, fwl_layers, masked_inputs = self._get_input_keras(dataset)

        # We reuse the same layer instance multiple times, the weights also being reused
        # (it is effectively *the same* layer)
        shared_layer = SharedLayer(hidden_layer_sizes, hidden_layer_activ, output_length)

        # The recurrent network gives to the SharedLayer as Input the current estimated wear of the roll.
        # This current wear corresponds to the sum of previously estimated wears. It begins at 0
        individual_wears = []
        # The first one is 0
        current_wear_layer = \
            Lambda(lambda a: 0 * a[:, 0:output_length], output_shape=(output_length,))(masked_inputs[0])

        first_input = layers.concatenate([current_wear_layer, masked_inputs[0]])
        individual_wears.append(layers.multiply([shared_layer(first_input), fwl_layers[0]]))

        # We create the rest of individual wears estimations
        for i in tqdm(range(1, seq_length), 'RNN - Creating Neural Net'):
            # The current wear is incremented of the previous wear
            previous_wear = individual_wears[-1]
            current_wear_layer = layers.add([current_wear_layer, previous_wear])

            new_input = layers.concatenate([current_wear_layer, masked_inputs[i]])
            individual_wears.append(layers.multiply([shared_layer(new_input), fwl_layers[i]]))

        output = layers.add(individual_wears)

        # We call the super function to compile the model
        super(RecurrentStripsNN, self).__init__(dataset, hidden_layer_sizes, hidden_layer_activ, mask,
                                                inputs=input_list, outputs=output, *args, **kwargs)


class RecurrentStripsNNDeltas(StripsModel):
    """ Recurrent Neural Network based model to predict Wear at the Centre from list of strips of a campaign.

    The structure of the model is precised in the reports. To resume, it is based on the equation
    Wc = sum_strips(k * F/w * L)
    Where only k is computed with a Neural Network, while F/w*L is got from the inputs.
    The network is recurrent because the NN for k takes as input the sum of the previous estimated wears,
        hence the current wear of the roll
    """

    def __init__(self, dataset: abstact_ds.DataSet, hidden_layer_sizes: tuple,
                 hidden_layer_activ: tuple = None, mask: Mask = None, *args, **kwargs):
        """ Initialisation of the Recurrent Neural Network

        :param dataset:
        :param hidden_layer_sizes: Example (16, 16, 8, 4, 4, 4, 2)
        :param hidden_layer_activ: None or ('r', 'sigmoid', 'r', 'softsign', 'sigmoid', 'sigmoid')
        :param mask: Mask applied to the inputs
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_activ = hidden_layer_activ
        self.mask: Mask = mask

        seq_length = dataset.get_x().shape[1]
        output_shape = dataset.get_y().shape
        output_length = 1 if np.ndim(dataset.get_y()) == 1 else output_shape[-1]

        # We load the inputs
        input_list, fwl_layers, masked_inputs = self._get_input_keras(dataset)

        # We reuse the same layer instance multiple times, the weights also being reused
        # (it is effectively *the same* layer)
        shared_layer = SharedLayer(hidden_layer_sizes, hidden_layer_activ, output_length)

        # The recurrent network gives to the SharedLayer as Input the current estimated wear of the roll.
        # This current wear corresponds to the sum of previously estimated wears. It begins at 0
        individual_wears = []
        # The first one is 0
        current_wear_layer = \
            Lambda(lambda a: 0 * a[:, 0:output_length], output_shape=(output_length,))(masked_inputs[0])

        first_input = layers.concatenate([current_wear_layer, masked_inputs[0]])
        individual_wears.append(layers.multiply([shared_layer(first_input), fwl_layers[0]]))

        # We create the rest of individual wears estimations
        for i in tqdm(range(1, seq_length), 'RNN - Creating Neural Net'):
            # The current wear is incremented of the previous wear
            previous_wear = individual_wears[-1]
            current_wear_layer = layers.add([current_wear_layer, previous_wear])

            new_input = layers.concatenate([current_wear_layer, masked_inputs[i]])
            individual_wears.append(shared_layer(new_input))

        output = layers.add(individual_wears)

        # We call the super function to compile the model
        super(RecurrentStripsNNDeltas, self).__init__(dataset, hidden_layer_sizes, hidden_layer_activ, mask,
                                                      inputs=input_list, outputs=output, *args, **kwargs)
