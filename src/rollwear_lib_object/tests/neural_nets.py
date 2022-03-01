"""
This file contains function to test the training and loading of Neural Networks.
Those functions are not meant to be used for actual training but for code testing and quick demonstrations

@author: Antonin GAY (U051199)
"""

import matplotlib.pyplot as plt

from .. import wearcentre_from_means
from .. import wearcentre_from_strips
from ..data import datasets
from ..data.inputs import InputModes
from ..models.strips import StripsNN, Mask
from ..plot import training as plot_training


def test_neuralnet_fullprofile(epochs: int = 250):
    """ Tests if the training of the Neural Network is working.
    For Neural Network predicting **Full Profiles** from all the **Strips data**

    :param epochs: Number of training steps for the Neural Network
    """
    # We load the dataset and network
    dataset = datasets.StripsProfile(f6=True, top=True)
    neural_net = StripsNN(dataset, (16, 8, 4, 4), ('r', 'r', 'r', 'sigmoid'))
    neural_net.summary()

    # We train it
    x_train, x_dev, _, y_train, y_dev, _ = dataset.get_train_var()
    neural_net.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=1, validation_data=(x_dev, y_dev))

    # We plot the training
    plot_training.profiles_prediction(neural_net, dataset)
    plt.show()


def test_neuralnet_mean_wc(epochs=250):
    """ Tests if the training of the Neural Network is working.
    For Neural Network predicting **Wear at the Centre** from **Means over the campaigns**

    :param epochs: Number of training steps for the Neural Network
    """
    wearcentre_from_means.training(
        epochs=epochs, layers_sizes=(20, 8, 4, 4), layers_activations=('selu', 'selu', 'selu', 'selu'),
        f6=True, top=True, mode=InputModes.SELECTED, verbose=1)
    plt.show()


def test_neuralnet_strips_wc(epochs=250):
    """ Tests if the training of the Neural Network is working.
    For Neural Network predicting **Wear at the Centre** from all the **Strips** data

    :param epochs: Number of training steps for the Neural Network
    """
    wearcentre_from_strips.train_neuralnet(
        epochs=epochs, layers_sizes=(20, 8), layers_activations=('selu', 'selu', 'sigmoid'),
        mask=Mask(True, True, False, False, False, False, False, True, True, False),
        savefile_name='test', verbose=1, recurrent=False, f6=True, top=True)


def test_recurrentnn_strips_wc(epochs=250):
    """ Tests if the training of the Neural Network is working.
    For **Recurrent** Neural Network predicting **Wear at the Centre** from all the **Strips** data

    :param epochs: Number of training steps for the Neural Network
    """
    wearcentre_from_strips.train_neuralnet(
        epochs=epochs, layers_sizes=(20, 8), layers_activations=('selu', 'selu', 'sigmoid'),
        mask=Mask(True, True, False, False, False, False, False, True, True, False),
        savefile_name='test_recurrent', verbose=1, recurrent=True, f6=True, top=True)


def test_recurrentnn_strips_weardeltas(epochs=250):
    """ Tests if the training of the Neural Network is working.
    For **Recurrent** Neural Network predicting **Deltas for wear at the centre** from all the **Strips** data

    :param epochs: Number of training steps for the Neural Network
    """
    wearcentre_from_strips.train_neuralnet_deltas(
        epochs=epochs, layers_sizes=(20, 8), layers_activations=('selu', 'selu', 'linear'),
        mask=Mask(True, True, False, False, False, False, False, True, True, False),
        savefile_name='test_recurrent_deltas', verbose=1, f6=True, top=True, all_rolls=False)


def test_loading_neuralnet():
    """ Testing the loading of an already-trained network """
    wearcentre_from_strips.load_neuralnet('Fam_Sup_20_8_SeSeSi', True, True)


def test_loading_recurrentnn():
    """ Testing the loading of an already-trained recurrent network """
    wearcentre_from_strips.load_neuralnet('Recurrent', True, True)


def test_loading_recurrentnn_deltas():
    """ Testing the loading of an already-trained recurrent network """
    wearcentre_from_strips.load_neuralnet_deltas('test_recurrent_deltas', f6=True, top=True)
