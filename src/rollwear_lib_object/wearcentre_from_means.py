from sklearn.linear_model import LinearRegression

from .data import datasets
from .data.inputs import InputModes
from .models.mean_campaigns import MeanCampNN
from .plot import training as plot_training


def linear_regressor(f6: bool, top: bool, mode: str = InputModes.SELECTED):
    """ Trains a Linear Regressor to predict **Wear at the Centre** from **Means over campaigns**

    :param f6: True if the Roll is from F6 stand, False if f7
    :param top: True if top Roll, False if bottom
    :param mode: Which inputs should be kept. List of modes: InputModes
    """
    # We load the network
    dataset = datasets.MeanWearCenter(f6, top, mode)
    regressor = LinearRegression()

    # We train it
    x_train, x_dev, _, y_train, y_dev, _ = dataset.get_train_var()
    regressor.fit(x_train, y_train)

    # We plot the training
    # noinspection PyTypeChecker
    plot_training.wearcentre_predictions(regressor, dataset)


def training(epochs: int, layers_sizes: tuple, layers_activations: tuple,
             f6: bool, top: bool, mode: str = InputModes.SELECTED, verbose: int = 1):
    """ Create and train a Neural Network for estimating the Wear at the Centre from the mean parameters over a campaign

    :param epochs: number of steps for training. A classical value would be around 250
    :param layers_sizes: The sizes of the layers of the NN.
    :param layers_activations: The activations of the neurons of each layer
    :param f6: True if the Roll is from F6 stand, False if f7
    :param top: True if top Roll, False if bottom
    :param mode: Which inputs should be kept. List of modes: InputModes
    :param verbose: Parameter to define the quantity of written text
    """
    # We load the network
    dataset = datasets.MeanWearCenter(f6, top, mode)
    neural_net = MeanCampNN(dataset, layers_sizes, layers_activations, True)
    neural_net.summary()

    # We train it
    x_train, x_dev, _, y_train, y_dev, _ = dataset.get_train_var()
    neural_net.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=verbose, validation_data=(x_dev, y_dev))

    # We plot the training
    plot_training.wearcentre_predictions(neural_net, dataset)
