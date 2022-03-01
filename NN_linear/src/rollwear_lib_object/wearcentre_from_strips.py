from tensorflow.keras.callbacks import EarlyStopping

from .data import datasets
from .models.strips import StripsNN, RecurrentStripsNN, Mask, RecurrentStripsNNDeltas
from .plot import training as plot_training


def train_neuralnet(epochs: int, layers_sizes: tuple, layers_activations: tuple, savefile_name: str,
                    mask: Mask, recurrent: bool, f6: bool, top: bool, verbose: int = 1, all_rolls: bool = False):
    """ Create, train and save a Neural Network for estimating the Wear at the Centre from all the strips of a campaign

    :param all_rolls: Taking into account all the rolls
    :param epochs: number of steps for training. A classical value would be around 250
    :param layers_sizes: The sizes of the layers of the NN.
    :param layers_activations: The activations of the neurons of each layer
    :param savefile_name: The name of the filder in which the model should be saved
    :param mask: The data mask
    :param recurrent: True if the model should be recurrent, using current estimated wear as an input
    :param f6: True if the Roll is from F6 stand, False if f7
    :param top: True if top Roll, False if bottom
    :param verbose: Parameter to define the quantity of written text
    """
    # Loading the DataSet
    dataset = datasets.StripsWearCenter(f6, top, random_seed=42, all_rolls=all_rolls)
    x_train, x_dev, x_test, y_train, y_dev, y_test = dataset.get_train_var()

    # Loading the network
    if recurrent:
        neural_net = RecurrentStripsNN(dataset,
                                       hidden_layer_sizes=layers_sizes,
                                       hidden_layer_activ=layers_activations,
                                       mask=mask)
    else:
        neural_net = StripsNN(dataset,
                              hidden_layer_sizes=layers_sizes,
                              hidden_layer_activ=layers_activations,
                              mask=mask)

    # Printing and training the network
    neural_net.summary()
    neural_net.fit(x_train, y_train, epochs=epochs, validation_data=(x_dev, y_dev), verbose=verbose, batch_size=16,
                   callbacks=[
                       EarlyStopping('val_loss', min_delta=0.1, patience=15, mode='min', restore_best_weights=True)])

    # We plot the results
    plot_training.wearcentre_predictions(neural_net, dataset)

    # We save and returns the results
    neural_net.save(savefile_name, )
    return neural_net


def train_neuralnet_deltas(epochs: int, layers_sizes: tuple, layers_activations: tuple, savefile_name: str,
                           mask: Mask, f6: bool, top: bool, verbose: int = 1, all_rolls: bool = False):
    """ Create, train and save a Neural Network for estimating the Wear at the Centre from all the strips of a campaign

    :param all_rolls: Taking into account all the rolls
    :param epochs: number of steps for training. A classical value would be around 250
    :param layers_sizes: The sizes of the layers of the NN.
    :param layers_activations: The activations of the neurons of each layer
    :param savefile_name: The name of the filder in which the model should be saved
    :param mask: The data mask
    :param f6: True if the Roll is from F6 stand, False if f7
    :param top: True if top Roll, False if bottom
    :param verbose: Parameter to define the quantity of written text
    """
    # Loading the DataSet
    dataset = datasets.StripsWearDiff(f6, top, random_seed=42, all_rolls=all_rolls)

    x_train, x_dev, x_test, y_train, y_dev, y_test = dataset.get_train_var()

    # Loading the network
    neural_net = RecurrentStripsNNDeltas(dataset,
                                         hidden_layer_sizes=layers_sizes,
                                         hidden_layer_activ=layers_activations,
                                         mask=mask)

    # Printing and training the network
    neural_net.summary()
    neural_net.fit(x_train, y_train, epochs=epochs, validation_data=(x_dev, y_dev), verbose=verbose, batch_size=16,
                   callbacks=[
                       EarlyStopping('val_loss', min_delta=0.1, patience=15, mode='min', restore_best_weights=True)])

    # We plot the results
    plot_training.wearcentre_predictions(neural_net, dataset)

    # We save and returns the results
    neural_net.save(savefile_name, )
    return neural_net


def load_neuralnet(savefile_name: str, f6: bool, top: bool, all_rolls: bool = False):
    """ Loads and evaluate a given neural network. Works for both Recurrent and non-recurrent networks

    :param all_rolls:
    :param savefile_name: The name of the filder in which the model should be saved
    :param f6: True if the Roll is from F6 stand, False if f7
    :param top: True if top Roll, False if bottom
    """
    # Creating dataset
    dataset = datasets.StripsWearCenter(f6, top, random_seed=42, all_rolls=all_rolls)

    # Loading Neural Net
    try:
        neural_net = StripsNN.load(savefile_name, dataset)
    except ValueError:
        print('Failed loading Fully Connected NN, trying Recurrent NN')
        neural_net = RecurrentStripsNN.load(savefile_name, dataset)

    # Plotting results
    plot_training.wearcentre_predictions(neural_net, dataset, savefile_name=savefile_name)
    return neural_net


def load_neuralnet_deltas(savefile_name: str, f6: bool, top: bool, all_rolls: bool = False):
    """ Loads and evaluate a given neural network. Works for both Recurrent and non-recurrent networks

    :param all_rolls:
    :param savefile_name: The name of the filder in which the model should be saved
    :param f6: True if the Roll is from F6 stand, False if f7
    :param top: True if top Roll, False if bottom
    """
    # Creating dataset
    dataset = datasets.StripsWearDiff(f6, top, random_seed=42, all_rolls=all_rolls)

    # Loading Neural Net
    try:
        neural_net = RecurrentStripsNN.load(savefile_name, dataset)
    except ValueError:
        print('Failed loading Recurrent NN, trying Fully Connected NN')
        neural_net = StripsNN.load(savefile_name, dataset)

    # Plotting results
    plot_training.wearcentre_predictions(neural_net, dataset, savefile_name=savefile_name)
    return neural_net
