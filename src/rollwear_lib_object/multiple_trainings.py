"""
This file contains function to test multiple trainings of Neural Networks.
Those functions are not meant to be used for actual training but for code testing and quick demonstrations

@author: Antonin GAY (U051199)
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt

from .data import datasets
from .models.strips import *
from .plot.training import print_results_mae


def get_path(savefile):
    """ Returns a Path object for a given filename

    :param savefile: Name of the savefile
    :return: Path to the file
    """
    Path('data/Outputs/MultiTrainResults/').mkdir(parents=True, exist_ok=True)
    if '.npy' in savefile:
        return Path('data/Outputs/MultiTrainResults/' + savefile)
    else:
        return Path('data/Outputs/MultiTrainResults/' + savefile + '.npy')


def load_file(savefile):
    """ Returns the matrix stored in a savefile

    :param savefile: Name of the savefile
    :return: Matrix stored in the savefile of dimension n*3
    """
    p = get_path(savefile)
    with p.open('rb') as f:
        fsz = os.fstat(f.fileno()).st_size
        out = np.load(f)
        while f.tell() < fsz:
            out = np.vstack((out, np.load(f)))
    return out


def get_savefile_name(hidden_layer_sizes, hidden_layer_activ, mask: Mask):
    """ Creates the normalized filename from the network parameters

    :param hidden_layer_sizes: Sizes of the hidden layers. Example (16, 16, 8, 4, 4, 4, 2)
    :param hidden_layer_activ: Activations of the layers. Example ('r', 'sigmoid', 'r', 'selu', 'sigmoid', 'sigmoid')
    :param mask: Mask applied to the inputs
    :return: Savefile name
    """
    # Creating savefile1 name
    savefile = ''
    savefile += 'Fw&l_' * mask.fw_and_l + \
                'Fw*l_' * mask.fwl + \
                'Str_' * mask.strips_param + \
                'Rol_' * mask.roll_param + \
                'HaI_' * mask.hardness_indic + \
                'Fam_' * mask.family + \
                'Sup_' * mask.suppliers + \
                'CLe_' * mask.cum_length
    for size in hidden_layer_sizes:
        savefile += str(size) + '_'
    for activ in hidden_layer_activ:
        savefile += str.upper(activ[0])
        try:
            savefile += str.lower(activ[1])
        except IndexError:
            pass

    return savefile


def trainings(epochs, hidden_layer_sizes, hidden_layer_activ, mask: Mask, number_try: int):
    """ Trains multiple times a Neural Network with the given parameters and saves the results in a file.
    This function is used to get multiple training results to compare the results of networks

    :param epochs: Number of passing of the Neural Network. Should be around 250. Can go up to 2000
    :param hidden_layer_sizes: Sizes of the hidden layers. Example (16, 16, 8, 4, 4, 4, 2)
    :param hidden_layer_activ: Activations of the layers. Example ('r', 'sigmoid', 'r', 'selu', 'sigmoid', 'sigmoid')
    :param mask: Mask applied to the inputs
    :param number_try: Number of trainings. Should not be greater than 5 as training time increase for each new training
    :return: Nothing
    """
    # Creating savefile name
    savefile = get_savefile_name(hidden_layer_sizes, hidden_layer_activ, mask)
    print('Savefile: %s' % savefile)

    # Loading the savefile
    p = get_path(savefile)
    if p.exists():
        print('ALREADY EXISTING SAVEFILE !')
        assert load_file(savefile).shape[-1] == 3

    # Loading the dataset and the network
    dataset = datasets.StripsWearCenter(True, True, random_seed=42)
    neural_net = StripsNN(dataset, hidden_layer_sizes=hidden_layer_sizes,
                          hidden_layer_activ=hidden_layer_activ, mask=mask)
    x_train, x_dev, _, y_train, y_dev, _ = dataset.get_train_var()

    # Trying to load the best historical NN saved if already existing
    try:
        _, mae_dev, mae_test = print_results_mae(StripsNN.load(savefile, dataset), dataset)
        best_mae = (2 * mae_dev + mae_test) / 3
        print('Historical Best MAE: %.1f' % best_mae)
    except FileNotFoundError:
        best_mae = np.inf

    # We do the training multiple times NB: Each training is longer than the previous one...
    for _ in tqdm(range(number_try), desc='Training multiple times'):
        # Training network
        neural_net.reset_training()
        neural_net.fit(x_train, y_train, epochs=epochs, validation_data=(x_dev, y_dev), verbose=0)

        # Saving the results
        mae_train, mae_dev, mae_test = print_results_mae(neural_net, dataset)
        # noinspection PyTypeChecker
        np.save(p.open('ab'), [mae_train, mae_dev, mae_test])

        # If the results are better than the historical, saving the network
        new_mae = (2 * mae_dev + mae_test) / 3
        if new_mae < best_mae:
            print('Best results yet (%.1fµm). Saving it to folder' % new_mae)
            neural_net.save(savefile, )
            best_mae = new_mae

        plt.close('all')  # Closing figures to limit the number of open figures

    # Finally, we plot the results
    plottings(savefile)


def plottings(savefiles: list, titles: list = None, x_max: int = 80):
    """ Plot multiples distribution of MAE

    :param savefiles: str or list of str: names of the savefiles (with or without '.npy')
    :param titles: str or list of str: Figure titles
    :param x_max: Maximum MAE in the horizontal axis. Default to 80µm, can be increased if bad networks
    :return: Nothing
    """

    def plot_one_file(save_file: str, title: str):
        """ Method to plot one file. Dev and Test are merged into one 'validation' set

        :param save_file: Filename
        :param title: Title of the figure
        """
        labels = ['train', 'val']
        out = load_file(save_file)
        out[:, 1] = 0.66 * out[:, 1] + 0.33 * out[:, 2]  # Merging Dev and Test

        # For each column, printing MAE and plotting results
        print('%s mean MAE : ' % save_file)
        for j in range(2):
            plt.hist(out[:, j], bins='auto', alpha=0.5, label=labels[j])
            print('\t%s: %.1f (+/- %.1f) µm' % (labels[j], out[:, j].mean(), out[:, j].std()))

        # Setting the plots parameters
        plt.xlim([0, x_max])
        plt.xlabel('Mean Absolute Error (µm)')
        plt.title(title)
        plt.legend()

    plt.figure()
    if type(savefiles) is list:
        # If titles is not a list or its len is different than the one of savefiles, titles=savefile
        if type(titles) is not list or len(titles) != len(savefiles):
            titles = savefiles

        plt.subplots_adjust(hspace=0.4)  # Enhancing the space between plots for readability
        for i, savefile in enumerate(savefiles):
            plt.subplot(len(savefiles), 1, i + 1)
            plot_one_file(savefile, titles[i])
    elif type(savefiles) is str:  # If only one file
        if type(titles) is not str:
            titles = savefiles
        # noinspection PyTypeChecker
        plot_one_file(savefiles, titles)
