"""
This file contains test functions for plotting various results. Those functions can be used to illustrate some results
and have been used to illustrate the report.

@author: Antonin GAY (U051199)
"""

import matplotlib.pyplot as plt
import numpy as np

from .. import wearcentre_from_strips
from ..data import datasets as datasets
from ..data.inputs import InputModes
from ..plot.demos import plot_all_profiles
from ..plot.training import plot_individual_wears_and_coeffs


def test_hierarchisation(mode=InputModes.SELECTED):
    """ Plots the results of parameters hierarchisation proposed by Bernard Beauzamy.
    The effects of the parameters are evaluated based on the difference between two histograms of the outputs: one
    when the inputs are above the median, one when they are below.

    :param mode: Which inputs should be tested. Most interesting are 'input_modes.SELECTED' and 'input_modes.COMPLETE'
    """
    # Loading data
    dataset = datasets.MeanWearCenter(True, True, mode)
    denormalize = dataset.denormalize
    x = dataset.get_x_dataframe()
    y = 1000 * denormalize(dataset.get_y())  # µm
    bins = np.linspace(np.percentile(y, 5), np.percentile(y, 95), 100)

    # Plotting a total histogram
    plt.figure()
    plt.hist(y, 'auto')
    plt.title('Histogram of Wear at the centre')
    plt.xlabel('Wear at the centre (µm)')
    plt.legend(['Mean: %.1fµm (%s %.1fµm)' % (y.mean(), chr(177), y.std())])

    # Plotting differences histograms
    for key in x.keys():
        plt.figure()

        plt.hist(y[x[key] >= x[key].median()], bins, cumulative=True, label='X %s median' % chr(8805),  # 'X ≥ median'
                 alpha=0.5, density=True)
        plt.hist(y[x[key] <= x[key].median()], bins, cumulative=True, label='X %s median' % chr(8804),  # 'X ≤ median'
                 alpha=0.5, density=True)

        plt.title('Cumulative histogram comparison for :\n%s' % key)
        plt.xlabel('Wear at the centre (µm)')
        plt.legend()

    # Plotting the Area Under the Curve differences.
    dataset.compute_auc()
    plt.show()


def test_plot_dependencies():
    """ Plots the dependency of the outputs against each input """
    # Loading data
    dataset = datasets.MeanWearCenter(True, True, InputModes.SELECTED)
    x = dataset.get_x_dataframe()
    y = dataset.get_y()

    # Plotting all dependencies
    for key in x.keys():
        plt.figure()
        plt.plot(x[key], y, '.')

        plt.title(key)
        plt.ylabel('Wear at the centre / strip (mm)')

    # Plotting the hierarchisation results
    dataset.compute_auc()
    plt.show()


def test_plot_individual_pred():
    """ Plots an example of Neural Net results and dependencies """
    neural_net = wearcentre_from_strips.load_neuralnet('Fam_Sup_20_8_SeSeSi', True, True)
    dataset = datasets.StripsWearCenter(True, True, random_seed=42)

    plot_individual_wears_and_coeffs(dataset, neural_net)


def test_plot_all_profiles():
    """ Plots all the profiles """
    plot_all_profiles()


if __name__ == '__main__':
    test_hierarchisation()
