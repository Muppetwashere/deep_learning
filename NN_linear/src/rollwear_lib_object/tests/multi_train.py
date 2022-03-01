"""
This file contains function to test multiple trainings of Neural Networks.
Those functions are not meant to be used for actual training but for code testing and quick demonstrations

@author: Antonin GAY (U051199)
"""

import matplotlib.pyplot as plt

from .. import multiple_trainings
from ..models.strips import Mask

defaultMask: Mask = Mask(True, True,
                         False, False, False, False, False, True, True, False)


def test_training():
    """ Function to test the multiple train.
    Should only be used for testing if the code works and for quick demonstrations """
    multiple_trainings.trainings(250, (20, 8), ('selu', 'selu', 'sigmoid'), defaultMask, 3)
    plt.plot(block=False)


def test_plotting():
    """ Function to test the multiple train plotting.
    Should only be used for testing if the code works and for quick demonstrations """
    multiple_trainings.plottings(['Fam_Sup_20_8_SeSeSi.npy', 'HaI_Sup_20_8_SeSeSi.npy'],
                                 ['With Families & Suppliers', 'With Hardness Indicator & Supliers'])
    plt.plot(block=False)
