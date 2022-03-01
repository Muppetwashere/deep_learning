"""
This file contains various test functions.
Those functions can be used to illustrate some results and have been used to illustrate the report.

@author: Antonin GAY (U051199)
"""

from time import time

import matplotlib.pyplot as plt

from .. import wearcentre_from_means
from ..data import datasets
from ..data.inputs import MeanCampaignsInputDB, StripsCentreInputDS, StripsFullProfileInputDS, \
    InputModes
from ..data.outputs import WearCentreOutputDB, FullProfileOutputDB, ThreePointsOutputDB
from ..models.strips import *


def test_linear_wearcentre():
    """ Trains a Linear Regressor to predict **Wear at the Centre** from **Means over campaigns** """

    wearcentre_from_means.linear_regressor(True, True, InputModes.SELECTED)
    plt.show(block=False)


def test_data_sets_classes(f6: bool = np.random.choice([0, 1]), top: bool = np.random.choice([0, 1])):
    """ This function tests that the DataSets can be correctly loaded """

    t0 = time()
    print('Testing conditions:'
          '\n\tf6 = %r'
          '\n\ttop = %r' % (f6, top))

    # Testing outputs
    print('\n*Testing Output DB:*')
    print('Testing WearCentreOutputDB :')
    print('\t%r' % WearCentreOutputDB(f6, top))
    print('Testing FullProfileOutputDS :')
    print('\t%r' % FullProfileOutputDB(f6, top))
    print('Testing ThreePointsOutputDS :')
    print('\t%r' % ThreePointsOutputDB(f6, top))

    # Testing inputs
    print('\n*Testing Input DB:*')
    print('Testing MeanCampaignsInputDB :')
    print('\t%r' % MeanCampaignsInputDB(f6, top, MeanCampaignsInputDB.SELECTED_ONEHOT_F6T))
    print('Testing StripsCentreInputDS :')
    print('\t%r' % StripsCentreInputDS(f6, top))
    print('Testing StripsFullProfileInputDS :')
    print('\t%r' % StripsFullProfileInputDS(f6, top, FullProfileOutputDB(f6, top)))

    # Testing DS
    print('\n*Testing Full DataSets:*')
    print('Testing MeanWearCenter :')
    print('\t%r' % datasets.MeanWearCenter(f6, top, MeanCampaignsInputDB.SELECTED_ONEHOT_F6T))
    print('Testing StripsProfile :')
    print('\t%r' % datasets.StripsProfile(f6, top))
    print('Testing StripsWearCenter :')
    print('\t%r' % datasets.StripsWearCenter(f6, top))
    print('Testing ThreePoints :')
    print('\t%r' % datasets.ThreePoints(f6, top))

    print('\nOverall Success in %.1fs !' % (time() - t0))
