import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.activations \
    import softmax, elu, selu, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, exponential, linear
from tqdm import tqdm

from ..data import datasets
from ..data.inputs import MeanCampaignsInputDB


def plot_cum_sums(nb_profiles: int):
    """ Plot the cumulative sum of strips of multiple campaigns

    :param nb_profiles: Number of profiles' cumulative sums to plot
    """

    def get_cum_sums(_df_strips: pd.DataFrame):
        max_camp_id = _df_strips['N° CAMPAIGN'].max()

        linsp = np.linspace(-1000, 1000, 2001)
        deriv = np.zeros((max_camp_id, 2001))

        for row in tqdm(range(_df_strips.shape[0]), "Cumulative sum"):
            # We get the slab parameters
            camp = _df_strips["N° CAMPAIGN"][row]
            width = _df_strips["STRIP WIDTH"][row]
            shift = _df_strips["SHIFTINGF6"][row]

            half_width = int(np.ceil(width / 2))

            # We add the beginning and end of the slab to the derivative
            deriv[camp - 1][1001 - half_width + shift] += 1
            deriv[camp - 1][1001 + half_width + shift] -= 1

        return np.cumsum(deriv, axis=1), linsp

    strips = MeanCampaignsInputDB.load_strips()
    cum_sums, x = get_cum_sums(strips.loc[strips['N° CAMPAIGN'] < nb_profiles])
    y_1_5 = 1.5 * np.ones_like(cum_sums[0])

    for cum_sum in cum_sums:
        length_left = sum(cum_sum[0:1001] == 1)
        length_right = sum(cum_sum[1001:2000] == 1)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(x, cum_sum, label='Nb of slabs')
        plt.plot(x, y_1_5, '--', label='y = 1.5')
        plt.xlabel('Width, mm')
        plt.ylabel('Number of seen slabs')
        plt.title('Distribution of slabs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x, cum_sum, label='Nb of slabs')
        plt.plot(x, y_1_5, '--', label='y = 1.5')
        plt.xlabel('Width, mm')
        plt.ylabel('Number of seen slabs')
        plt.ylim([-.5, 2.5])
        plt.title('Zoom\n1-slab : %d (left) - %d (right)' % (length_left, length_right))
        plt.legend()


def plot_keras_activation():
    activ_list = [softmax, elu, selu, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, exponential, linear]

    x = np.linspace(-10, 10, 500)
    x_tf = tf.convert_to_tensor([x.tolist()])

    with tf.compat.v1.Session() as sess:
        fig = plt.figure()
        for i, f in enumerate(activ_list):
            ax: plt.Axes = fig.add_subplot(4, 3, i + 1)
            plt.plot(x, f(x_tf).eval(session=sess)[0])
            plt.title(f.__name__)

            ax.spines['left'].set_position(('data', 0.0))
            ax.spines['bottom'].set_position(('data', 0.0))
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')


def plot_all_profiles(max_profiles_to_plot: int = 395):
    """ Goes through all profiles and plot them """
    profile_ds = datasets.StripsProfile(True, True)
    wcentre_ds = datasets.MeanWearCenter(True, True, MeanCampaignsInputDB.SELECTED)

    dn_profile_y = profile_ds.denormalize
    dn_wcentre = wcentre_ds.denormalize

    campaigns = np.union1d(profile_ds.common_campaigns, wcentre_ds.common_campaigns)
    x_min, x_max = -1000, 1000

    for i, camp_id in enumerate(campaigns):
        plt.figure()

        # Plotting the profiles
        try:
            if camp_id in wcentre_ds.common_campaigns:
                # Wear at the Centre in µm
                wear_center = 1000 * dn_wcentre(wcentre_ds.get_y([camp_id]).values[0])
                plt.plot([-750, 750], [wear_center, wear_center], 'r', label='Wear at center')
                x_min, x_max = -750, 750

            if camp_id in profile_ds.common_campaigns:
                # Profile in µm
                y_profile = 1000 * dn_profile_y(profile_ds.get_y([camp_id]).reset_index(drop=True))
                # noinspection PyUnresolvedReferences
                x_coordinates = profile_ds.output.get_coordinates([camp_id]).reset_index(drop=True)
                plt.plot(x_coordinates, y_profile, 'b', label='Profile')
                x_min, x_max = x_coordinates.min(), x_coordinates.max()

        except IndexError:
            print('Error with #%d' % camp_id)

        plt.plot([-1000, 1000], [0, 0], ':k')
        plt.ylim([-100, 800])
        plt.xlim([x_min, x_max])
        plt.ylabel('Roll wear (µm)')
        plt.xlabel('Horizontal coordiante on the roll (mm)')
        plt.legend()
        plt.title('Campaign #%d' % camp_id)

        if (i + 1) % 10 == 0:
            plt.show()

        if i >= max_profiles_to_plot:
            break
