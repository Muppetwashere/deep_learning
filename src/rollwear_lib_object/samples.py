import matplotlib.pyplot as plt

import src.rollwear_lib_object.wearcentre_from_strips as wc_strips
from .models.abstract_models import Mask


def report_examples(args):
    """ Calls functions depending on the arguments gave to Python when launching running main.py
    Those functions are used as examples for the report.

    :return: Nothing
    """

    # We define the mask for the tests
    mask = Mask(f6=True, top=True, fw_and_l=False, fwl=False, strips_param=False,
                roll_param=False, hardness_indic=False, family=True, suppliers=True, cum_length=False)

    # Printing all profiles
    if 'all_profiles' in args:
        all_profiles()

    # Fitting a linear regression on the *Mean* Dataset
    if 'linear_reg' in args:
        linear_regression_wear_centre(args)

    # Fitting a Neural Network on the Mean Dataset, either on all the inputs or on selected ones
    if 'means_wc' in args:
        wear_center_predicted_from_means(args)

    # Plotting the hierarchisation of the input variables
    if 'hierarchisation' in args:
        hierarchisation(args)

    # Training or loading a Neural Network on all the strips
    if 'strips_nn' in args:
        strips_neural_net(args, mask)

    # Training or loading a Recurrent Neural Network on all the strips
    if 'strips_rnn' in args:
        strips_recurrent_neural_net(args, mask)

    # Training a Neural Network to fit the full profiles
    if 'full_profile' in args:
        full_profiles_neural_net()

    if 'plot_4_res' in args:
        plot_4_results()


def all_profiles():
    """ Sample function plotting all the profiles """
    import src.rollwear_lib_object.plot.demos

    src.rollwear_lib_object.plot.demos.plot_all_profiles()


def linear_regression_wear_centre(args):
    """ Example function fitting a linear regression to predict 'wear at the centre' """
    import src.rollwear_lib_object.wearcentre_from_means
    from src.rollwear_lib_object.wearcentre_from_means import InputModes

    if 'complete' in args:
        src.rollwear_lib_object.wearcentre_from_means.linear_regressor(f6=True, top=True, mode=InputModes.COMPLETE)
    else:
        src.rollwear_lib_object.wearcentre_from_means.linear_regressor(f6=True, top=True, mode=InputModes.SELECTED)


def wear_center_predicted_from_means(args):
    """ Use a Neural Network to predict the wear at the centre based on means of the strips parameters """
    import src.rollwear_lib_object.wearcentre_from_means
    from src.rollwear_lib_object.wearcentre_from_means import InputModes

    if 'complete' in args:
        src.rollwear_lib_object.wearcentre_from_means.training(
            epochs=250, layers_sizes=(20, 8, 4, 4), layers_activations=('selu', 'selu', 'selu', 'selu'),
            f6=True, top=True, mode=InputModes.COMPLETE, verbose=1)
    else:
        src.rollwear_lib_object.wearcentre_from_means.training(
            epochs=250, layers_sizes=(20, 8, 4, 4), layers_activations=('selu', 'selu', 'selu', 'selu'),
            f6=True, top=True, mode=InputModes.SELECTED, verbose=1)


def hierarchisation(args):
    """ Do and plots a parameters hierarchisation """
    from .tests import plots
    from src.rollwear_lib_object.wearcentre_from_means import InputModes

    if 'complete' in args:
        plots.test_hierarchisation(InputModes.COMPLETE)
    else:
        plots.test_hierarchisation(InputModes.SELECTED)


def strips_neural_net(args, mask: Mask):
    """ Predicts the wear at the centre from the list of strips of each campaign with a Neural Network """
    import src.rollwear_lib_object.wearcentre_from_strips

    if 'load' in args:
        src.rollwear_lib_object.wearcentre_from_strips.load_neuralnet('Fam_Sup_20_8_SeSeSi', f6=True, top=True)
    else:
        src.rollwear_lib_object.wearcentre_from_strips.train_neuralnet(
            250, (20, 8), ('selu', 'selu', 'sigmoid'), 'test',
            mask=mask, recurrent=False, f6='f7' not in args, top='bottom' not in args)


def strips_recurrent_neural_net(args, mask: Mask):
    """ Predicts the wear at the centre from the list of strips of each campaign with a Recurrent Neural Network """
    import src.rollwear_lib_object.wearcentre_from_strips

    if 'load' in args:
        src.rollwear_lib_object.wearcentre_from_strips.load_neuralnet('Recurrent', f6=True, top=True)
    else:
        src.rollwear_lib_object.wearcentre_from_strips.train_neuralnet(
            250, (20, 8), ('selu', 'selu', 'sigmoid'), 'test',
            mask=mask, recurrent=True, f6=True, top=True)


def full_profiles_neural_net():
    """ Tries to predict the full wear profile from all the strips of the campaign with a Neural Network """
    from .tests import neural_nets

    neural_nets.test_neuralnet_fullprofile(250)


def plot_4_results():
    """ Plots the results of four individual trainings. Those are parts of the best ones """

    try:
        wc_strips.load_neuralnet('f6_bot', True, False)
        wc_strips.load_neuralnet('f6_top', True, True)
        wc_strips.load_neuralnet('f7_bot', False, False)
        wc_strips.load_neuralnet('f7_top', False, True)
        plt.show()
    except FileNotFoundError:
        print('Impossible to load networks: they do not exist')
