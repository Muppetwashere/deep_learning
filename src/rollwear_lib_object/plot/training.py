import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import Model

from ..data import datasets
from ..data.abstact_ds import DataSet
from ..models.abstract_models import MyModel


def profiles_prediction(model: MyModel, dataset: datasets.StripsProfile):
    """ Plots the results of the training for profiles-dataset and models

    :param model:
    :param dataset:
    """

    def plot_8_profiles(campaign_idx):
        """ Plot the predictions of campaigns profiles, based on per-sample predictions

        :param campaign_idx: The idx of the campaigns to plot. Only 8 will be randomly kept
        """
        if len(campaign_idx > 8):
            campaign_idx = np.random.choice(campaign_idx, 8, replace=False)

        plt.figure()
        for i, cp_id in enumerate(campaign_idx):
            y_true = dataset.get_y(cp_id)
            y_pred = model.predict(dataset.get_x(cp_id))

            plt.subplot(2, 4, i + 1)
            plt.plot(np.array(y_true), label='Ground Truth')
            plt.plot(y_pred, label='Predictions')
            plt.legend()
            plt.title('Campaign n#%d' % cp_id)

    x_train, x_dev, x_test, y_train, y_dev, y_test = dataset.get_train_var()

    test_results = model.evaluate(x_test, y_test)
    print('Tests results:'
          '\n\tMAE: %f - MSE %f' % (test_results[0], test_results[1]))

    model.plot_history()

    plot_8_profiles(dataset.idx_campaign_train)
    plt.suptitle('/!\\ Train Dataset /!\\')

    plot_8_profiles(dataset.idx_campaign_dev)
    plt.suptitle('Dev Dataset')

    plot_8_profiles(dataset.idx_campaign_test)
    plt.suptitle('Test Dataset')


def print_results_mae(model: MyModel, dataset: DataSet):
    """ Print three MAE (µm) and returns them """
    denormalize = dataset.denormalize
    x_train, x_dev, x_test, y_train, y_dev, y_test = dataset.get_train_var()
    print('Training and testing results:')

    def print_one(x, y_true, name: str):
        y_true, y_pred = denormalize(y_true), denormalize(model.predict(x))
        mae_micromoeters = mean_absolute_error(1000 * y_true, 1000 * y_pred)
        print('\tMAE %s: %.1f µm' % (name, mae_micromoeters))
        try:
            return mae_micromoeters[0]
        except IndexError:
            return mae_micromoeters

    return print_one(x_train, y_train, 'train'), print_one(x_dev, y_dev, 'dev'), print_one(x_test, y_test, 'test'),


def wearcentre_predictions(model: MyModel, dataset: DataSet, savefile_name: str = None):
    try:
        model.plot_history()
    except AttributeError:
        print('Could not plot history')
    denorm = dataset.denormalize

    def plot_one_roll(y_tr, y_va, pred_tr, pred_va, name: str):
        mae_train, mae_val = mean_absolute_error(y_tr, pred_tr), mean_absolute_error(y_va, pred_va)

        plt.plot(y_tr, pred_tr, '.', alpha=0.25, label='train' + r' MAE =  %.1f $\mu$m' % mae_train)
        plt.plot(y_va, pred_va, '.', label='val' + r' MAE =  %.1f $\mu$m' % mae_val)
        plt.plot([0, y_tr.max()], [0, y_tr.max()], '--g')

        plt.title('Results of training - ' + name)
        plt.xlabel('Ground Truth (µm)')
        plt.ylabel('Predictions (µm)')
        plt.legend()

        if savefile_name is not None:
            df_tr = pd.DataFrame(index=y_tr.index)
            df_tr['Prediction (µm)'] = pred_tr
            df_tr['Ground Truth (µm)'] = y_tr

            df_va = pd.DataFrame(index=y_va.index)
            df_va['Prediction (µm)'] = pred_va
            df_va['Ground Truth (µm)'] = y_va

            df_tr.to_csv('Data/Outputs/Models/' + savefile_name + '/results_train_' + name + '.csv', index=True,
                         float_format='%.2f')
            df_va.to_csv('Data/Outputs/Models/' + savefile_name + '/results_val_' + name + '.csv', index=True,
                         float_format='%.2f')

    # Getting data
    x_train, x_dev, x_test, y_train, y_dev, y_test = dataset.get_train_var()
    x_val, y_val = np.concatenate([x_dev, x_test]), pd.concat([y_dev, y_test])
    pred_train, pred_val = model.predict(x_train), model.predict(x_val)

    # De-normalisation and setting to µm
    y_train, y_val, pred_train, pred_val = denorm(y_train), denorm(y_val), denorm(pred_train), denorm(pred_val)
    y_train, y_val, pred_train, pred_val = 1000 * y_train, 1000 * y_val, 1000 * pred_train, 1000 * pred_val

    # Plotting results, roll by roll
    if np.ndim(y_train) == 1:
        plot_one_roll(y_train, y_val, pred_train, pred_val, dataset.input.rollname)
    else:
        names = y_train.keys()
        plt.figure()
        for i in range(y_train.shape[-1]):
            plt.subplot(2, np.ceil(y_train.shape[-1] / 2), i + 1)
            key = names[i]
            plot_one_roll(y_train[key], y_val[key], pred_train[:, i], pred_val[:, i], names[i])

    return print_results_mae(model, dataset)


# Functions for plotting individual effects
def plot_individual_variable(x, var, dataset, neural_network, is_wear: bool):
    if is_wear:
        var_name = 'Wear'
        unit = 'µm'
    else:
        var_name = 'k coefficient'
        unit = 'AU'

    plt.figure()
    plt.hist(var, 100)
    plt.xlabel('Individual %s (%s)' % (var_name, unit))
    plt.ylabel('Number of values')
    plt.title('Histogram of the individual %s for the dev campaigns' % var_name)

    # noinspection PyUnresolvedReferences
    col_name = dataset.input.columns_name
    fig_family = plt.figure()
    plt.title("Dependency of the individual %s over the families" % var_name)
    plt.xlabel('Family (AU)')
    plt.ylabel('Individual %s (%s)' % (var_name, unit))

    fig_supplier = plt.figure()
    plt.title("Dependency of the individual %s over the roll suppliers" % var_name)
    plt.xlabel('Roll supplier (AU)')
    plt.ylabel('Individual %s (%s)' % (var_name, unit))

    fig = None
    subplot = 10
    for i in range(x.shape[1]):
        # If the parameter was kept in the model
        if neural_network.mask.mask[i]:
            # For parameters which are not OneHot
            if 'family_' in col_name[i]:
                plt.figure(fig_family.number)
                data = var[x[:, i] == 1]
                plt.plot([i] * data.size, data, label=col_name[i])
                plt.legend()
            elif 'f6t_' in col_name[i]:
                plt.figure(fig_supplier.number)
                data = var[x[:, i] == 1]
                plt.plot([i] * data.size, data, label=col_name[i])
                plt.legend()
            else:
                if subplot >= 5 or fig is None:
                    fig = plt.figure()
                    subplot = 1
                    plt.suptitle("Depedency of the individual %s over parameters" % var_name)
                plt.figure(fig.number)
                plt.subplot(2, 2, subplot)
                plt.xlabel('%s (SI)' % col_name[i])
                plt.ylabel('Individual %s (%s)' % (var_name, unit))
                plt.plot(x[:, i], var, '.')
                subplot += 1


def plot_individual_wears_and_coeffs(dataset, neural_network, camp_id: np.array = None):
    # NN and dataset
    camp_id = dataset.idx_campaign_dev.tolist() + dataset.idx_campaign_test.tolist() if camp_id is None else camp_id

    # Getting data
    camp_id.sort()
    x = dataset.get_x(camp_id)
    y_true = np.array(dataset.get_y(camp_id))
    y_pred = neural_network.predict(x)
    print('Example: Camp ID = %d | Y_true = %.1fµm | Y_pred = %.1fµm)' %
          (camp_id[0], 1000 * y_true[0], 1000 * y_pred[0]))

    # Predicting individual coeffs
    def get_pred(input_layer, output_layer):
        model = Model(input_layer, output_layer)
        model.compile('adam', 'mae')
        pred = np.array(model.predict(neural_network.reshape_inputs(x)))
        return pred.transpose((1, 0, 2))

    out_wc_i = neural_network.layers[-2].input
    w_i = get_pred(neural_network.input, out_wc_i)

    out_k = [mum_layer.input[0] for mum_layer in neural_network.layers[-2 - 306:-2]]
    k_i = get_pred(neural_network.input, out_k)

    # Keeping only non-zero strips
    idx_nonzero = np.where(w_i != 0)
    x = x[idx_nonzero[0:2]]
    k_i = k_i[idx_nonzero]
    w_i = w_i[idx_nonzero]

    # Plotting depedencies
    plot_individual_variable(x, k_i, dataset, neural_network, is_wear=False)
    plot_individual_variable(x, w_i, dataset, neural_network, is_wear=True)
