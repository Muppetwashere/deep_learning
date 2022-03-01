import datetime
import os

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from .metrics import get_mae_denormalized, get_mse_denormalized

models_save_folder = 'Data/models/'


def split_campaigns(inputs: pd.DataFrame, outputs: pd.DataFrame):
    """ Splits campaign IDs between train, dev and test sets, and returns the three IDs lists.
    The campaign IDs are obtained from common campaigns between inputs and outputs

    :param inputs: DataFrame of Inputs. Should have an index called 'id_campaign'
    :param outputs: DataFrame of Outputs. Should be with Simple Index

    :return: tuple (camp_train, camp_dev, camp_test) - lists of campaign IDs
    """
    campaigns_list = np.intersect1d(outputs.index, inputs.index.get_level_values('id_campaign').unique())

    camp_train, _camp_devtest = train_test_split(campaigns_list, test_size=0.33)
    camp_dev, camp_test = train_test_split(_camp_devtest, test_size=0.33)

    return camp_train, camp_dev, camp_test


def get_training_functions(camp_train, inputs: pd.DataFrame, outputs: pd.DataFrame, fwl: pd.DataFrame,
                           batch_size: int, metric_name: str = 'mae', return_out_scaler: bool = False):
    """ Returns the batch generator, the metric to use (mae or mse) and if asked, the output scaler.

    :param camp_train: Lists of train campaigns. Used to fit the scaler for outputs
    :param inputs: Inputs which will be used by the generator
    :param outputs: Outputs which will be used by the generator
    :param fwl: DataFrame of F/w *L values, which will be used by the generator and given to the model
    :param batch_size: Number of campaigns per batch
    :param metric_name: Name of the metric : 'mae' or 'mse'. Default is 'mae'
    :param return_out_scaler: if True, the output scaler is returned as a third argument.

    :return: tuple (campaign_generator, metric) + (output_scaler if asked)
            - campaign_generator: Generator to give to the fit() function to generate data batches
            - metric: Metric to give to the fit() function, adapted to campaigns data format. Returns error in µm(**2)
            - (output_scaler): MyScaler: Class to use to denormalize the outputs given by the generator
    """
    # Normalize data
    input_scaler, output_scaler, fwl_scaler, input_norm, output_norm, fwl_norm = \
        get_normalized_data(MyScaler, camp_train, inputs, outputs, fwl)
    fwl_norm = fwl_norm[['FwL F6', 'FwL F6', 'FwL F7', 'FwL F7']]

    # Define generator function
    def campaign_generator(camp_ids, _batch_size: int = batch_size, shuffle: bool = True,
                           output_dataframe: bool = False):
        """ Data generator for training.

        :param camp_ids: List of campaigns ID to use to generate batches
        :param _batch_size: Size of the batch to generate
        :param shuffle: Do the campaigns numbers get shuffled between epochs ?
        :param output_dataframe: if True, the output batch is returned as a DataFrame. If False, as a np array.
                For training with TF, an array is necessary, while a DataFrame keeps the information of the campaign ID

         Usage: give it to Model.fit :
         >>> model = tf.keras.Model()
         >>> model.fit(x=campaign_generator(camp_ids), \
                        steps_per_epochs=len(camp_ids)/batch_size)
         """

        if _batch_size is None:
            _batch_size = len(camp_ids)

        new_index = pd.MultiIndex.from_product([np.sort(camp_ids), np.arange(start=1, stop=307, step=1, dtype=int)],
                                               names=input_norm.index.names)
        input_gen = input_norm.reindex(new_index, fill_value=0, copy=True)
        fwl_gen = fwl_norm.reindex(new_index, fill_value=0, copy=True)
        assert fwl_gen.columns.array == ['FwL F6', 'FwL F6', 'FwL F7', 'FwL F7']

        i = 0
        while True:
            # Creating the list of IDs
            i += _batch_size
            list_ids_temp = camp_ids[i - _batch_size:i]

            if i >= len(camp_ids):
                if shuffle:
                    np.random.shuffle(camp_ids)
                i -= len(camp_ids)
                list_ids_temp = np.append(list_ids_temp, camp_ids[0: i])
            list_ids_temp.sort()

            # Create batches
            x_batch = input_gen.loc[list_ids_temp]  # x_batch.shape = (batch_size * seq_length, nb_param)
            fwl_batch = fwl_gen.loc[list_ids_temp]
            y_batch = output_norm.loc[list_ids_temp]
            y_batch = y_batch if output_dataframe else np.array(y_batch)

            yield {'input': np.array(x_batch), 'fwl_input': np.array(fwl_batch)}, y_batch

    # Create metric
    metric = get_mse_denormalized(output_scaler) if metric_name == 'mse' else get_mae_denormalized(output_scaler)

    # Return all
    if return_out_scaler:
        return campaign_generator, metric, output_scaler
    else:
        return campaign_generator, metric


def get_normalized_data(scaler_class, camp_train, inputs: pd.DataFrame, outputs: pd.DataFrame, fwl: pd.DataFrame):
    """ Normalize the data and returns the normalisers (scalers) for denormalisation

    :param scaler_class: Class to use for the scalers
    :param camp_train: list of train campaign IDs, used to fit the scalers
    :param inputs: DataFrame of inputs
    :param outputs: DataFrame of outputs
    :param fwl: DataFrame of F/w * L

    :return: tuple (input_scaler, output_scaler, fwl_scaler, input_norm, output_norm, fwl_norm)
    """
    input_scaler = scaler_class().fit(inputs.loc[camp_train])
    output_scaler = scaler_class().fit(outputs.loc[camp_train])
    fwl_scaler = scaler_class().fit(fwl.loc[camp_train])

    input_norm = pd.DataFrame(data=input_scaler.transform(inputs), index=inputs.index, columns=inputs.columns)
    output_norm = pd.DataFrame(data=output_scaler.transform(outputs), index=outputs.index, columns=outputs.columns)
    fwl_norm = pd.DataFrame(data=fwl_scaler.transform(fwl), index=fwl.index, columns=fwl.columns)

    return input_scaler, output_scaler, fwl_scaler, input_norm, output_norm, fwl_norm


def get_callbacks():
    """ Create the callbacks for Keras fit method.
    Includes : EarlyStop, TensorBoard

    :return: list of callbacks, which can directly be given to fit()
    """
    early_stop_callback = EarlyStopping('val_mae_micrometers', min_delta=0.1, patience=15, mode='min',
                                        restore_best_weights=True)
    log_dir = "Data\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=3)

    return [early_stop_callback, tensorboard_callback]


def save_model(model: tf.keras.Model, model_name: str, camp_train, camp_dev, camp_test):
    """ Saves the model to the given path. Save in the same file the train/dev/test splitting

    :param model: keras Model to save
    :param model_name: File name, used as save folder name
    :param camp_train: list of train campaign IDs
    :param camp_dev: list of dev campaign IDs
    :param camp_test: list of test campaign IDs
    """
    model_folder: str = models_save_folder + model_name + '/'
    model_savefile: str = model_folder + 'model.h5'

    os.makedirs(model_folder, exist_ok=True)
    model.save(model_savefile, include_optimizer=True)
    with h5py.File(model_savefile, 'a') as mod_file:
        mod_file['camp_train'] = camp_train
        mod_file['camp_dev'] = camp_dev
        mod_file['camp_test'] = camp_test


def get_available_models():
    """ List the available models to be loaded

    :return: list of model names
    """
    return os.listdir(models_save_folder)


class MyScaler:
    """ Custom scaler, which can be used to normalize data.
    It is a simple linear scaler for now.
    """

    def __init__(self):
        self.min = 0.0
        self.max = 1.0

    def fit(self, var):
        self.min = np.min(var, axis=0)
        self.max = np.max(var, axis=0)

        # We add 1.0 to the values of self.max where self.max == self.min, to avoid division by 0
        self.max = self.max + 1.0 * (self.max - self.min == 0.0)

        return self

    def transform(self, var):
        return (var - self.min) / (self.max - self.min)

    def fit_transform(self, var):
        self.fit(var)
        return self.transform(var)

    def inverse_transform(self, var):
        try:
            return var * (self.max - self.min) + self.min
        except ValueError:
            try:
                a = var * np.transpose((self.max - self.min))
                return a + np.transpose(self.min)
            except ValueError:
                return var * np.array(self.max - self.min) + np.array(self.min)
