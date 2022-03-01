import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model

from roll_wear_lib_functional import data, training, models
from roll_wear_lib_functional.metrics import strip_prediction_to_campaign_np_array, \
    get_mae_denormalized, get_mse_denormalized
from streamlit_rw.tqdm_streamlit import TQDMStreamlit


def main_results_plotting(path_h5_preprocessed_data: str, path_h5_raw_data: str):
    st.header('Results plotting')
    st.warning('Be careful, a model can be plot only if the last data loading and processing '
               'are the same as the one used when the model was trained !')

    model_name = st.selectbox('Choose model by name', ['-'] + training.get_available_models(), index=0)
    if not model_name == '-':
        st.subheader(model_name)
        results_plotting(model_name, path_h5_preprocessed_data, path_h5_raw_data)


def results_plotting(model_name: str, path_h5_preprocessed_data: str, path_h5_raw_data: str):
    model_savefile: str = training.models_save_folder + model_name + '/model.h5'

    # Loading data
    tr_progress = TQDMStreamlit(desc='Loading data', total=7, leave=True, remaining_est=False)
    camp_train, camp_dev, camp_test, camp_val = load_camp_split(model_savefile)
    input_df, output_df, fwl_df = data.load_from_h5(path_h5_preprocessed_data, path_h5_raw_data)

    # Normalizing data
    tr_progress.update(desc='Normalizing data')
    campaign_generator, _, output_scaler = \
        training.get_training_functions(camp_train, input_df, output_df, fwl_df,
                                        batch_size=16, return_out_scaler=True)

    # Creating model
    tr_progress.update(desc='Loading model')
    mae_rw = get_mae_denormalized(output_scaler)
    mse_rw = get_mse_denormalized(output_scaler)
    model_full = load_model(model_savefile, {'DiagonalWeight': models.DiagonalWeight,
                                             'mae_micrometers': mae_rw, 'mse_micrometers': mse_rw})

    # Predicting outputs
    tr_progress.update(desc='Predicting Outputs')
    train_inputs, y_train = next(campaign_generator(camp_train, _batch_size=None, shuffle=False, output_dataframe=True))
    val_inputs, y_val = next(campaign_generator(camp_val, _batch_size=None, shuffle=False, output_dataframe=True))

    pred_train = model_full.predict(train_inputs, steps=1)
    pred_train = strip_prediction_to_campaign_np_array(pred_train)
    pred_val = model_full.predict(val_inputs, steps=1)
    pred_val = strip_prediction_to_campaign_np_array(pred_val)

    pred_train = pd.DataFrame(data=pred_train, index=camp_train, columns=output_df.columns)
    pred_val = pd.DataFrame(data=pred_val, index=camp_val, columns=output_df.columns)

    # Denormalizing outputs
    tr_progress.update(desc='Denormalizing outputs')
    denormalize = output_scaler.inverse_transform
    y_train, y_val = 1000 * denormalize(y_train), 1000 * denormalize(y_val)
    pred_train, pred_val = 1000 * denormalize(pred_train), 1000 * denormalize(pred_val)

    # Saving data
    tr_progress.update(desc='Saving predictions')
    save_predictions(y_train, y_val, pred_train, pred_val, model_name)

    # Plotting
    tr_progress.update(desc='Plotting results')
    plot_4_rolls_prediction(y_train, y_val, pred_train, pred_val)

    # Done !
    tr_progress.update(desc='Done !')


def load_camp_split(model_savefile):
    with h5py.File(model_savefile, 'r') as mod_file:
        camp_train = np.array(mod_file['camp_train'])
        camp_dev = mod_file['camp_dev']
        camp_test = mod_file['camp_test']
        camp_val = np.concatenate([camp_dev, camp_test])

    return camp_train, camp_dev, camp_test, camp_val


def save_predictions(y_train: pd.DataFrame, y_val: pd.DataFrame, pred_train: pd.DataFrame, pred_val: pd.DataFrame,
                     model_name: str):
    df_train = pd.DataFrame(index=y_train.index)
    df_val = pd.DataFrame(index=y_val.index)
    for key in y_train.columns:
        df_train[key + ' Prediction (µm)'] = pred_train[key]
        df_train[key + ' Ground Truth (µm)'] = y_train[key]

        df_val[key + ' Prediction (µm)'] = pred_val[key]
        df_val[key + ' Ground Truth (µm)'] = y_val[key]

    model_folder = training.models_save_folder + model_name + '/'
    os.makedirs(model_folder, exist_ok=True)
    df_train.to_csv(model_folder + 'res_train.csv', index=True, float_format='%.2f')
    df_val.to_csv(model_folder + 'res_val.csv', index=True, float_format='%.2f')


def plot_4_rolls_prediction(y_train: pd.DataFrame, y_val: pd.DataFrame, pred_train: pd.DataFrame,
                            pred_val: pd.DataFrame):
    for key in y_train.columns:
        plt.figure()

        # Train
        mae_train = mean_absolute_error(y_train[key], pred_train[key])
        plt.plot(y_train[key], pred_train[key], '.', alpha=0.25,
                 label=r'train MAE =  %.1f $\mu$m' % mae_train)

        # Val
        mae_val = mean_absolute_error(y_val[key], pred_val[key])
        plt.plot(y_val[key], pred_val[key], '.', alpha=1.,
                 label=r'val MAE =  %.1f $\mu$m' % mae_val)

        # Legends
        plt.plot([0, y_train[key].max()], [0, y_train[key].max()], '--g')
        plt.title('Results of training - %s' % key)
        plt.xlabel('Ground Truth (µm)')
        plt.ylabel('Predictions (µm)')
        plt.legend()

        st.pyplot()
