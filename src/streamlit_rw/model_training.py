import time

import numpy as np
import pandas as pd
import streamlit as st

from roll_wear_lib_functional import training
from roll_wear_lib_functional.data import load_from_h5
from roll_wear_lib_functional.metrics import strip_prediction_to_campaign_np_array, get_mae_denormalized
from roll_wear_lib_functional.models import get_model
from streamlit_rw.tqdm_streamlit import TQDMStreamlit, TQDMStreamlitCallback


def main_model_training(path_h5_preprocessed_data, path_h5_raw_data: str):
    # Tunable parameters
    model_name = st.sidebar.text_input("Name of the savefile", 'Test_model')

    training_type = st.sidebar.radio('Training type', ('Single run', 'Cross Validation'))
    n_fold = st.sidebar.number_input('Number of fold', min_value=2, value=5) if training_type == 'Cross Validation' \
        else 1
    # model_type = st.sidebar.selectbox('Model type', ('Recurrent Neural Net', 'Neural Net'))
    loss_name = st.sidebar.selectbox('Loss metric', ('mse - Mean Squared Error', 'mae - Mean Absolute Error'))[:3]
    epochs = st.sidebar.number_input('Number of training epochs (advised ~500)', min_value=0, value=500)
    batch_size = st.sidebar.number_input('Size of training batch (advised ~8)', min_value=0, value=8)

    st.write('Model name: %s - Loss: \'%s\' - Epochs %d' % (model_name, loss_name, epochs))

    if model_name == '':
        st.info('The model name is empty, please specify one before training !')

    else:
        if st.button('Train'):
            if training_type == 'Single run':
                single_training(path_h5_preprocessed_data, path_h5_raw_data, loss_name=loss_name, epochs=epochs,
                                batch_size=batch_size, model_name=model_name)

            else:
                cross_val_training(n_fold)


def single_training(path_h5_preprocessed_data: str, path_h5_raw_data: str, loss_name: str,
                    epochs: int, batch_size: int, model_name: str, in_streamlit: bool = True):
    # Loading data
    tr_progress = TQDMStreamlit(desc='Loading data', total=9, leave=True, remaining_est=False)
    input_df, output_df, fwl_df = load_from_h5(path_h5_preprocessed_data, path_h5_raw_data)

    # Splitting campaigns
    tr_progress.update(desc='Splitting campaigns')
    camp_train, camp_dev, camp_test = training.split_campaigns(input_df, output_df)

    # Normalizing data
    tr_progress.update(desc='Normalizing data')
    campaign_generator, loss, output_scaler = \
        training.get_training_functions(camp_train, input_df, output_df, fwl_df, batch_size,
                                        metric_name=loss_name, return_out_scaler=True)

    # Creating model
    tr_progress.update(desc='Creating model')
    model_full, model_nn = get_model(input_df, output_df)

    # Compiling model
    tr_progress.update(desc='Compiling model')
    metric = get_mae_denormalized(output_scaler)
    model_full.compile('adam', loss=loss, metrics=[metric])

    # Loading Callbacks
    tr_progress.update(desc='Loading Callbacks')
    time.sleep(0.1)
    callbacks = training.get_callbacks() + [TQDMStreamlitCallback()]

    # Training model
    tr_progress.update(desc='Training model')
    time.sleep(0.1)
    model_full.fit(x=campaign_generator(camp_train), steps_per_epoch=np.ceil(len(camp_train) / batch_size),
                   validation_data=campaign_generator(camp_dev, _batch_size=None), validation_steps=1,
                   epochs=epochs, verbose=not in_streamlit, use_multiprocessing=True, workers=6, callbacks=callbacks)

    # Saving model
    tr_progress.update(desc='Saving model')
    training.save_model(model_full, model_name, camp_train, camp_dev, camp_test)
    st.write('Model saved')

    # Saving dev results
    tr_progress.update(desc='Evaluating model')
    pred_dev = model_full.predict(campaign_generator(camp_dev, _batch_size=None), steps=1)
    pred_dev = strip_prediction_to_campaign_np_array(pred_dev)
    pred_test = model_full.predict(campaign_generator(camp_test, _batch_size=None), steps=1)
    pred_test = strip_prediction_to_campaign_np_array(pred_test)

    pred_df = pd.DataFrame(columns=output_df.columns)
    pred_df = pred_df.append(pd.DataFrame(data=pred_dev, index=camp_dev, columns=pred_df.columns))
    pred_df = pred_df.append(pd.DataFrame(data=pred_test, index=camp_test, columns=pred_df.columns))

    # Returning predictions for full DataSet
    tr_progress.update(desc='Done !')
    return pred_df, model_full, model_nn


# todo: write CrossValidation function
def cross_val_training(n_folds: int):
    return n_folds
