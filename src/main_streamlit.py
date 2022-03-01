import datetime

# import tensorflow as tf
import streamlit as st

from streamlit_rw import data_loading, input_processing, output_processing, model_training, results_plotting, \
    streamlit_demos

# # noinspection PyProtectedMember
# tb._SYMBOLIC_SCOPE.value = True

st.title('Streamlit demo - Roll Wear prediction')
st.text('Choose a category on the left pane to start\nSession initiated on %s' % datetime.datetime.now().strftime(
    "%Y.%m.%d-%H:%M:%S"))

category = st.sidebar.radio('Choose the operation you would like to do',
                            ('None', 'Streamlit Demos', 'Data loading from Excel', 'Input pre-processing',
                             'Output pre-processing', 'Model training', 'Results plotting'))

# Files location
path_h5_raw_data = 'Data/raw_data.h5'
path_h5_preprocessed_data = 'Data/preprocessed_data.h5'

# StreamLit demos
if category == 'Streamlit Demos':
    st.header('Streamlit Demos')
    st.sidebar.header('Streamlit Demos')
    streamlit_demos.main_streamlit_demos()

# Data Loading
if category == 'Data loading from Excel':
    st.header('Data loading from Excel')
    st.sidebar.header('Data loading from Excel')
    data_loading.main_data_loading(path_h5_raw_data)

# Input processing
if category == 'Input pre-processing':
    st.header('Input processing')
    st.sidebar.header('Input processing')
    input_processing.main_input_preprocessing(path_h5_raw_data, path_h5_preprocessed_data)

# Output processing
if category == 'Output pre-processing':
    st.header('Output processing')
    st.sidebar.header('Output processing')
    output_processing.main_output_processing(path_h5_raw_data, path_h5_preprocessed_data)

# Model training
if category == 'Model training':
    st.header('Model training')
    st.sidebar.header('Model training')
    model_training.main_model_training(path_h5_preprocessed_data, path_h5_raw_data)

# Results plotting
if category == 'Results plotting':
    st.sidebar.header('Results plotting')
    results_plotting.main_results_plotting(path_h5_preprocessed_data, path_h5_raw_data)
