import datetime

import streamlit as st

from roll_wear_lib_functional.data import load_raw_data, save_to_h5


def main_data_loading(path_savefile_h5):
    """ Load data from Excel, plot them in Streamlit and save them in h5 files.

    :param path_savefile_h5: Path to save the raw data in h5 file
    """

    # Loading inputs
    st.write('Input loading... Duration up to 2mn, started at %s' % datetime.datetime.now().strftime("%H:%M:%S"))

    try:
        # Defining paths
        path_input_data = 'Data/RawData/WearDataForDatamining.xlsx'
        path_output_data = 'Data/RawData/WearCentres.xlsx'

        # Loading data
        input_df, output_df = load_raw_data(path_input_data, path_output_data, verbose=True)

        # Plotting examples
        st.subheader('Inputs')
        st.write(input_df.sample(5))
        st.write(input_df.describe())

        st.subheader('Outputs')
        st.write(output_df.sample(5))
        st.write(output_df.describe())

        # Saving data
        save_to_h5(path_savefile_h5, input_df, output_df, verbose=True)
        st.write('Input loading... Done!\nData saved under %s' % path_savefile_h5)

    except FileNotFoundError:
        st.write('Input loading... Failed: file not found!')
