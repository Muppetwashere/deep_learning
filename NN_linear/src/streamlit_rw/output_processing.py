import numpy as np
import pandas as pd
import streamlit as st

from roll_wear_lib_functional.data import load_from_h5_output, save_to_h5


def main_output_processing(path_h5_raw_data, path_h5_preprocessed_data):
    """

    :param path_h5_raw_data:
    :param path_h5_preprocessed_data:
    :return:
    """

    output_df: pd.DataFrame = load_from_h5_output(path_h5_raw_data)
    st.write('Original shape of output data : %s' % str(output_df.shape))

    if st.checkbox('Remove missing values', value=True):
        st.write('Removed campaigns:')
        st.write(output_df[output_df.isna().any(axis=1)])

        output_df.dropna(inplace=True)

    if st.checkbox('Remove Negative values', value=True):
        # noinspection PyTypeChecker
        negative_index = output_df[np.any((output_df < 0), axis=1)].index

        st.write('Removed campaigns:')
        st.write(output_df.loc[negative_index])

        output_df.drop(negative_index, inplace=True, errors='ignore')

    if st.checkbox('Remove hand-idolated outliers', value=True):
        # Original list : [25, 56, 86, 75, 93, 103, 131, 188, 257, 271, 365]
        null_camp = [25, 56, 86, 75, 93, 103, 131, 188, 257, 271, 365]
        st.write('The following campaigns has been manually found to have outliers values:\n%r' % null_camp)

        # Initialising empty dataframe
        tmp_df = pd.DataFrame(columns=output_df.columns)
        tmp_df.index.names = output_df.index.names

        # For all column in null columns, if it exists in output_df, we add it to tmp_df
        for campaign in null_camp:
            try:
                tmp_df = tmp_df.append(output_df.loc[campaign])
            except KeyError:
                pass

        st.write('Removed campaigns:')
        st.write(tmp_df)

        output_df.drop(null_camp, inplace=True, errors='ignore')

    save_to_h5(path_h5_preprocessed_data, output_data=output_df)
    st.write('The processed data have been saved. Final shape of output data : %s' % str(output_df.shape))
