import os

import pandas as pd
import streamlit as st


# Raw data

def load_raw_data(path_input_data: str, path_output_data: str, verbose: bool = False):
    """ Load inputs and outputs from excel files located on the given paths

    :param path_input_data: path to the input Excel file
    :param path_output_data: path to the output Excel file
    :param verbose: 0 - no verbose; 1 - streamlit & print verbose
    :return: [input_df, output_df] DataFrames of inputs (with MultiIndex) and outputs
    """
    return load_raw_input(path_input_data, verbose), load_raw_output(path_output_data, verbose)


@st.cache(suppress_st_warning=True)
def load_raw_input(path_input_data: str, verbose: bool):
    """ Load inputs from an excel located in the given path

    :param path_input_data:
    :param verbose: 0 - no verbose; 1 - streamlit & print verbose
    :return: DataFrame of input, with MultiIndex
    :rtype: pd.DataFrame
    """

    text = None
    if verbose:
        text = st.text("Loading Input data from excel. About 2mn left")
        print("Loading Input data from excel. About 2mn left")

    # Loading raw strips data
    strips_df: pd.DataFrame = pd.read_excel(io=path_input_data, sheet_name='Strips_data', usecols='B, F:AP, AS:BN',
                                            index_col=[0, 1], header=2, skiprows=[3])
    strips_df.index.names = ['id_campaign', 'id_strip']  # Renaming the indexes

    # Data processing.
    # 1. We extract the families as one_hot vector
    strips_df = pd.get_dummies(strips_df, prefix=['family'], columns=['STIP GRADE FAMILY'])
    # 2. Oil flow rate is considered as ON/OFF
    strips_df['F6 Oil Flow Rate, ml/min'] = (strips_df['F6 Oil Flow Rate, ml/min'] > 0).astype(int)
    strips_df.rename(columns={'F6 Oil Flow Rate, ml/min': 'F6 Oil Flow Rate, on/off'}, inplace=True)

    # Loading campaigns data
    if verbose:
        text.text("Loading Input data from excel. About 1mn left")
        print("Loading Input data from excel. About 1mn left")

    camp_df: pd.DataFrame = pd.read_excel(io=path_input_data, sheet_name='Campaign_data', header=1, skiprows=[2],
                                          usecols='A, C:E, J:M, N:Q, R:U', index_col=0)
    camp_df.index.names = ['id_campaign']

    # We transform the line up and supplier columns into one_hot vectors
    camp_df = pd.get_dummies(camp_df, prefix=['lineup'], columns=['LINE_UP'])
    camp_df = pd.get_dummies(camp_df, prefix=['supplier_f6t', 'supplier_f6b', 'supplier_f7t', 'supplier_f7b'],
                             columns=['F6 TOP SUPPLIER', 'F6 BOT SUPPLIER', 'F7 TOP SUPPLIER', 'F7 BOT SUPPLIER'])

    if verbose:
        text.text('Loading Input data from excel: Done !')
        print('Loading Input data from excel: Done !')

    return strips_df.join(camp_df, how='inner')


@st.cache(suppress_st_warning=True)
def load_raw_output(path_output_data: str, verbose: int):
    """ Load outputs (i.e. wear at centre) from Excel file

    :param path_output_data: path to the output Excel file
    :param verbose: 0 - no verbose; 1 - streamlit & print verbose
    :return: DataFrame of outputs
    """

    text = None
    if verbose:
        text = st.text("Loading Output data from excel. Takes about 1mn")
        print("Loading Output data from excel. Takes about 1mn")

    # We read the data from the Excel file
    wearcenter_df: pd.DataFrame = pd.read_excel(io=path_output_data, sheet_name='Feuil1', usecols="A:E",
                                                header=2, skiprows=[3], index_col=0)

    # renaming columns
    wearcenter_df.rename(inplace=True, columns={'Usure F6 TOP': 'f6t', 'Usure F6 BOT': 'f6b',
                                                'Usure F7 TOP': 'f7t', 'Usure F7 BOT': 'f7b'})
    wearcenter_df.index.names = ['id_campaign']

    if verbose:
        text.text('Loading Output data from excel: Done !')
        print('Loading Output data from excel: Done !')

    return wearcenter_df


# Saving and loading from H5 files

def save_to_h5(path_savefile: str, input_data: pd.DataFrame = None, output_data: pd.DataFrame = None, verbose: int = 0):
    """ Saving inputs and outputs DataFrame to a .h5 file. If a DataFrame is equal to None, it will be ignored.

    :param path_savefile: Path to the savefileµ
    :param input_data: DataFrame of Inputs. With MultiIndex.
            None to ignore and only save/update Outputs
    :param output_data: DataFrame of Outputs
            None to ignore and only save/update Inputs
    :param verbose: 0 - no verbose; 1 - streamlit & print verbose
    """
    assert input_data is not None or output_data is not None, "No data were provided"

    try:
        os.mkdir(os.path.dirname(path_savefile))
    except FileExistsError:
        pass

    if input_data is not None:
        input_data.to_hdf(path_savefile, key='inputs')
    if output_data is not None:
        output_data.to_hdf(path_savefile, key='outputs')

    if verbose:
        st.text("Loading Output data from excel. Takes about 1mn")
        print("Loading Output data from excel. Takes about 1mn")


def load_from_h5(path_savefile: str, path_raw_data: str = 'Data/raw_data.h5'):
    """ Loading data from h5 file, returns inputs, outputs and fwl as DataFrames.
    Inputs and Outputs must have been saved before in the file. FwL is computed from inputs

    :param path_savefile: path to inputs and outputs savefile (can be preprocessed data)
    :param path_raw_data: path to raw data o compute fwl. As it should not be computed from normalized data,
            if the previous savepath is for preprocessed data, the raw data should be loaded from another file
    :type path_savefile: str
    :type path_raw_data: str

    :returns: tuple (input_df, output_df, fwl_df)
    """
    input_df = load_from_h5_input(path_savefile)
    output_df = load_from_h5_output(path_savefile)
    fwl_df = load_from_h5_fwl(path_raw_data)
    fwl_df = fwl_df[['FwL F6', 'FwL F7']]

    return input_df, output_df, fwl_df


@st.cache  # To be removed ?
def load_from_h5_input(path_savefile: str):
    """ Load the Inputs DatFrame from a savefile

    :param path_savefile: path to the savefile
    :return: input_df - DataFrame loaded from the file
    """
    return pd.DataFrame(pd.read_hdf(path_savefile, key='inputs'))


@st.cache  # To be removed ?
def load_from_h5_output(path_savefile: str):
    """ Load the Outputs DatFrame from a savefile

    :param path_savefile: path to the savefile
    :return: output_df - DataFrame loaded from the file
    """
    return pd.DataFrame(pd.read_hdf(path_savefile, key='outputs'))


@st.cache  # To be removed ?
def load_from_h5_fwl(path_raw_data: str):
    """ Load the FwL DatFrame from an raw input savefile

    :param path_raw_data: path to the savefile of raw inputs
    :return: fwl_df - DataFrame of computed F/w * L
    """
    input_df = load_from_h5_input(path_raw_data)
    fwl_df = input_df[['STRIP LENGTH F6 EXIT*', 'STRIP LENGTH F7 EXIT',
                       'STAND FORCE / WIDTH F6*', 'STAND FORCE / WIDTH F7*']]
    fwl_df['FwL F6'] = fwl_df['STRIP LENGTH F6 EXIT*'] * fwl_df['STAND FORCE / WIDTH F6*']
    fwl_df['FwL F7'] = fwl_df['STRIP LENGTH F7 EXIT'] * fwl_df['STAND FORCE / WIDTH F7*']

    return fwl_df
