import pandas as pd
import streamlit as st

from roll_wear_lib_functional.data import load_from_h5_input, save_to_h5

supplier_description, family_description, lineup_description = 'Roll Supplier', 'Strip Families', 'Line Up'


def main_input_preprocessing(path_raw_data: str, path_processed_data: str):
    """

    :param path_raw_data:
    :param path_processed_data:
    """
    st.write('Select in the following list of available parameters for the strips the one you would like to use '
             'as inputs for the roll wear predictions, then press the button at the end of the list.')

    input_df: pd.DataFrame = load_from_h5_input(path_raw_data)
    columns_names = input_df.columns

    checkbox_dict = []
    supplier_list, family_list, lineup_list = [], [], []

    # For each column of the DataFrame, we create a single checkbox.
    # Except for suppliers, families and lineups which have a single checkbox per category
    for col_name in columns_names:
        if 'supplier' in col_name:
            supplier_list.append(col_name)
        elif 'family' in col_name:
            family_list.append(col_name)
        elif 'lineup' in col_name:
            lineup_list.append(col_name)
        else:
            checkbox_dict.append((col_name, st.checkbox(col_name)))

    checkbox_dict.append((supplier_description, st.checkbox(supplier_description)))
    checkbox_dict.append((family_description, st.checkbox(family_description)))
    checkbox_dict.append((lineup_description, st.checkbox(lineup_description)))

    if st.button('Apply selection'):
        selected_columns = columns_selection(checkbox_dict, supplier_list, family_list, lineup_list)

        if selected_columns:  # if selected columns not empty
            selected_input_df: pd.DataFrame = input_df[selected_columns]
            save_to_h5(path_processed_data, input_data=selected_input_df)

            st.write('Columns_selected and saved in file.\nSelected columns are : [%r]' % ', '.join(selected_columns))
        else:
            st.write('No columns were selected, nothing has been done')


def columns_selection(checkbox_list, supplier_list, family_list, lineup_list):
    """ This function gets the selected columns in the checkboxes and returns the new DataFrame with selected columns 

    :return: List of selected columns names
    :rtype: list
    """
    new_columns = []
    # We go through all our checkbox and check is they are checked
    for name, checkbox in checkbox_list:
        if checkbox:
            # # If the CheckBox is selected we add the corresponding column name to the new list
            # name = checkbox.value
            # If the name correspond to one of the conglomerate list, we add this list
            if name == supplier_description:
                new_columns.extend(supplier_list)
            elif name == family_description:
                new_columns.extend(family_list)
            elif name == lineup_description:
                new_columns.extend(lineup_list)
            # Otherwise, we add the checkbox name
            else:
                new_columns.append(name)

    return new_columns
