"""
This file defines Abstract Classes for DataBases and Datasets.
Those Abstract Classes set the standard forms and methods for further Datasets.

A DatBase (DB) here is the matrix of values of one Input or one Output. The MotherClass InOutDB defines the methods
Inheritors specify Inputs and Outputs cases.

A DataSet (DS) is composed of two DB, one for Input, one for Output.

Specific classes are defined for Strips cases.

@author: Antonin GAY (U051199)
"""

import os
from abc import ABC
from os.path import isfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# todo: make sure ampaign ids are ints
class InOutDB(ABC):
    """
    Abstract Class for DataBases. It defines the campaigns_ids, savefile, variables, etc.
    Used as MotherClass for following classes.
    """

    def __init__(self, savedir: str, savefile: str):

        # Initialisation of inherited attributes
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        self.savefile: str = savedir + savefile

        # Initialization of variables automatically computed from methods
        # noinspection PyTypeChecker
        self.campaign_ids: np.ndarray = None  # The campaign ID corresponding to each sample
        self.var = None
        self.norm_factor = None

        # Defining previous variables
        self.load()
        self.remove_outliers()
        self.normalize()

        assert self.var is not None and self.campaign_ids is not None and self.norm_factor is not None, \
            "One of the variables has not been initialized properly"

    def load(self):
        """ Should load self.var and self.campaign_ids.
        If file not found, calls self.extract_from_file """
        pass

    def extract_from_raw_data(self):
        """ Should load self.var and self.campaign_ids from raw data file.
        Then save it in an easier to load savefile"""
        pass

    def split(self, idx_campaign_train, idx_campaign_dev, idx_campaign_test):
        """ Should split self.var in train/dev/test depending on the given campaign ids """
        return \
            self.get_campaigns(idx_campaign_train), \
            self.get_campaigns(idx_campaign_dev), \
            self.get_campaigns(idx_campaign_test)

    def get_campaigns(self, idx_campaigns):
        """ Return the data corresponding to the given campaigns """
        return self.var[np.in1d(self.campaign_ids, idx_campaigns)]

    def normalize(self):
        """ Should normalize self.var and define the normalization factor """
        if np.ndim(self.var) <= 2:
            self.norm_factor = np.max(np.abs(self.var), axis=0)
            # Replacing 0s with 1s to avoid DIV0
            self.norm_factor = np.where(self.norm_factor == 0, np.ones_like(self.norm_factor), self.norm_factor)
        else:
            axis_for_max = tuple(range(max(np.ndim(self.var) - 1, 1)))  # We average over all the axis but the last one
            self.norm_factor = np.max(np.abs(self.var), axis=axis_for_max)
            # We replace all the potential 0s by 1s to avoid 0div
            self.norm_factor[self.norm_factor == 0] = 1
        self.var = self.var / self.norm_factor

    def denormalize(self, var, _=None):
        """ Should denormalize the given variable. Can take a second input if necessary """
        return var * self.norm_factor

    def remove_outliers(self):
        """ Remove the outliers of the data and campaigns """

        # Removing the lines with NaN values
        nan_rows = np.where(np.isnan(self.var))[0]
        self.drop_rows(nan_rows)

    def drop_rows(self, rows_index):
        """ Should remove the rows from self.var and self.campaigns """
        mask = np.ones(len(self.campaign_ids), dtype=bool)
        mask[rows_index] = False
        self.var = self.var[mask]
        self.campaign_ids = self.campaign_ids[mask]


class RollWearDB(InOutDB, ABC):
    """ DataBase specific to Roll Wear applications. 
    Main additions are roll name and position """

    def __init__(self, savedir: str, savefile: str, f6: bool, top: bool, all_rolls: bool = False):
        """ DataBase for Roll Wear applications, storing roll information.
        
        :param savedir: Save directory
        :param savefile: Save file name
        :param f6: If the roll is in F6 or F7
        :param top: If the roll is top or bottom
        """

        # Position of the roll (f6 or f7, top or bottom)
        self.f6 = f6
        self.top = top
        self.all_rolls = all_rolls
        # We create the name, 'f6', 'f6b', etc.
        self.rollname = 'all_rolls' if self.all_rolls else 'f6' * self.f6 + 'f7' * (
            not self.f6) + 't' * self.top + 'b' * (not self.top)

        super(RollWearDB, self).__init__(savedir=savedir, savefile=savefile)


class InputDB(RollWearDB, ABC):
    """ Database for Inputs """
    savefile_strips = 'Data/DataBases/Strips.feather'  # SaveFile for strips raw data
    rawfile_strips = 'Data/RawData/WearDataForDatamining.xlsx'

    def __init__(self, savefile, f6: bool, top: bool, all_rolls: bool = False):
        """ Creating an Input DataBase
        
        :param savefile: Location of DataBase save
        :param f6: If the roll is in F6 or F7
        :param top: If the roll is top or bottom
        """
        self.var: np.array = None  # For inputs, data are stored in numpy arrays
        super(InputDB, self).__init__(savedir='Data/DataBases/InputsDB/',
                                      savefile=savefile, f6=f6, top=top, all_rolls=all_rolls)

    def load(self):
        """ Loading the data, either from savefile, either from raw data """

        try:
            # Trying loading from savefile, if alerady previously computed.
            file = np.load(self.savefile)
            self.var: np.array = file[self.rollname]
            self.campaign_ids: np.array = file[self.rollname + '_campaigns']
        except (FileNotFoundError, KeyError):
            # Otherwise, extracting from raw data
            self.extract_from_raw_data()

    def extract_from_raw_data(self):
        # After extracting it, we save the data to a .npz file
        self.save_to_npz()
        super(InputDB, self).extract_from_raw_data()

    @classmethod
    def load_strips(cls):
        """ Load DataFrame from HF5 file. If file does not exist, load excel and save as HF5

        :return x_strips: DataFrame
        """
        if isfile(cls.savefile_strips):
            return pd.read_feather(cls.savefile_strips)
        else:
            return cls._excel2strips()

    @classmethod
    def _excel2strips(cls):
        """ Load the data of one excel file of input data
        https://datacarpentry.org/python-ecology-lesson/05-merging-data/ """

        file = cls.rawfile_strips

        print("Loading Input data from excel. About 2mn left")
        # We read the data from the Excel file
        # Strips data
        strips_df: pd.DataFrame = pd.read_excel(io=file, sheet_name='Strips_data',
                                                header=2, usecols='B:AP, AS:BN', skiprows=[3])

        # We extract the families as one_hot vector
        strips_df = pd.get_dummies(strips_df, prefix=['family'], columns=['STIP GRADE FAMILY'])
        strips_df['F6 Oil Flow Rate, ml/min'] = (strips_df['F6 Oil Flow Rate, ml/min'] > 0).astype(int)
        strips_df.rename(columns={'F6 Oil Flow Rate, ml/min': 'F6 Oil Flow Rate, on/off'}, inplace=True)

        print("Loading Input data from excel. About 1mn left")
        # Campaign data (with supplier information)
        camp_df: pd.DataFrame = pd.read_excel(io=file, sheet_name='Campaign_data',
                                              header=1, usecols='A, C:E, J:M, N:Q, R:U', skiprows=[2])

        # We transform the supplier rows into one_hot vectors
        camp_df = pd.get_dummies(camp_df, prefix=['lineup'], columns=['LINE_UP'])

        camp_df = pd.get_dummies(camp_df, prefix=['f6t', 'f6b', 'f7t', 'f7b'],
                                 columns=['F6 TOP SUPPLIER', 'F6 BOT SUPPLIER', 'F7 TOP SUPPLIER', 'F7 BOT SUPPLIER'])

        strips_df = pd.merge(left=strips_df, right=camp_df, left_on="N° CAMPAIGN", right_on="N° CAMPAIGN")

        strips_df.to_feather(cls.savefile_strips)
        return strips_df

    def save_to_npz(self):
        """ Saves the matrix into file """
        # We load the file in case it would already exist
        try:
            prev_mx = np.load(self.savefile)
            args = {name: prev_mx[name] for name in prev_mx.files}
        except FileNotFoundError:
            args = {}

        # We add the new value and save the file
        args.update({self.rollname: self.var})
        args.update({self.rollname + '_campaigns': self.campaign_ids})
        np.savez_compressed(self.savefile, **args)


class OutputDB(RollWearDB, ABC):
    """ Database for Outputs """
    null_camp = []  # List of campaigns ID for null campaigns

    def __init__(self, savefile, f6: bool, top: bool, all_rolls: bool = False):
        """ Creating an Output DataBase

        :param savefile: Location of DataBase save
        :param f6: If the roll is in F6 or F7
        :param top: If the roll is top or bottom
        """
        # noinspection PyTypeChecker
        self.var: pd.DataFrame = None  # For outputs, data are stored in pandas dataframe
        super(OutputDB, self).__init__(savedir='Data/DataBases/OutputsDB/',
                                       savefile=savefile, f6=f6, top=top, all_rolls=all_rolls)

    def load(self):
        """ Loading the data, either from savefile, either from raw data """

        if isfile(self.savefile):
            # Trying loading from savefile, if alerady previously computed.
            dataframe: pd.DataFrame = pd.read_feather(self.savefile)
            self.extract_from_dataframe(dataframe)
        else:
            # Otherwise, extracting from raw data
            self.extract_from_raw_data()

    def extract_from_dataframe(self, dataframe: pd.DataFrame):
        """ Extracts self.var and self.campaign_ids from dataframe """
        if self.all_rolls:
            self.var = dataframe[['f6t', 'f6b', 'f7t', 'f7b']]
        else:
            self.var = dataframe[self.rollname]
        self.campaign_ids = dataframe['N° CAMPAIGN']

    def remove_outliers(self):
        super(OutputDB, self).remove_outliers()

        # Removing too high rows (outliers)
        # noinspection PyTypeChecker
        outliers_rows = np.where(self.var > 1.0)[0]
        self.drop_rows(outliers_rows)

        # Removing the rows identified as null or negatives
        null_rows = np.where(np.in1d(self.campaign_ids, self.null_camp))
        self.drop_rows(null_rows)


class DataSet(object):
    """ Dataset class, defined by an input and an output databases """

    def __init__(self, input_db: InputDB, output_db: OutputDB, validation_split: float, random_seed: int):
        """ Initialise the DataSet with an input and an output.

        :param input_db: The Input DataBase
        :param output_db: The Output DataBase
        :param validation_split: Proportion of validation samples
        :param random_seed: Seed for splitting
        """
        self.input: InputDB = input_db
        self.output: OutputDB = output_db

        # We only consider common campaigns between input and output
        self.common_campaigns = np.intersect1d(self.input.campaign_ids, self.output.campaign_ids)

        # Splitting
        self.idx_campaign_train, self.idx_campaign_dev, self.idx_campaign_test = None, None, None
        self.split(validation_split, random_seed)

    def split(self, validation_split: float, random_seed: int):
        """ Splitting the campaigns into train, dev and test """
        self.idx_campaign_train, idx_campaign_dev_test = \
            train_test_split(self.common_campaigns, test_size=validation_split, random_state=random_seed)
        self.idx_campaign_dev, self.idx_campaign_test = \
            train_test_split(idx_campaign_dev_test, test_size=validation_split, random_state=random_seed)

    def get_train_var(self):
        """ Returns all the training data

        :return: x_train, x_dev, x_test, y_train, y_dev, y_test
        """
        x_train, x_dev, x_test = \
            self.input.split(self.idx_campaign_train, self.idx_campaign_dev, self.idx_campaign_test)
        y_train, y_dev, y_test = \
            self.output.split(self.idx_campaign_train, self.idx_campaign_dev, self.idx_campaign_test)

        return x_train, x_dev, x_test, y_train, y_dev, y_test

    def denormalize(self, y, x=None):
        return self.output.denormalize(y, x)

    def get_x(self, campaigns_idx: list = None):
        """ Return the complete output.
        If a list of campaigns is given, returns only the selected campaigns, if they are in the common campaigns """
        if campaigns_idx is None:
            # If no list given, we return the common campaigns
            return self.input.get_campaigns(self.common_campaigns)
        else:
            # Otherwise, we return the asked campaigns within the common campaigns
            return self.input.get_campaigns(np.intersect1d(campaigns_idx, self.common_campaigns))

    def get_y(self, campaigns_idx: list = None):
        """ Return the complete output.
        If a list of campaigns is given, returns only the selected campaigns, if they are in the common campaigns """
        if campaigns_idx is None:
            # If no list given, we return the common campaigns
            return self.output.get_campaigns(self.common_campaigns)
        else:
            # Otherwise, we return the asked campaigns within the common campaigns
            return self.output.get_campaigns(np.intersect1d(campaigns_idx, self.common_campaigns))


# Strips Specific Classes
class StripsInputDB(InputDB, ABC):
    """ Abstract Input DataBase for Strips data """

    # Parameters to store. Suppliers are stored as One Hot vector, forbid any new supplier.
    columns_f6_fwl = ['STAND FORCE / WIDTH F6*', 'STRIP LENGTH F5 EXIT*']
    columns_f6 = ['STRIP HARDNESS INDICATOR', 'BENDING FORCE F6', 'TEMPERATURE F6 EXIT', 'LEAD  SPEED F6',
                  'TRAIL SPEED F6', 'REDUCTION F6*', 'F6 Oil Flow Rate, on/off', 'CUMULATIVE ROLLING LENGTH F6*']

    columns_f7_fwl = ['STAND FORCE / WIDTH F7*', 'STRIP LENGTH F6 EXIT*']
    columns_f7 = ['STRIP HARDNESS INDICATOR', 'BENDING FORCE F7', 'TEMPERATURE F7 EXIT', 'LEAD SPEED F7',
                  'TRAIL SPEED F7', 'REDUCTION F7*', 'CUMULATIVE ROLLING LENGTH F7*']

    columns_f6t = ['CONTACT LENGTH F6 TOP*', 'F6 TOP DIAMETER', 'F6 TOP HARDNESS',
                   'f6t_Akers National Micra X', 'f6t_ESW IRON (VIS)', 'f6t_ESW VANIS',
                   'f6t_Kubota ECC-CX2 Type', 'f6t_National ICON', 'f6t_OZPV (LPHNMD-80)',
                   'f6t_Union Electric UK Apex Alloy', 'f6t_Villares Vindex VRP0313']
    columns_f6b = ['CONTACT LENGTH F6  BOT*', 'F6 BOT DIAMETER', 'F6 BOT HARDNESS',
                   'f6b_Akers National Micra X', 'f6b_ESW IRON (VIS)', 'f6b_ESW VANIS',
                   'f6b_Kubota ECC-CX2 Type', 'f6b_National ICON', 'f6b_OZPV (LPHNMD-80)',
                   'f6b_Union Electric UK Apex Alloy', 'f6b_Villares Vindex VRP0313']
    columns_f7t = ['CONTACT LENGTH F7 TOP*', 'F7 TOP DIAMETER', 'F7 TOP HARDNESS',
                   'f7t_ESW VANIS', 'f7t_Kubota ECC-CX2 Type', 'f7t_National ICON',
                   'f7t_Union Electric UK Apex Alloy', 'f7t_Villares Vindex VRP0313']
    columns_f7b = ['CONTACT LENGTH F7 BOT*', 'F7 BOT DIAMETER', 'F7 BOT HARDNESS',
                   'f7b_Akers National Micra X', 'f7b_ESW VANIS', 'f7b_Kubota ECC-CX2 Type',
                   'f7b_National ICON', 'f7b_Union Electric UK Apex Alloy', 'f7b_Villares Vindex VRP0313']

    columns_family = ['family_1', 'family_2', 'family_3', 'family_4', 'family_5', 'family_6', 'family_7', 'family_8',
                      'family_9']
    columns_lineup = ['lineup_1', 'lineup_4', 'lineup_5', 'lineup_6', 'lineup_9']

    nb_additional_columns = 1  # In classic cases, only F/w*L is added

    def __init__(self, savefile: str, f6: bool, top: bool, all_rolls: bool = False):
        """ We determine the list of columns to keep from the F6 and top bool """

        if all_rolls:
            self.nb_additional_columns = 2
        self._set_initial_columns_name(f6, top, all_rolls)
        super(StripsInputDB, self).__init__(savefile, f6, top, all_rolls)

    def _set_initial_columns_name(self, f6: bool, top: bool, all_rolls: bool = False):
        if f6:  # f6
            self.columns_name = self.columns_f6_fwl + self.columns_f6
            if top:  # f6t
                self.columns_name += self.columns_f6t
            else:  # f6b
                self.columns_name += self.columns_f6b
        else:  # f7
            self.columns_name = self.columns_f7_fwl + self.columns_f7
            if top:  # f7t
                self.columns_name += self.columns_f7t
            else:  # f7b
                self.columns_name += self.columns_f7b

        if all_rolls:
            self.columns_name = self.columns_f6_fwl + self.columns_f7_fwl + \
                                self.columns_f6 + self.columns_f6t + self.columns_f6b \
                                + self.columns_f7 + self.columns_f7t + self.columns_f7b

        # We add families and line-ups
        self.columns_name += self.columns_family + self.columns_lineup

    def load(self):
        super(StripsInputDB, self).load()
        # We check if the length of the data and the list of columns name are the same.
        # If no, this means something has changed in parameters and the data should be loaded again
        if self.var.shape[-1] != len(self.columns_name) + self.nb_additional_columns:
            self.extract_from_raw_data()

        # Adding final F/w L columns
        if self.all_rolls:
            self.columns_name += ['F/w L F6']
            self.columns_name += ['F/w L F7']
        else:
            self.columns_name += ['F/w L']

    def extract_from_raw_data(self):
        # After doing the inheritor loading, we add the FwL parameter
        self.add_fwl()
        super(StripsInputDB, self).extract_from_raw_data()

    def add_fwl(self):
        # Finally, we add the F/w * L feature to each of them and we save them. F/w is always #0 and L #1
        if self.all_rolls:
            self.var[:, :, -2] = self.var[:, :, 0] * self.var[:, :, 1]
            self.var[:, :, -1] = self.var[:, :, 2] * self.var[:, :, 3]
        else:
            self.var[:, :, -1] = self.var[:, :, 0] * self.var[:, :, 1]


class StripsOutputDB(OutputDB, ABC):
    """ Abstract Output DataBase for Strips data """
    null_camp = [25, 56, 86, 75, 103, 133, 131, 148, 257, 256, 251, 271, 338, 365, 363, 380]


# todo: could be removed ?
class StripsDataSet(DataSet):

    def __init__(self, input_db: StripsInputDB, output_db: OutputDB, validation_split: float, random_seed: int):
        self.input: StripsInputDB = input_db
        super(StripsDataSet, self).__init__(input_db, output_db, validation_split, random_seed)
