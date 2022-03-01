from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from xlrd import XLRDError

from .abstact_ds import OutputDB, StripsOutputDB


class WearCentreOutputDB(OutputDB):
    null_camp = [25, 56, 86, 75, 93, 103, 131, 188, 257, 271, 365]
    rawfile_wearcentre = 'Data/RawData/WearCentres.xlsx'

    def __init__(self, f6: bool, top: bool, all_rolls: bool = False):
        super(WearCentreOutputDB, self).__init__('WearCentre.feather', f6, top, all_rolls)

    def extract_from_raw_data(self):
        """ Load the data of one excel file of wear centre data """

        print("Loading Output data from excel. Takes about 1mn")

        # We read the data from the Excel file
        dataframe: pd.DataFrame = pd.read_excel(io=self.rawfile_wearcentre,
                                                sheet_name='Feuil1', header=2, usecols="A:E", skiprows=[3])

        # renaming columns and saving the DataFrame
        dataframe.rename(inplace=True, columns={'Usure F6 TOP': 'f6t', 'Usure F6 BOT': 'f6b',
                                                'Usure F7 TOP': 'f7t', 'Usure F7 BOT': 'f7b',
                                                'Num Campagne': 'N° CAMPAIGN'})
        dataframe.to_feather(self.savefile)
        self.extract_from_dataframe(dataframe)

    def extract_from_dataframe(self, dataframe: pd.DataFrame):
        dataframe.set_index('N° CAMPAIGN', inplace=True, drop=False)
        super(WearCentreOutputDB, self).extract_from_dataframe(dataframe)

    def remove_outliers(self):
        super(WearCentreOutputDB, self).remove_outliers()

        # Removing negative rows
        # noinspection PyTypeChecker
        neg_rows = np.where(self.var < 0)[0]
        self.drop_rows(neg_rows)

        # Removing the rows identified as null or negatives
        null_rows = np.where(np.in1d(self.campaign_ids, self.null_camp))
        self.drop_rows(null_rows)


class WearDiffOutputDB(OutputDB):
    null_camp = [25, 56, 86, 75, 93, 103, 131, 188, 257, 271, 365]
    rawfile_wearcentre = 'Data/RawData/WearDiff.xlsx'

    savefile_train = r'Data\DataBases\OutputsDB\WearDiffTrain.feather'
    savefile_val = r'Data\DataBases\OutputsDB\WearDiffVal.feather'

    def __init__(self, f6: bool, top: bool, all_rolls: bool = False):
        super(WearDiffOutputDB, self).__init__('WearDiff.feather', f6, top, all_rolls)

    def extract_from_raw_data(self):
        """ Load the data of one excel file of wear centre data """

        print("Loading Output data from excel. Takes about 1mn")

        dataframe_train = self.extract_from_exel_wear_diff(r'Data/RawData/WearDiffTrain.xlsx')
        dataframe_val = self.extract_from_exel_wear_diff(r'Data/RawData/WearDiffVal.xlsx')
        dataframe = pd.concat([dataframe_train, dataframe_val]).reset_index(drop=True)

        dataframe_train.to_feather(self.savefile_train)
        dataframe_val.to_feather(self.savefile_val)
        dataframe.to_feather(self.savefile)

        self.extract_from_dataframe(dataframe)

    @staticmethod
    def extract_from_exel_wear_diff(raw_file):
        a: pd.DataFrame = pd.read_excel(io=raw_file, header=6, usecols="A:B", skiprows=[])
        a = a.dropna().astype({"Nb_camp": int})
        a = a.rename(columns={"∆Wear[mm]": "f6t", "Nb_camp": "N° CAMPAIGN"})
        a = a.set_index("N° CAMPAIGN")
        dataframe: pd.DataFrame = a.copy()

        a = pd.read_excel(io=raw_file, header=6, usecols="C:D", skiprows=[])
        a = a.dropna().astype({"Nb_camp.1": int})
        a = a.rename(columns={"∆Wear[mm].1": "f6b", "Nb_camp.1": "N° CAMPAIGN"})
        a = a.set_index("N° CAMPAIGN")
        dataframe = dataframe.join(a.copy())

        a = pd.read_excel(io=raw_file, header=6, usecols="E:F", skiprows=[])
        a = a.dropna().astype({"Nb_camp.2": int})
        a = a.rename(columns={"∆Wear[mm].2": "f7t", "Nb_camp.2": "N° CAMPAIGN"})
        a = a.set_index("N° CAMPAIGN")
        dataframe = dataframe.join(a.copy())

        a = pd.read_excel(io=raw_file, header=6, usecols="G:H", skiprows=[])
        a = a.dropna().astype({"Nb_camp.3": int})
        a = a.rename(columns={"∆Wear[mm].3": "f7b", "Nb_camp.3": "N° CAMPAIGN"})
        a = a.set_index("N° CAMPAIGN")
        dataframe = dataframe.join(a.copy())

        dataframe = dataframe.reset_index(drop=False)

        return dataframe

    def extract_from_dataframe(self, dataframe: pd.DataFrame):
        dataframe.set_index('N° CAMPAIGN', inplace=True, drop=False)
        super(WearDiffOutputDB, self).extract_from_dataframe(dataframe)

    def remove_outliers(self):
        super(WearDiffOutputDB, self).remove_outliers()

        # Removing the rows identified as null or negatives
        null_rows = np.where(np.in1d(self.campaign_ids, self.null_camp))
        self.drop_rows(null_rows)


class FullProfileOutputDB(StripsOutputDB):
    rawfiles_directory = 'Data/RawData/Wear_profiles/'  # This is a directory, should end by '/'

    def __init__(self, f6: bool, top: bool):
        super(FullProfileOutputDB, self).__init__('FullProfile.feather', f6, top)

    def extract_from_raw_data(self):
        """ Load the samples from the excel files """
        dataframe = pd.DataFrame(columns=['# Camp', 'X', 'f6b', 'f6t', 'f7b', 'f7t'])
        camp_idx, value_err, key_err, not_loaded = [], [], [], []
        directory = self.rawfiles_directory

        # Campaign ID goes from 1 to 395, keeping the ID of the original Excel file
        for cp_id in tqdm(range(1, 396), "Creating Y DS"):

            try:
                file = glob(directory + "Camp %d" % cp_id + "-*")[0]
                y_i = pd.read_excel(file, sheet_name=['f6', 'f7'], header=1, usecols="A, F, M, AG", index_col=0)
                y_i_f6 = y_i['f6']
                y_i_f7 = y_i['f7']

                camp_idx.append(cp_id)
                y_i_f6 = y_i_f6.rename(index=str,
                                       columns={'X.1': 'X', 'Top Wear, mm': 'f6t', 'Bottom Wear, mm': 'f6b'})
                y_i_f7 = y_i_f7.rename(index=str,
                                       columns={'X.1': 'X', 'Top Wear, mm': 'f7t', 'Bottom Wear, mm': 'f7b'})

                y_i = pd.concat([y_i_f6, y_i_f7.drop(columns='X')], axis=1, sort=False).dropna(axis=0)
                y_i['N° CAMPAIGN'] = cp_id
                y_i['X'] -= y_i['X'].mean()

                dataframe = dataframe.append(y_i, ignore_index=True, sort=False)

            # We append the error to the corresponding lists
            except ValueError:
                value_err.append(cp_id)
            except KeyError:
                key_err.append(cp_id)
            except XLRDError:
                not_loaded.append(cp_id)
            except IndexError:
                not_loaded.append(cp_id)

        print('Errors while loading profile files:'
              '\tValueError on Camp #\t%r\n'
              '\tKeyError on Camp #\t%r\n'
              '\tNot Loaded Camp #\t%r\n' % (value_err, key_err, not_loaded))

        dataframe.to_feather(self.savefile)
        self.extract_from_dataframe(dataframe)

    def get_coordinates(self, idx_campaigns: int = None):
        """ Returns the coordinates of each point. Mostly useful for debugging """
        try:
            dataframe: pd.DataFrame = pd.read_feather(self.savefile)
            if idx_campaigns is None:
                return dataframe['X'][self.var.index]  # We only return coordinates of the kept samples
            else:
                return dataframe['X'][self.var.index][
                    np.in1d(self.campaign_ids, idx_campaigns)]  # We only return coordinates of the kept samples
        except FileNotFoundError:
            print('The file has not been found. Try reloading the object')
            return None


class ThreePointsOutputDB(StripsOutputDB):
    null_camp = [25, 56, 86, 75, 103, 133, 131, 148, 257, 256, 251, 271, 338, 365, 363, 380]

    def __init__(self, f6: bool, top: bool, first_point_id: int = 20, last_point_id: int = 44):
        self.first_pt_id = first_point_id
        self.last_pt_id = last_point_id
        super(ThreePointsOutputDB, self).__init__('ThreePoints.feather', f6, top)

    def extract_from_raw_data(self):
        fpo = FullProfileOutputDB(self.f6, self.top)

        self.campaign_ids = fpo.campaign_ids.unique()
        self.var = pd.DataFrame(index=self.campaign_ids, columns=['left_sample', 'center_sample', 'right_sample'],
                                dtype=float)

        for cp_id in self.campaign_ids:
            profile = fpo.get_campaigns([cp_id])

            try:
                self.var.loc[cp_id] = profile.iloc[[self.first_pt_id, 32, self.last_pt_id]].to_list()
            except IndexError:
                # If the profile is not long enough, we remove the row
                error_row = np.where(self.campaign_ids == cp_id)[0]
                self.drop_rows(error_row)

        super(ThreePointsOutputDB, self).extract_from_raw_data()
