import numpy as np
import pandas as pd
from tqdm import tqdm

from .abstact_ds import InputDB, StripsInputDB
from .outputs import FullProfileOutputDB


# Definition of inputs modes
class InputModes:
    FWL = 'fwl'
    EXPERT_FEATURES = 'expt_feat'
    SELECTED = 'selected'
    SELECTED_no_EK = 'selected_no_ek'
    COMPLETE = 'complete'
    FULL = 'full'
    SELECTED_ONEHOT_F6T = 'selected_onehot'


class MeanCampaignsInputDB(InputDB):
    FWL = 'fwl'
    EXPERT_FEATURES = 'expt_feat'
    SELECTED = 'selected'
    SELECTED_no_EK = 'selected_no_ek'
    COMPLETE = 'complete'
    FULL = 'full'
    SELECTED_ONEHOT_F6T = 'selected_onehot'

    dict_column_names = {
        FWL: ['F/w_*L'],
        EXPERT_FEATURES: ['F/w_*L', 'L_HB'],
        SELECTED: ['THICKNESS F6 EXIT', 'ROLLING TIME', 'CONTACT LENGTH F6 TOP*', 'SUM ROLLING LENGTH F6',
                   'F6 TOP HARDNESS', 'F/w_*L', 'L_HB'],
        SELECTED_no_EK: ['STRIP LENGTH F6 EXIT*', 'STAND FORCE / WIDTH F6*', 'THICKNESS F6 EXIT', 'ROLLING TIME',
                         'CONTACT LENGTH F6 TOP*', 'SUM ROLLING LENGTH F6', 'F6 TOP HARDNESS'],
        COMPLETE: ['STRIP HARDNESS INDICATOR', 'STRIP WIDTH', 'STRIP LENGTH F6 EXIT*', 'STAND FORCE / WIDTH F6*',
                   'BENDING FORCE F6', 'SHIFTINGF6', 'TOP WR F6 DIAMETER [mm]', 'TEMPERATURE F5 EXIT',
                   'TEMPERATURE F6 EXIT', 'THICKNESS F6 EXIT', 'LEAD  SPEED F6', 'TRAIL SPEED F6', 'ROLLING TIME',
                   'REDUCTION F6*', 'CONTACT LENGTH F6 TOP*', 'F/w_*L', 'L_HB'],
        FULL: ['STRIP HARDNESS INDICATOR', 'STRIP WIDTH', 'STRIP LENGTH F6 EXIT*', 'STAND FORCE / WIDTH F6*',
               'BENDING FORCE F6', 'TOP WR F6 DIAMETER [mm]', 'TEMPERATURE F6 EXIT', 'THICKNESS F6 EXIT',
               'LEAD  SPEED F6', 'TRAIL SPEED F6', 'REDUCTION F6*', 'CONTACT LENGTH F6 TOP*', 'F/w_*L', 'L_HB']}

    dict_column_names.update({
        SELECTED_ONEHOT_F6T: dict_column_names[SELECTED]
                             + ['f6t_Akers National Micra X', 'f6t_ESW IRON (VIS)', 'f6t_ESW VANIS',
                                'f6t_Kubota ECC-CX2 Type', 'f6t_National ICON', 'f6t_OZPV (LPHNMD-80)',
                                'f6t_Union Electric UK Apex Alloy', 'f6t_Villares Vindex VRP0313']})

    def __init__(self, f6: bool, top: bool, mode):
        self.mode = mode
        self.column_names = self.dict_column_names[mode]
        super(MeanCampaignsInputDB, self).__init__('MeanCampaigns.npz', f6, top)

    def load(self):
        self.rollname = self.rollname + '_' + self.mode
        super(MeanCampaignsInputDB, self).load()

    def extract_from_raw_data(self):
        # Loading strips and adding new composed parameters
        df: pd.DataFrame = self.load_strips()
        df['F/w_*L'] = df['STAND FORCE / WIDTH F6*'] * df['STRIP LENGTH F6 EXIT*'] * 1000
        df['L_HB'] = df['STRIP LENGTH F6 EXIT*'] / (
                np.pi * df['TOP WR F6 DIAMETER [mm]'] * df['STRIP HARDNESS INDICATOR'])

        # Getting campaigns IDs and intializing self.var
        self.campaign_ids, counts = np.unique(df['N° CAMPAIGN'], return_counts=True)
        self.var = np.zeros((self.campaign_ids.shape[0], len(self.column_names) + 1))

        # For each campaign, we add the mean of strips parameters to self.var
        for i, camp_id in enumerate(self.campaign_ids):
            self.var[i][: -1] = np.mean(df[self.column_names][df['N° CAMPAIGN'] == camp_id])

        self.var[:, -1] = counts

        super(MeanCampaignsInputDB, self).extract_from_raw_data()

    def get_dataframe(self, idx_campaigns: np.array):
        return pd.DataFrame(data=self.denormalize(self.get_campaigns(idx_campaigns)),
                            index=idx_campaigns, columns=self.column_names + ['Counts'])


class StripsCentreInputDS(StripsInputDB):

    def __init__(self, f6: bool, top: bool, all_rolls: bool = False):
        super(StripsCentreInputDS, self).__init__('StripsCentre.npz', f6, top, all_rolls)

    def extract_from_raw_data(self):
        """ Loads the Excel files of strips into an input matrix """

        # Matrix of all the strips
        strips_df = self.load_strips()

        # Number of parameters and initialisation of matrices
        m = len(self.columns_name) + self.nb_additional_columns
        self.campaign_ids = np.unique(strips_df['N° CAMPAIGN'])
        self.var = np.zeros((self.campaign_ids.size, 306, m))

        # For all campaign, we go through all the strips to add them to the input
        for i, camp_id in tqdm(enumerate(self.campaign_ids), desc='Iterating through campaigns',
                               total=self.campaign_ids.size, smoothing=0):
            # Loading the list of the campaign strips
            strips_camp = strips_df[self.columns_name][strips_df['N° CAMPAIGN'] == camp_id]

            self.var[i, 0:strips_camp.shape[0], 0:-1 * self.nb_additional_columns] = strips_camp

        super(StripsCentreInputDS, self).extract_from_raw_data()


class StripsFullProfileInputDS(StripsInputDB):
    nb_additional_columns = 3

    def __init__(self, f6: bool, top: bool, full_profile_output_ds: FullProfileOutputDB):
        # We need the corresponding Profiles Outputs as the dataset must be created knowing the horizontal coordinates
        self.fpo: FullProfileOutputDB = full_profile_output_ds
        super(StripsFullProfileInputDS, self).__init__('StripsFullProfile.npz', f6, top)

    def extract_from_raw_data(self):
        """ Loads the Excel files of strips into an input matrix. """

        self._set_initial_columns_name(self.f6, self.top)  # Reinitialisation of columns name
        # Number of paramters: the selected columns + horiz. coordinate + cumulative length + FwL
        m = len(self.columns_name) + 3
        strips_df = self.load_strips()
        self.campaign_ids = self.fpo.campaign_ids
        coordinates = self.fpo.get_coordinates()

        self.var = np.zeros((coordinates.shape[0], 306, m))
        cpgn_id, camp_full_df = -1, None

        # For all samples, we'll go through all the campaign strips to verify if they are seen by this sample
        for idx_sample, iter_cp in tqdm(enumerate(self.campaign_ids.iteritems()), desc='Iterating through samples',
                                        total=self.campaign_ids.shape[0], smoothing=0):

            # There are two parallel index:
            #   The 'numpy' one, going from 0 to 17k with no hole, named idx_sample
            #   The 'pandas' one, going from 0 to 19k, with holes, named idx_df
            idx_df, cp_id = iter_cp

            # We reload the list of the campaign strips for each new campaign. This test enhance greatly performances
            if cpgn_id != cp_id:
                cpgn_id = cp_id
                camp_full_df = strips_df.loc[strips_df['N° CAMPAIGN'] == cp_id]

            # Initialization of the current ID of the strip
            idx_strip_in_sample, cum_length = 0, 0
            coordinate = coordinates[idx_df]

            # We get arrays for performances
            camp_params = np.array(camp_full_df[self.columns_name])
            camp_widths = np.array(camp_full_df['STRIP WIDTH'])
            camp_shifts = np.array(camp_full_df['SHIFTINGF6' if self.f6 else 'SHIFTING F7'])

            # We go through the strips of the campaign
            for idx_strip in range(camp_full_df.shape[0]):

                # Strip width and position parameters
                strip_param = camp_params[idx_strip]
                strip_width = camp_widths[idx_strip]
                strip_shift = camp_shifts[idx_strip]

                # We check if the strip is in the sample
                if self._is_strip_in_sample(strip_width, strip_shift, coordinate, top=self.top):
                    # Adding: strip parameters
                    self.var[idx_sample, idx_strip_in_sample][0: -3] = strip_param
                    # Adding: position of the sample
                    self.var[idx_sample, idx_strip_in_sample][-3] = coordinate
                    # Adding: Cumulative length seen by the cylinder over the campaign
                    self.var[idx_sample, idx_strip_in_sample][-2] = cum_length

                    # We increment the cumulative length
                    idx_strip_in_sample += 1
                    cum_length += strip_param[1]

        # Finaly, we add the F/w * L feature and save them
        super(StripsFullProfileInputDS, self).extract_from_raw_data()

    def load(self):
        super(StripsFullProfileInputDS, self).load()
        self.columns_name += ['X_Coord', 'Cum_Length']

    @staticmethod
    def _is_strip_in_sample(width: float, shift: float, sample_coord_x: float, top: bool):
        """ Returns True if the given strip is seen by the given sample """
        width_half = width / 2
        if top:
            return abs(sample_coord_x - shift) <= width_half
        else:
            return abs(sample_coord_x + shift) <= width_half
