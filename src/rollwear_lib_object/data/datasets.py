import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import inputs
from . import outputs
from .abstact_ds import DataSet, StripsDataSet


class MeanWearCenter(DataSet):
    modes = inputs.MeanCampaignsInputDB

    def __init__(self, f6: bool, top: bool, mode, validation_split: float = 0.33, random_seed: int = 69):
        input_ds = inputs.MeanCampaignsInputDB(f6, top, mode)
        output_ds = outputs.WearCentreOutputDB(f6, top)

        self.input: inputs.MeanCampaignsInputDB = input_ds
        super(MeanWearCenter, self).__init__(input_ds, output_ds, validation_split, random_seed)

    def get_x_dataframe(self):
        return self.input.get_dataframe(self.common_campaigns)

    def compute_auc(self):
        """ Computes Area Under the Curve for estimating impact of parameters on the final results """
        auc_dict = {}
        y = self.denormalize(self.get_y())
        x = self.get_x_dataframe()

        def print_auc(outp, title):
            bins = np.linspace(np.percentile(outp, 5), np.percentile(outp, 95), 100)
            for key in x.keys():
                hist_sup = np.histogram(outp[x[key] >= x[key].median()], bins, density=True)
                hist_inf = np.histogram(outp[x[key] <= x[key].median()], bins, density=True)

                cum_hist_sup = np.cumsum(hist_sup[0]) / np.sum(hist_sup[0])
                cum_hist_inf = np.cumsum(hist_inf[0]) / np.sum(hist_inf[0])

                assert np.array_equal(hist_sup[1], hist_inf[1]), 'Histograms have not been computed on the same bins'

                auc_dict[key] = np.sum(cum_hist_sup - cum_hist_inf)

            plt.figure()
            plt.subplot(211)
            plt.bar(range(len(auc_dict)), np.abs(list(auc_dict.values())), align='center')
            plt.ylabel('Absolute AUC (UA)')
            plt.title(title)
            plt.xticks(range(len(auc_dict)), list(auc_dict.keys()), rotation=90)

        print_auc(y, 'Beauzamy method applied to total wear')
        print_auc(y / x['Counts'], 'Beauzamy method applied to mean individual wear')


class StripsWearCenter(StripsDataSet):

    def __init__(self, f6: bool, top: bool, validation_split: float = 0.33, random_seed: int = 69,
                 all_rolls: bool = False):
        input_ds = inputs.StripsCentreInputDS(f6, top, all_rolls)
        output_ds = outputs.WearCentreOutputDB(f6, top, all_rolls)

        super(StripsWearCenter, self).__init__(input_ds, output_ds, validation_split, random_seed)


class StripsProfile(StripsDataSet):

    def __init__(self, f6: bool, top: bool, validation_split: float = 0.33, random_seed: int = 69):
        output_ds: outputs.FullProfileOutputDB = outputs.FullProfileOutputDB(f6, top)
        input_ds: inputs.StripsFullProfileInputDS = inputs.StripsFullProfileInputDS(f6, top, output_ds)

        super(StripsProfile, self).__init__(input_ds, output_ds, validation_split, random_seed)


class ThreePoints(StripsDataSet):

    def __init__(self, f6: bool, top: bool, validation_split: float = 0.33, random_seed: int = 69):
        input_ds = inputs.StripsCentreInputDS(f6, top)
        output_ds = outputs.ThreePointsOutputDB(f6, top)

        super(ThreePoints, self).__init__(input_ds, output_ds, validation_split, random_seed)


class StripsWearDiff(StripsDataSet):

    def __init__(self, f6: bool, top: bool, validation_split: float = 0.33, random_seed: int = 69,
                 all_rolls: bool = False):
        input_ds = inputs.StripsCentreInputDS(f6, top, all_rolls)
        output_ds = outputs.WearDiffOutputDB(f6, top, all_rolls)

        super(StripsWearDiff, self).__init__(input_ds, output_ds, validation_split, random_seed)

    def split(self, validation_split: float, random_seed: int):
        """ Splitting the campaigns into train, dev and test """

        self.output: outputs.WearDiffOutputDB

        df_train = pd.read_feather(self.output.savefile_train)
        df_val = pd.read_feather(self.output.savefile_val)

        self.idx_campaign_train = list(df_train["N° CAMPAIGN"])
        self.idx_campaign_dev = list(df_val["N° CAMPAIGN"])

        self.idx_campaign_test = self.idx_campaign_dev


