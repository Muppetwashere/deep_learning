from tensorflow import metrics

from ..data.abstact_ds import OutputDB


def mae_denorm(output_ds: OutputDB, y_true, y_pred, micrometers: bool = False):
    """ Returns the Mean Absolute Error (and std) of the given input, after denormalizing them.
     The values are given in µm (for Roll Wears) """
    dn = output_ds.denormalize
    if micrometers:
        return metrics.mae(1000 * dn(y_true), 1000 * dn(y_pred))
    else:
        return metrics.mae(dn(y_true), dn(y_pred))
