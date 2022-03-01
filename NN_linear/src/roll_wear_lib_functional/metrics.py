import tensorflow as tf
import numpy as np

# todo: find a way to not hard-code this value
max_strips_per_campaigns = 306


# Conversion from strip predictions to campaign predictions

def strip_prediction_to_campaign_tensor(y_pred: tf.Tensor):
    """ Convert a list of predictions for individual strips into predicted campaign wear

    :param y_pred: List of predicted wear per strip tensors. Shape = (n * 306, 4)

    :return campaign_wear_tensor: list of predicted wear per campaign. Shape = (n, 4)
    """
    y_pred = tf.reshape(y_pred, [-1, max_strips_per_campaigns, 4])
    return tf.math.reduce_sum(y_pred, axis=1)


def strip_prediction_to_campaign_np_array(y_pred: np.ndarray):
    """ Convert a list of predictions for individual strips into predicted campaign wear

    :param y_pred: List of predicted wear per strip tensors. Shape = (n * 306, 4)

    :return campaign_wear_tensor: list of predicted wear per campaign. Shape = (n, 4)
    """
    y_pred = np.reshape(y_pred, [-1, max_strips_per_campaigns, 4])
    return np.sum(y_pred, axis=1)


@tf.function
def mse_rw(y_true, y_pred):
    """ Compute the error while given the individual estimations for strips and the truth values for campaigns

    :param y_true: List of wear per campaign. Shape (n, 4)
    :param y_pred: List of predicted wear per strip tensors. Shape = (n * 306, 4)

    :return mse: Mean Squared Error on batch
    """
    campaigns_wear_pred = strip_prediction_to_campaign_tensor(y_pred)

    return tf.losses.mse(y_true, campaigns_wear_pred)


@tf.function
def mae_rw(y_true, y_pred):
    """ Compute the error while given the individual estimations for strips and the truth values for campaigns.
    This mae is not denormalized, so the error is not in micrometers !

    :param y_true: List of wear per campaign. Shape (n, 4)
    :param y_pred: List of predicted wear per strip tensors. Shape = (n * 306, 4)

    :return mae: Mean Absolute Error on batch
    """
    campaigns_wear_pred = strip_prediction_to_campaign_tensor(y_pred)

    return tf.losses.mae(y_true, campaigns_wear_pred)


# Functions returning the metrics with denormalization

def get_mae_denormalized(output_scaler):
    """ Returns the MAE function with denormaliation using the given scaler

    :param output_scaler: Must possess a output_scaler.inverse_transform method

    :return mae_micrometers: metric function
    """

    def mae_micrometers(y_true, y_pred):
        """ Compute the error while given the individual estimations for strips and the truth values for campaigns

        :param y_true: List of wear per campaign. Shape (n, 4)
        :param y_pred: List of predicted wear per strip tensors. Shape = (n * 306, 4)

        :return mae: Mean Absolute Error on batch in µm.
        """

        campaigns_wear_pred = strip_prediction_to_campaign_tensor(y_pred)

        campaigns_wear_pred = 1000 * output_scaler.inverse_transform(campaigns_wear_pred)
        y_true = 1000 * output_scaler.inverse_transform(y_true)

        return tf.metrics.mae(y_true, campaigns_wear_pred)

    return mae_micrometers


def get_mse_denormalized(output_scaler):
    """ Returns the MSE function with denormaliation using the given scaler

    :param output_scaler: Must possess a output_scaler.inverse_transform method

    :return mse_micrometers: metric function
    """

    def mse_micrometers(y_true, y_pred):
        """ Compute the error while given the individual estimations for strips and the truth values for campaigns

        :param y_true: List of wear per campaign. Shape (n, 4)
        :param y_pred: List of predicted wear per strip tensors. Shape = (n * 306, 4)

        :return mse: Mean Squared Error on batch in µm.
        """

        campaigns_wear_pred = strip_prediction_to_campaign_tensor(y_pred)

        campaigns_wear_pred = 1000 * output_scaler.inverse_transform(campaigns_wear_pred)
        y_true = 1000 * output_scaler.inverse_transform(y_true)

        return tf.metrics.mse(y_true, campaigns_wear_pred)

    return mse_micrometers
