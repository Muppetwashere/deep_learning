"""
Created on Tue Apr 23 2019

@author: Antonin GAY (U051199)
"""
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from tqdm import tqdm

from rollwear_lib_object.data import datasets


# 1. DataSet

def adapt_xy_for_pytorch(x, y: pd.Series):
    new_x = []
    for row in x:
        # In this DataSet, adapted for TF, all the campaigns are the same length of 306 (i.e. same nb of strips).
        # In our case, we want each campaigns to have the correct length. To do that, we will re-create the database.
        # For each row, we get only the rows (=strips) where at least one of the column is not-zero
        no_null_row = row[np.where((row != 0).any(axis=1))]

        new_x.append(no_null_row)

    new_y = y.to_numpy()

    return new_x, new_y


def lstm_data_loading() -> (List[np.ndarray], List[np.ndarray], List[np.ndarray],
                            np.ndarray, np.ndarray, np.ndarray):
    # We use the TF-based objects to load the Strips
    dataset = datasets.StripsWearCenter(f6=True, top=True, all_rolls=False)
    x_train, x_dev, x_test, y_train, y_dev, y_test = dataset.get_train_var()

    # We adapt them to our new shape [n_campaigns, n_strips, n_param]
    x_train, y_train = adapt_xy_for_pytorch(x_train, y_train)
    x_dev, y_dev = adapt_xy_for_pytorch(x_dev, y_dev)
    x_test, y_test = adapt_xy_for_pytorch(x_test, y_test)

    return x_train, x_dev, x_test, y_train, y_dev, y_test


if __name__ == '__main__':
    print('Hello World!')

    # I'd like to test LSTM on this dataset, with PyTorch
    # The steps could be :
    # 1. Loading DataSet in a correct shape with numpy
    # 2. Creating neural net
    # 3. Data normalisation
    # 4. Training neural net with for loop
    # 3b. De-normalisation of data

    # 1. DS loading
    # Shape of the Dataset : each line should be a full campaign of shape [n_brames, n_parameters]

    x_train, x_dev, x_test, y_train, y_dev, y_test = lstm_data_loading()

    # Now that the data are loaded, we create the Neural Net
    # 2. Neural Net creation
    # todo: try various hidden_sizes and num_layers
    n_hidden = 8
    num_layers = 3

    lstm = nn.LSTM(input_size=x_train[0].shape[1], hidden_size=n_hidden, num_layers=num_layers, dropout=0.3)
    linear = nn.Sequential(
        nn.Flatten(start_dim=0),
        nn.Linear(n_hidden * num_layers, 1),
    )

    loss_fn = torch.nn.MSELoss()
    mae_fn = torch.nn.L1Loss()

    # Defining optimizer
    learning_rate = 1e-3
    optimizer = torch.optim.RMSprop(
        list(lstm.parameters()) + list(linear.parameters()),
        lr=learning_rate)

    # Now that the Neural Net is initialised, we will normalize data with standard Scaler (sklearn)
    # 3. Data Normalisation
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_train_strips = np.concatenate(x_train)
    x_scaler.fit(x_train_strips)
    y_scaler.fit(y_train.reshape(-1, 1))

    x_train = list(map(lambda a: x_scaler.transform(a), x_train))
    x_dev = list(map(lambda a: x_scaler.transform(a), x_dev))
    x_test = list(map(lambda a: x_scaler.transform(a), x_test))

    y_train = y_scaler.transform(y_train.reshape(-1, 1))
    y_dev = y_scaler.transform(y_dev.reshape(-1, 1))
    y_test = y_scaler.transform(y_test.reshape(-1, 1))

    # 4. Neural Network training
    # 4a. Transforming numpy to torch

    x_train = [torch.from_numpy(x).float() for x in x_train]
    x_dev = [torch.from_numpy(x).float() for x in x_dev]
    x_test = [torch.from_numpy(x).float() for x in x_test]

    y_train = torch.from_numpy(y_train).float()
    y_dev = torch.from_numpy(y_dev).float()
    y_test = torch.from_numpy(y_test).float()

    epochs = 100
    batch_size = 16
    for i in range(epochs):
        pred_train_list = torch.zeros(size=(len(x_train), 1))
        pred_train_batch = torch.zeros(size=(batch_size, 1))
        for j, x in enumerate(x_train):
            # Forward
            out_lstm, (hn, cn) = lstm(x.reshape((-1, 1, x.shape[1])))
            pred_train = linear(hn)
            pred_train_list[j] = pred_train
            pred_train_batch[j % batch_size] = pred_train

            if j % batch_size == batch_size - 1:
                idx_min = batch_size * (j // batch_size)

                # Loss
                loss = loss_fn(pred_train_batch, y_train[idx_min: j + 1])

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred_train_batch = torch.zeros(size=(batch_size, 1))

        # Printing every 100 epochs
        if i % 1 == 0:
            pred_dev = torch.zeros(size=(len(x_dev), 1))
            for j, x in enumerate(x_dev):
                out_lstm, (hn, cn) = lstm(x.reshape((-1, 1, x.shape[1])))
                pred_dev[j] = linear(hn)

            pred_dev_denormalised = y_scaler.inverse_transform(pred_dev.detach())
            y_dev_denormalised = y_scaler.inverse_transform(y_dev)

            pred_dev_denormalised = torch.from_numpy(pred_dev_denormalised).float()
            y_dev_denormalised = torch.from_numpy(y_dev_denormalised).float()

            loss_train = loss_fn(pred_train_list, y_train)
            val_loss = loss_fn(pred_dev, y_dev)
            mae_loss = mae_fn(pred_dev_denormalised, y_dev_denormalised)

            print(f"{i} - mse train {loss_train.item(): .4f} - mse val {val_loss.item(): .4f} - "
                  f"mae val {1000 * mae_loss: .1f}µm", )

            # todo: add EarlyStopping


def old():
    pass
    # # Creating dataset
    # dataset = datasets.StripsWearDiff(f6=True, top=True, random_seed=42, all_rolls=False)
    #
    # # Loading Neural Net
    # neural_net = RecurrentStripsNNDeltas.load("test_recurrent_deltas", dataset)
    #
    # # Plotting results
    # plot_training.wearcentre_predictions(neural_net, dataset)
    #
    # plt.ylim([-10.5, -10])
    #
    # # wearcentre_from_strips.load_neuralnet_deltas('test_recurrent_deltas', f6=True, top=True)
    # plt.show()

    """
    # Lib object samples
    # from rollwear_lib_object.samples import report_examples
    # report_examples(args='all_profiles')

    # Lib functionnal examples
    # Files location
    path_h5_raw_data = 'Data/raw_data.h5'
    path_h5_preprocessed_data = 'Data/preprocessed_data.h5'
    # Parameter
    batch_size = 8

    # from streamlit_rw.model_training import single_training
    # _, model_ful, model_nn = single_training(path_h5_preprocessed_data, path_h5_raw_data, loss_name='mse', epochs=250,
    #                                          batch_size=batch_size, model_name='Model main py', in_streamlit=False)

    from streamlit_rw.results_plotting import results_plotting

    results_plotting('Test_model', path_h5_preprocessed_data, path_h5_raw_data)
    plt.show()
    """
