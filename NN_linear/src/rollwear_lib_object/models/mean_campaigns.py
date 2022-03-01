import numpy as np
from sklearn.metrics import recall_score, mean_absolute_error
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

from .abstract_models import MyModel, RollWearModel
from ..data import datasets
from ..plot.training import wearcentre_predictions


class MeanCampNN(RollWearModel):
    """ Neural Network computing the wear at the centre from average values of the strips parameters over a campaign
    """

    def __init__(self, dataset: datasets.MeanWearCenter,
                 hidden_layer_sizes: tuple = (8, 4, 4, 4, 2),
                 hidden_layer_activ: tuple = None,
                 single_output: bool = True):
        x = dataset.get_x()
        shape = (x.shape[1],)
        inputs = Input(shape=shape)
        # When we reuse the same layer instance multiple times, the weights of the layer are also being reused
        # (it is effectively *the same* layer)

        # We assert the activations are the same size than the dense layers
        assert \
            hidden_layer_activ is None \
            or (len(hidden_layer_activ) == len(hidden_layer_sizes)), \
            'The activation list must be the same size than the layer list'

        hidden_layer = inputs
        activ = 'relu'
        # We add the dense layers, followed by activations
        for i, n in enumerate(hidden_layer_sizes):
            if hidden_layer_activ is not None:
                activ = hidden_layer_activ[i]
                if activ == 'r':
                    activ = 'relu'

            hidden_layer = Dense(n, activation=activ)(hidden_layer)

        # Adding last layer
        n_output = 1 + 3 * (not single_output)
        final_layer = Dense(n_output, activation='sigmoid')

        predictions = final_layer(hidden_layer)

        # We define a trainable model linking the inputs to the predictions
        super(MeanCampNN, self).__init__(inputs=inputs, outputs=predictions, dataset=dataset)

    def summary(self, line_length=None, positions=None, print_fn=None):
        super(MyModel, self).summary(line_length=None, positions=None, print_fn=None)


class WearCDoubleDT:
    def __init__(self, threshold):
        self.dt_inf = DecisionTreeRegressor(max_depth=6, max_features=None, max_leaf_nodes=50, min_samples_leaf=8,
                                            min_samples_split=8, presort=True)
        self.dt_sup = DecisionTreeRegressor(max_depth=7, max_features='sqrt', max_leaf_nodes=6, min_samples_leaf=2,
                                            min_samples_split=2, presort=True)
        self.classif = SVC(kernel='poly', gamma='scale', degree=3, class_weight='balanced')

        self.thres = threshold

    def fit(self, campaign_ds: datasets.MeanWearCenter):
        x_train, _, _, y_train, _, _ = campaign_ds.get_train_var()
        y_train_denorm = campaign_ds.denormalize(y_train, x_train)

        # Fitting classifier
        labels_train = y_train_denorm > self.thres
        self.classif.fit(x_train, labels_train)

        # Fitting Decision Trees
        x_train_inf = x_train[y_train_denorm < self.thres]
        y_train_inf = y_train[y_train_denorm < self.thres]

        x_train_sup = x_train[y_train_denorm > self.thres]
        y_train_sup = y_train[y_train_denorm > self.thres]

        self.dt_inf.fit(x_train_inf, y_train_inf)
        self.dt_sup.fit(x_train_sup, y_train_sup)

    def predict(self, x):
        labels = self.classif.predict(x)
        index_inf = (labels == 0)
        index_sup = (labels == 1)

        y_pred = np.zeros((x.shape[0]))
        y_pred[index_inf] = self.dt_inf.predict(x[index_inf]) if index_inf.sum() else []
        y_pred[index_sup] = self.dt_sup.predict(x[index_sup]) if index_sup.sum() else []

        return y_pred

    def evaluate(self, dataset: datasets.MeanWearCenter):
        x_train, x_dev, x_test, y_train, y_dev, y_test = dataset.get_train_var()

        def split_inf_sup(x, y):
            y_den = dataset.denormalize(y, x)

            id_inf = (y_den < self.thres)
            labels = id_sup = (y_den > self.thres)

            return x[id_inf], x[id_sup], y[id_inf], y[id_sup], labels

        x_train_inf, x_train_sup, y_train_inf, y_train_sup, labels_train = split_inf_sup(x_train, y_train)
        x_dev_inf, x_dev_sup, y_dev_inf, y_dev_sup, labels_dev = split_inf_sup(x_dev, y_dev)
        x_test_inf, x_test_sup, y_test_inf, y_test_sup, labels_test = split_inf_sup(x_test, y_test)

        # Print classifier scores
        print('Classifier recall scores:\n\tTrain = %.2f | Dev = %.2f | Test = %.2f' %
              (recall_score(labels_train, self.classif.predict(x_train)),
               recall_score(labels_dev, self.classif.predict(x_dev)),
               recall_score(labels_test, self.classif.predict(x_test))))
        print('Classifier precision scores:\n\tTrain = %.2f | Dev = %.2f | Test = %.2f' %
              (self.classif.score(x_train, labels_train), self.classif.score(x_dev, labels_dev),
               self.classif.score(x_test, labels_test)))

        def print_decision_tree_score(decision_tree, _x_train, _x_dev, _x_test, _y_train, _y_dev, _y_test, title: str):
            def get_mae(x, y):
                return mean_absolute_error(1000 * y, 1000 * decision_tree.predict(x))

            # Print Inf Tree scores
            mae_train = get_mae(_x_train, _y_train)
            mae_dev = get_mae(_x_dev, _y_dev)
            mae_test = get_mae(_x_test, _y_test)

            print("Decision tree %s:\n"
                  "\tMAE train = %.1f µm\n\tMAE dev  = %.1f µm\n\tMAE test = %.1f µm"
                  % (title, mae_train, mae_dev, mae_test))

        print_decision_tree_score(self.dt_inf, x_train_inf, x_dev_inf, x_test_inf,
                                  y_train_inf, y_dev_inf, y_test_inf, title='inferior')
        print_decision_tree_score(self.dt_sup, x_train_sup, x_dev_sup, x_test_sup,
                                  y_train_sup, y_dev_sup, y_test_sup, title='superior')

        # Plotting results
        print("Decision tree total:")
        # noinspection PyTypeChecker
        wearcentre_predictions(self, dataset)
