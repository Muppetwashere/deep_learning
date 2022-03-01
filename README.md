# Datamining Roll Wear Project

This project consists in predicting the wear of rolls used in hot strips mills. 
This file describes the structure of files, as for resources and python sources, 
and how to use the main functions.

## File organisation

All the Python sources are located in the folder **Python_src**.
All the data are in the folder **Data**, sub-divided as:
- **RawData**: Folder containing all the raw data files. Those are the excel files of resutls. 
They are organised as follows:
    - _WearDataForDatamining.xlsx_: list of parameters for all the Strips
    - _WearCentres.xlsx_: list of the wear at the centre values for all the campaigns
    - **Wear_profiles**: folder of all the Excel files for complete profiles
- **DataBases**: Folder containing all the storing files for the DataBases created by the code.
Those have the advantage of being loaded much faster than Excel files
- **Outputs**: All the results of the Models, which contains historic of results and model saves.

## Python code and inheritance

### Object-oriented organisation

The code has been created with an object oriented approach. The structure is the following:

### Data

Elementary data are stored in `DataBase` objects.
In a `Database`, data are stored in a matrix, each row corresponding to a campaign.
This class has inheritors which are `InputDB` and `OutputDB`, used to store Inputs and Outputs of the models.
- For instance, the `WearCentreOutputDB` stores the wears at the centre for all the campaigns.
 
Then, `DataBase` are stored in `DataSet`: a `DataSet` contains two `DataBase`, one for Inputs (an `InputDB`) 
and one for Outputs (an `OutputDB`). 
- For instance, the `StripsWearCenter` is a DataSet used to predict _wear at the centre_ based on _strips_ data.

### Models

The same object-oriented approach is used for models. Most of models in this project are based on NeuralNetworks.
All models inherit from the `RollWearModel` class. They are all based on `Keras model`, and share their methods.
- For instance, the Neural Network to predict the wear at the centre from strips (using the `StripsWearCenter` DataSet)
is `StripsNN`. The one doing that with a recurrent architecture is `RecurrentStripsNN`.

## How-to: examples of usage
### Training a model

To train and use a model, the steps are:
1. Loading the DataSet
2. Creating the Network
3. Training the Model on the DataSet (train campaigns)
4. Validating the Model on the DataSet (validate campaigns)
5. Saving the model

#### Code sample:

**Loading DataSet**

    import py_rollwear.data.datasets as datasets
    dataset = datasets.StripsWearCenter(f6=True, top=True)
    x_train, x_dev, x_test, y_train, y_dev, y_test = dataset.get_train_var()

**Creating the Network**
    
    from py_rollwear.models.strips import StripsNN, RecurrentStripsNN, Mask
    neural_net = StripsNN(dataset, [other parameters])
    # or
    neural_net = RecurrentStripsNN(dataset, [other parameters])    

**Training the Model on the DataSet (train campaigns)**
    
    neural_net.fit(x_train, y_train, epochs=epochs, validation_data=(x_dev, y_dev), verbose=verbose, batch_size=16)

**Validating the Model on the DataSet (validate campaigns)**
    
    neural_net.evaluate(x_test, y_test)

Or, in order to plot it:

    import py_rollwear.plot.training as plot_training
    plot_training.wearcentre_predictions(neural_net, dataset)

**Saving the model**
    
    neural_net.save(savefile)

**Quick function**

All those steps can be done easily with the function `py_rollwear.wearcentre_from_strips.train_neuralnet(epochs, layers_sizes, layers_activations, savefile_name,
mask, recurrent, f6, top)` which trains a network and save it. For instance:

    import py_rollwear.wearcentre_from_strips
    py_rollwear.wearcentre_from_strips.train_neuralnet(250, (20, 8), ('selu', 'selu', 'sigmoid'), 
                                                       'test',mask=mask, recurrent=False, f6=True, top=True)



### Loading a model

To train and use a model, the steps are:
1. Loading the DataSet
2. Loading the Model from save
3. Testing the Model on the DataSet (validating campaigns)

##### Code sample

**Loading the DataSet**

    import py_rollwear.data.datasets as datasets
    dataset = dataSets.StripsWearCenter(f6=True, top=True)

**Loading the Model from save**

    from py_rollwear.models.strips import StripsNN, RecurrentStripsNN, Mask
    neural_net = StripsNN.load(model_save, dataset)
    # or
    neural_net = RecurrentStripsNN.load(model_save, dataset)    

**Testing the Model on the DataSet (using specific function)**

    import py_rollwear.plot.training as plot_training
    plot_training.wearcentre_predictions(neural_net, dataset)
    
**Quick function**

All those steps can be done easily with the function `py_rollwear.wearcentre_from_strips.load_neuralnet(savefile_name, f6, top)` which trains a network and save it. For instance:

    import py_rollwear.wearcentre_from_strips
    py_rollwear.wearcentre_from_strips.load_neuralnet('Recurrent', f6=True, top=True)
