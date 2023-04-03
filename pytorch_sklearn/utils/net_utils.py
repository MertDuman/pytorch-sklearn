import os
from os.path import join as osj
import pandas as pd
import torch
from pytorch_sklearn import NeuralNetwork
from pytorch_sklearn.callbacks import Callback

from typing import Mapping, Iterable
import ast


def try_load_network_from_csv(
    csv_folder: str, 
    csv_filename: str,
    idx: int,
    column_mapper: Mapping[str, str],
    net_path: str = None,
    load_best: str = False,
    weight_path: str = None,
    relative_path: bool = True,
    callbacks: Iterable[Callback] = None,
    module_list: Iterable[str] = None,
    device: str = None,
): 
    '''
    Tries to load the neural network from a csv file.

    Parameters
    ----------
    csv_folder : str
        Path to the folder containing the csv file.
    csv_filename : str
        Filename without path prefixes.
    idx : int
        Index of the row in the csv file.
    column_mapper : dict of str to str
        Maps the column names in the csv file to the necessary arguments to build a NeuralNetwork.

        These keys are necessary:
            - model_name: class name of the model
            - optim_name: class name of the optimizer
            - crit_name: class name of the criterion
        If not given, assumes the column names are 'model_name', 'optim_name', and 'crit_name'.

        These keys are likely needed:
            - id: A unique identifier for the network. This csv row has a unique id that links to a folder within csv_folder.
            - model_constructor: dictionary of arguments to pass to the model constructor. If not present, assumes no arguments.
            - optim_constructor: dictionary of arguments to pass to the optimizer constructor. If not present, assumes no arguments.
            - crit_constructor: dictionary of arguments to pass to the criterion constructor. If not present, assumes no arguments.
            - lr: learning rate to pass to the optimizer constructor. Some optimizers require a learning rate, like SGD.

    net_path : str
        Path to the network file. By default will look for `os.path.join(csv_folder, net.pth)`.
    load_best : bool
        If True, will try to load the best weights from weight_path.
    weight_path : str
        Path to the weight file. By default will look for `os.path.join(csv_folder, weights.pth)`.
    relative_path : bool
        Whether net and weight path are relative within the csv folder, or the full path. If True, will look for `os.path.join(csv_folder, net_path)`.
        Otherwise, will look for `net_path`.
    callbacks : list of Callbacks
        Will be passed to the NeuralNetwork.load_class method, and callback states will be instantiated.
    module_list : list of str
        List of modules to look for the model, optimizer, and criterion classes in.
        If not passed, will look for the classes in all imported modules. This may cause name clashes and take more time.
    device : str
        Device to load the network to. If not passed, will use 'cuda' if available, else 'cpu'.
    '''
    import sys
    callbacks = [] if callbacks is None else callbacks
    module_list = sys.modules.copy() if module_list is None else module_list

    csv_file = pd.read_csv(osj(csv_folder, csv_filename), index_col=0, keep_default_na=False)
    
    row = csv_file.iloc[idx]
    model_name = row[column_mapper.get('model_name', 'model_name')]
    optim_name = row[column_mapper.get('optim_name', 'optim_name')]
    crit_name = row[column_mapper.get('crit_name', 'crit_name')]

    model_class = None
    optim_class = None
    crit_class = None
    for k in module_list:
        model_class = getattr(sys.modules[k], model_name, model_class)
        optim_class = getattr(sys.modules[k], optim_name, optim_class)
        crit_class = getattr(sys.modules[k], crit_name, crit_class)

    assert model_class is not None, f'Could not find model class {model_name}'
    assert optim_class is not None, f'Could not find optimizer class {optim_name}'
    assert crit_class is not None, f'Could not find criterion class {crit_name}'

    try:
        constructor = ast.literal_eval(row[column_mapper['model_constructor']])  # :)
        model = model_class(**constructor)
    except:
        model = model_class()

    get_device = lambda: 'cuda' if torch.cuda.is_available() else 'cpu'
    device = device if device is not None else get_device()
    model.to(device)

    try:
        constructor = ast.literal_eval(row[column_mapper['optim_constructor']])  # :)
        optimizer = optim_class(model.parameters(), **constructor)
    except:
        try:
            lr = row[column_mapper['lr']]
            optimizer = optim_class(model.parameters(), lr=lr)
        except:
            optimizer = optim_class(model.parameters())

    try:
        constructor = ast.literal_eval(row[column_mapper['crit_constructor']])  # :)
        crit = crit_class(**constructor)
    except:
        crit = crit_class()

    net = NeuralNetwork(model, optimizer, crit)

    try:
        id = row[column_mapper.get('id', 'id')]
        unique_folder = osj(csv_folder, id)
    except:
        unique_folder = csv_folder

    net_path = osj(unique_folder, 'net.pth') if net_path is None else (osj(unique_folder, net_path) if relative_path else net_path)
    weight_path = osj(unique_folder, 'weights.pth') if weight_path is None else (osj(unique_folder, weight_path) if relative_path else weight_path)
    
    NeuralNetwork.load_class(net, callbacks, net_path)

    if load_best:
        try:
            net.load_weights_from_path(weight_path)
        except:
            print('Did not find best weights, using original.')

    return net


def try_load_model_from_csv(
    csv_folder: str, 
    csv_filename: str,
    idx: int,
    column_mapper: Mapping[str, str],
    load_best: str = False,
    weight_path: str = None,
    relative_path: bool = True,
    module_list: Iterable[str] = None,
    device: str = None,
): 
    '''
    Tries to load model from a csv file. 
    More lightweight than try_load_net_from_csv as it does not load any other network components.

    Parameters
    ----------
    csv_folder : str
        Path to the folder containing the csv file.
    csv_filename : str
        Filename without path prefixes.
    idx : int
        Index of the row in the csv file.
    column_mapper : dict of str to str
        Maps the column names in the csv file to the necessary arguments to build a NeuralNetwork.

        These keys are necessary:
            - model_name: class name of the model
            - optim_name: class name of the optimizer
            - crit_name: class name of the criterion
        If not given, assumes the column names are 'model_name', 'optim_name', and 'crit_name'.

        These keys are likely needed:
            - id: A unique identifier for the network. This csv row has a unique id that links to a folder within csv_folder.
            - model_constructor: dictionary of arguments to pass to the model constructor. If not present, assumes no arguments.
            - optim_constructor: dictionary of arguments to pass to the optimizer constructor. If not present, assumes no arguments.
            - crit_constructor: dictionary of arguments to pass to the criterion constructor. If not present, assumes no arguments.
            - lr: learning rate to pass to the optimizer constructor. Some optimizers require a learning rate, like SGD.
            
    load_best : bool
        If True, will try to load the best weights from weight_path.
    weight_path : str
        Path to the weight file. By default will look for `os.path.join(csv_folder, weights.pth)`.
    callbacks : list of Callbacks
        Will be passed to the NeuralNetwork.load_class method, and callback states will be instantiated.
    module_list : list of str
        List of modules to look for the model, optimizer, and criterion classes in.
        If not passed, will look for the classes in all imported modules. This may cause name clashes and take more time.
    device : str
        Device to load the network to. If not passed, will use 'cuda' if available, else 'cpu'.
    '''
    import sys
    callbacks = [] if callbacks is None else callbacks
    module_list = sys.modules.copy() if module_list is None else module_list

    csv_file = pd.read_csv(osj(csv_folder, csv_filename), index_col=0, keep_default_na=False)
    
    row = csv_file.iloc[idx]
    model_name = row[column_mapper.get('model_name', 'model_name')]
    optim_name = row[column_mapper.get('optim_name', 'optim_name')]
    crit_name = row[column_mapper.get('crit_name', 'crit_name')]

    model_class = None
    optim_class = None
    crit_class = None
    for k in module_list:
        model_class = getattr(sys.modules[k], model_name, model_class)
        optim_class = getattr(sys.modules[k], optim_name, optim_class)
        crit_class = getattr(sys.modules[k], crit_name, crit_class)

    assert model_class is not None, f'Could not find model class {model_name}'
    assert optim_class is not None, f'Could not find optimizer class {optim_name}'
    assert crit_class is not None, f'Could not find criterion class {crit_name}'

    try:
        constructor = ast.literal_eval(row[column_mapper['model_constructor']])  # :)
        model = model_class(**constructor)
    except:
        model = model_class()

    get_device = lambda: 'cuda' if torch.cuda.is_available() else 'cpu'
    device = device if device is not None else get_device()
    model.to(device)

    try:
        constructor = ast.literal_eval(row[column_mapper['optim_constructor']])  # :)
        optimizer = optim_class(model.parameters(), **constructor)
    except:
        try:
            lr = row[column_mapper['lr']]
            optimizer = optim_class(model.parameters(), lr=lr)
        except:
            optimizer = optim_class(model.parameters())

    try:
        constructor = ast.literal_eval(row[column_mapper['crit_constructor']])  # :)
        crit = crit_class(**constructor)
    except:
        crit = crit_class()

    net = NeuralNetwork(model, optimizer, crit)

    try:
        id = row[column_mapper.get('id', 'id')]
        unique_folder = osj(csv_folder, id)
    except:
        unique_folder = csv_folder

    weight_path = osj(unique_folder, 'weights.pth') if weight_path is None else (osj(unique_folder, weight_path) if relative_path else weight_path)

    if load_best:
        try:
            net.load_weights_from_path(weight_path)
        except:
            print('Did not find best weights, using original.')

    return net


