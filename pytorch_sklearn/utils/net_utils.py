import os
from os.path import join as osj
import pandas as pd
import torch
from pytorch_sklearn import NeuralNetwork
from pytorch_sklearn.neural_network.generative_network import CycleGAN, R2CGAN, GAN
from pytorch_sklearn.callbacks import Callback

from typing import Mapping, Iterable, Type
import ast


def try_load_network_from_csv(
    csv_folder: str, 
    csv_filename: str,
    idx: int,
    column_mapper: Mapping[str, str] = None,
    net_path: str = None,
    load_best: str = False,
    weight_path: str = None,
    relative_path: bool = True,
    callbacks: Iterable[Callback] = None,
    module_list: Iterable[str] = None,
    device: str = None,
    net_class: Type[NeuralNetwork] = NeuralNetwork,
    supress_warnings: bool = False,
    strict: bool = True
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
    column_mapper = column_mapper if column_mapper is not None else {}

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
        if not supress_warnings: print('Could not find model constructor, using default.')
        model = model_class()

    get_device = lambda: 'cuda' if torch.cuda.is_available() else 'cpu'
    device = device if device is not None else get_device()
    model.to(device)

    try:
        constructor = ast.literal_eval(row[column_mapper['optim_constructor']])  # :)
        optimizer = optim_class(model.parameters(), **constructor)
    except:
        if not supress_warnings: print('Could not find optimizer constructor, using default.')
        try:
            lr = row[column_mapper['lr']]
            optimizer = optim_class(model.parameters(), lr=lr)
        except:
            optimizer = optim_class(model.parameters())

    try:
        constructor = ast.literal_eval(row[column_mapper['crit_constructor']])  # :)
        crit = crit_class(**constructor)
    except:
        if not supress_warnings: print('Could not find criterion constructor, using default.')
        crit = crit_class()

    net = net_class(model, optimizer, crit)

    try:
        id = row[column_mapper.get('id', 'id')]
        unique_folder = osj(csv_folder, id)
    except:
        unique_folder = csv_folder

    net_path = osj(unique_folder, 'net.pth') if net_path is None else (osj(unique_folder, net_path) if relative_path else net_path)
    weight_path = osj(unique_folder, 'weights.pth') if weight_path is None else (osj(unique_folder, weight_path) if relative_path else weight_path)
    
    net_class.load_class(net, callbacks, net_path, strict=strict)

    if load_best:
        try:
            net.load_weights_from_path(weight_path, strict=strict)
        except:
            if not supress_warnings: print('Did not find best weights, using original.')

    return net


def try_load_model_from_csv(
    csv_folder: str, 
    csv_filename: str,
    idx: int,
    column_mapper: Mapping[str, str] = None,
    load_best: str = False,
    weight_path: str = None,
    relative_path: bool = True,
    module_list: Iterable[str] = None,
    device: str = None,
    net_class: Type[NeuralNetwork] = NeuralNetwork,
    supress_warnings: bool = False
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
    module_list = sys.modules.copy() if module_list is None else module_list
    column_mapper = column_mapper if column_mapper is not None else {}

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
        constructor = ast.literal_eval(row[column_mapper.get('model_constructor', 'model_constructor')])  # :)
        model = model_class(**constructor)
    except:
        if not supress_warnings: print('Could not find model constructor, using default.')
        model = model_class()

    get_device = lambda: 'cuda' if torch.cuda.is_available() else 'cpu'
    device = device if device is not None else get_device()
    model.to(device)

    try:
        constructor = ast.literal_eval(row[column_mapper.get('optim_constructor', 'optim_constructor')])  # :)
        optimizer = optim_class(model.parameters(), **constructor)
    except:
        if not supress_warnings: print('Could not find optimizer constructor, using default.')
        try:
            lr = row[column_mapper['lr']]
            optimizer = optim_class(model.parameters(), lr=lr)
        except:
            optimizer = optim_class(model.parameters())

    try:
        constructor = ast.literal_eval(row[column_mapper.get('crit_constructor', 'crit_constructor')])  # :)
        crit = crit_class(**constructor)
    except:
        if not supress_warnings: print('Could not find criterion constructor, using default.')
        crit = crit_class()

    net = net_class(model, optimizer, crit)

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
            if not supress_warnings: print('Did not find best weights, using original.')

    return net








def try_load_cyclegan_from_csv(
    csv_folder: str, 
    csv_filename: str,
    idx: int,
    column_mapper: Mapping[str, str] = None,
    net_path: str = None,
    load_best: str = False,
    weight_path: str = None,
    relative_path: bool = True,
    callbacks: Iterable[Callback] = None,
    module_list: Iterable[str] = None,
    device: str = None,
    cyclegan_class: Type[CycleGAN] = CycleGAN,
    supress_warnings: bool = False,
    strict: bool = True,
): 
    '''
    Tries to load the CycleGAN network from a csv file.

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
            - models: class name of the model
            - optimizers: class name of the optimizer
        If not given, assumes the column names are 'models' and 'optimizers'.

        These keys are likely needed:
            - id: A unique identifier for the network. This csv row has a unique id that links to a folder within csv_folder.
            - constructors: dictionary of arguments to pass to the models. If not present, assumes no arguments.
            - optim_constructors: dictionary of arguments to pass to the optimizers. If not present, assumes no arguments.
            - lr: learning rate to pass to the optimizers. Some optimizers require a learning rate, like SGD.
            - gan_params: Dictionary of arguments to pass to CycleGAN. If not present, assumes no arguments.

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
    return _try_load_cyclegan_from_csv(
        csv_folder=csv_folder,
        csv_filename=csv_filename,
        idx=idx,
        column_mapper=column_mapper,
        net_path=net_path,
        load_best=load_best,
        weight_path=weight_path,
        relative_path=relative_path,
        callbacks=callbacks,
        module_list=module_list,
        device=device,
        cyclegan_class=cyclegan_class,
        supress_warnings=supress_warnings,
        strict=strict
    )


def try_load_r2cgan_from_csv(
    csv_folder: str, 
    csv_filename: str,
    idx: int,
    column_mapper: Mapping[str, str] = None,
    net_path: str = None,
    load_best: str = False,
    weight_path: str = None,
    relative_path: bool = True,
    callbacks: Iterable[Callback] = None,
    module_list: Iterable[str] = None,
    device: str = None,
    r2cgan_class: Type[R2CGAN] = R2CGAN,
    supress_warnings: bool = False,
    strict: bool = True
): 
    '''
    Tries to load the R2CGAN network from a csv file.

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
            - models: class name of the model
            - optimizers: class name of the optimizer
        If not given, assumes the column names are 'models' and 'optimizers'.

        These keys are likely needed:
            - id: A unique identifier for the network. This csv row has a unique id that links to a folder within csv_folder.
            - constructors: dictionary of arguments to pass to the models. If not present, assumes no arguments.
            - optim_constructors: dictionary of arguments to pass to the optimizers. If not present, assumes no arguments.
            - lr: learning rate to pass to the optimizers. Some optimizers require a learning rate, like SGD.
            - gan_params: Dictionary of arguments to pass to R2CGAN. If not present, assumes no arguments.

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
    return _try_load_cyclegan_from_csv(
        csv_folder=csv_folder,
        csv_filename=csv_filename,
        idx=idx,
        column_mapper=column_mapper,
        net_path=net_path,
        load_best=load_best,
        weight_path=weight_path,
        relative_path=relative_path,
        callbacks=callbacks,
        module_list=module_list,
        device=device,
        cyclegan_class=r2cgan_class,
        supress_warnings=supress_warnings,
        strict=strict
    )
        


def _try_load_cyclegan_from_csv(
    csv_folder: str, 
    csv_filename: str,
    idx: int,
    column_mapper: Mapping[str, str] = None,
    net_path: str = None,
    load_best: str = False,
    weight_path: str = None,
    relative_path: bool = True,
    callbacks: Iterable[Callback] = None,
    module_list: Iterable[str] = None,
    device: str = None,
    cyclegan_class = CycleGAN,
    supress_warnings: bool = False,
    strict: bool = True,
): 
    import sys
    callbacks = [] if callbacks is None else callbacks
    module_list = sys.modules.copy() if module_list is None else module_list
    column_mapper = column_mapper if column_mapper is not None else {}

    csv_file = pd.read_csv(osj(csv_folder, csv_filename), index_col=0, keep_default_na=False)
    
    row = csv_file.iloc[idx]
    models = row[column_mapper.get('models', 'models')]
    optimizers = row[column_mapper.get('optimizers', 'optimizers')]

    models = ast.literal_eval(models)
    optimizers = ast.literal_eval(optimizers)

    model_classes = []
    for model_name in models.values():
        for k in module_list:
            model_class = getattr(sys.modules[k], model_name, [])
            if model_class != []:
                model_classes.append(model_class)
                break
    optim_classes = []
    for optim_name in optimizers.values():
        for k in module_list:
            optim_class = getattr(sys.modules[k], optim_name, [])
            if optim_class != []:
                optim_classes.append(optim_class)
                break

    class_set = set([model_class.__name__ for model_class in model_classes])
    model_set = set(models.values())
    assert class_set == model_set, f'Could not find model classes {model_set - class_set}'
    class_set = set([optim_class.__name__ for optim_class in optim_classes])
    optim_set = set(optimizers.values())
    assert class_set == optim_set, f'Could not find optimizer classes {optim_set - class_set}'

    try:
        constructors = ast.literal_eval(row[column_mapper.get('constructors', 'constructors')])  # :)
        try:
            G_A = model_classes[0](**constructors['G_A'])
        except:
            if not supress_warnings: print('No constructor found for G_A, using default.')
            G_A = model_classes[0]()
        try:
            G_B = model_classes[1](**constructors['G_B'])
        except:
            if not supress_warnings: print('No constructor found for G_B, using default.')
            G_B = model_classes[1]()
        try:
            D_A = model_classes[2](**constructors['D_A'])
        except:
            if not supress_warnings: print('No constructor found for D_A, using default.')
            D_A = model_classes[2]()
        try:
            D_B = model_classes[3](**constructors['D_B'])
        except:
            if not supress_warnings: print('No constructor found for D_B, using default.')
            D_B = model_classes[3]()
    except:
        if not supress_warnings: print('No constructors found, using default.')
        G_A = model_classes[0]()
        G_B = model_classes[1]()
        D_A = model_classes[2]()
        D_B = model_classes[3]()
        
    get_device = lambda: 'cuda' if torch.cuda.is_available() else 'cpu'
    device = device if device is not None else get_device()
    G_A.to(device), G_B.to(device), D_A.to(device), D_B.to(device)

    lr = None
    try:
        lr = row[column_mapper['lr']]
    except:
        pass

    try:
        constructors = ast.literal_eval(row[column_mapper.get('optim_constructors', 'optim_constructors')])  # :)
        try:
            G_optim_constructor = constructors['G_optim']
            if lr is not None:
                G_A_optim = optim_classes[0](list(G_A.parameters()) + list(G_B.parameters()), lr=lr, **G_optim_constructor)
            else:
                G_A_optim = optim_classes[0](list(G_A.parameters()) + list(G_B.parameters()), **G_optim_constructor)
        except:
            if not supress_warnings: print('No constructor found for G_A_optim, using default.')
            if lr is not None:
                G_A_optim = optim_classes[0](list(G_A.parameters()) + list(G_B.parameters()), lr=lr)
            else:
                G_A_optim = optim_classes[0](list(G_A.parameters()) + list(G_B.parameters()))

        try:
            D_optim_constructor = constructors['D_optim']
            if lr is not None:
                D_A_optim = optim_classes[1](list(D_A.parameters()) + list(D_B.parameters()), lr=lr, **D_optim_constructor)
            else:
                D_A_optim = optim_classes[1](list(D_A.parameters()) + list(D_B.parameters()), **D_optim_constructor)
        except:
            if not supress_warnings: print('No constructor found for D_A_optim, using default.')
            if lr is not None:
                D_A_optim = optim_classes[1](list(D_A.parameters()) + list(D_B.parameters()), lr=lr)
            else:
                D_A_optim = optim_classes[1](list(D_A.parameters()) + list(D_B.parameters()))
    except: 
        if not supress_warnings: print('No constructor found for optimizers, using default.')
        if lr is not None:
            G_A_optim = optim_classes[0](list(G_A.parameters()) + list(G_B.parameters()), lr=lr)
            D_A_optim = optim_classes[1](list(D_A.parameters()) + list(D_B.parameters()), lr=lr)
        else:
            G_A_optim = optim_classes[0](list(G_A.parameters()) + list(G_B.parameters()))
            D_A_optim = optim_classes[1](list(D_A.parameters()) + list(D_B.parameters()))

    try:
        gan_params = ast.literal_eval(row[column_mapper.get('gan_params', 'gan_params')])
    except:
        if not supress_warnings: print('No constructor found for the network, using default.')
        gan_params = {}

    net = cyclegan_class(G_A, G_B, D_A, D_B, G_A_optim, D_A_optim, **gan_params)

    try:
        id = row[column_mapper.get('id', 'id')]
        unique_folder = osj(csv_folder, id)
    except:
        unique_folder = csv_folder

    net_path = osj(unique_folder, 'net.pth') if net_path is None else (osj(unique_folder, net_path) if relative_path else net_path)
    weight_path = osj(unique_folder, 'weights.pth') if weight_path is None else (osj(unique_folder, weight_path) if relative_path else weight_path)
    
    cyclegan_class.load_class(net, callbacks, net_path, strict=strict)

    if load_best:
        try:
            net.load_weights_from_path(weight_path, strict=strict)
        except:
            if not supress_warnings: print('Did not find best weights, using original.')

    return net








def try_load_gan_from_csv(
    csv_folder: str, 
    csv_filename: str,
    idx: int,
    column_mapper: Mapping[str, str] = None,
    net_path: str = None,
    load_best: str = False,
    weight_path: str = None,
    relative_path: bool = True,
    callbacks: Iterable[Callback] = None,
    module_list: Iterable[str] = None,
    device: str = None,
    gan_class = GAN,
    supress_warnings: bool = False,
    raise_errors: bool = False,
    strict: bool = True,
): 
    '''
    Tries to load the GAN network from a csv file.

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
            - models: class name of the model
            - optimizers: class name of the optimizer
        If not given, assumes the column names are 'models' and 'optimizers'.

        These keys are likely needed:
            - id: A unique identifier for the network. This csv row has a unique id that links to a folder within csv_folder.
            - constructors: dictionary of arguments to pass to the models. If not present, assumes no arguments.
            - optim_constructors: dictionary of arguments to pass to the optimizers. If not present, assumes no arguments.
            - lr: learning rate to pass to the optimizers. Some optimizers require a learning rate, like SGD.
            - gan_params: Dictionary of arguments to pass to R2CGAN. If not present, assumes no arguments.

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
    column_mapper = column_mapper if column_mapper is not None else {}

    csv_file = pd.read_csv(osj(csv_folder, csv_filename), index_col=0, keep_default_na=False)
    
    row = csv_file.iloc[idx]
    models = row[column_mapper.get('models', 'models')]
    optimizers = row[column_mapper.get('optimizers', 'optimizers')]
    criterion = row[column_mapper.get('criterion', 'criterion')]

    models = ast.literal_eval(models)
    optimizers = ast.literal_eval(optimizers)

    model_classes = []
    for model_name in models.values():
        for k in module_list:
            model_class = getattr(sys.modules[k], model_name, [])
            if model_class != []:
                model_classes.append(model_class)
                break
    optim_classes = []
    for optim_name in optimizers.values():
        for k in module_list:
            optim_class = getattr(sys.modules[k], optim_name, [])
            if optim_class != []:
                optim_classes.append(optim_class)
                break
    crit_class = None
    for k in module_list:
        crit_class = getattr(sys.modules[k], criterion, crit_class)

    class_set = set([model_class.__name__ for model_class in model_classes])
    model_set = set(models.values())
    assert class_set == model_set, f'Could not find model classes {model_set - class_set}'
    class_set = set([optim_class.__name__ for optim_class in optim_classes])
    optim_set = set(optimizers.values())
    assert class_set == optim_set, f'Could not find optimizer classes {optim_set - class_set}'
    assert crit_class is not None, f'Could not find criterion class {crit_name}'

    try:
        constructors = ast.literal_eval(row[column_mapper.get('model_ctors', 'model_ctors')])  # :)
        try:
            G = model_classes[0](**constructors['G'])
        except:
            if not supress_warnings: print('No constructor found for G, using default.')
            G = model_classes[0]()
            if raise_errors: raise
        try:
            D = model_classes[1](**constructors['D'])
        except:
            if not supress_warnings: print('No constructor found for D, using default.')
            D = model_classes[1]()
            if raise_errors: raise
    except:
        if not supress_warnings: print('No constructors found, using default.')
        G = model_classes[0]()
        D = model_classes[1]()
        if raise_errors: raise
        
    get_device = lambda: 'cuda' if torch.cuda.is_available() else 'cpu'
    device = device if device is not None else get_device()
    G.to(device), D.to(device)

    lr = None
    try:
        lr = row[column_mapper['lr']]
    except:
        pass

    try:
        constructors = ast.literal_eval(row[column_mapper.get('optimizer_ctors', 'optimizer_ctors')])  # :)
        try:
            G_optim_constructor = constructors['G_optim']
            if lr is not None:
                G_optim = optim_classes[0](G.parameters(), lr=lr, **G_optim_constructor)
            else:
                G_optim = optim_classes[0](G.parameters(), **G_optim_constructor)
        except:
            if not supress_warnings: print('No constructor found for G_optim, using default.')
            if lr is not None:
                G_optim = optim_classes[0](G.parameters(), lr=lr)
            else:
                G_optim = optim_classes[0](G.parameters())
            if raise_errors: raise

        try:
            D_optim_constructor = constructors['D_optim']
            if lr is not None:
                D_optim = optim_classes[1](D.parameters(), lr=lr, **D_optim_constructor)
            else:
                D_optim = optim_classes[1](D.parameters(), **D_optim_constructor)
        except:
            if not supress_warnings: print('No constructor found for D_optim, using default.')
            if lr is not None:
                D_optim = optim_classes[1](D.parameters(), lr=lr)
            else:
                D_optim = optim_classes[1](D.parameters())
            if raise_errors: raise
    except: 
        if not supress_warnings: print('No constructor found for optimizers, using default.')
        if lr is not None:
            G_optim = optim_classes[0](G.parameters(), lr=lr)
            D_optim = optim_classes[1](D.parameters(), lr=lr)
        else:
            G_optim = optim_classes[0](G.parameters())
            D_optim = optim_classes[1](D.parameters())
        if raise_errors: raise

    try:
        constructor = ast.literal_eval(row[column_mapper.get('criterion_ctor', 'criterion_ctor')])  # :)
        crit = crit_class(**constructor)
    except:
        if not supress_warnings: print('Could not find criterion constructor, using default.')
        crit = crit_class()
        if raise_errors: raise

    try:
        net_ctor = ast.literal_eval(row[column_mapper.get('net_ctor', 'net_ctor')])
    except:
        if not supress_warnings: print('No constructor found for the network, using default.')
        net_ctor = {}
        if raise_errors: raise

    net = gan_class(G, D, G_optim, D_optim, crit, **net_ctor)

    try:
        id = row[column_mapper.get('id', 'id')]
        unique_folder = osj(csv_folder, id)
    except:
        unique_folder = csv_folder
        if raise_errors: raise

    net_path = osj(unique_folder, 'net.pth') if net_path is None else (osj(unique_folder, net_path) if relative_path else net_path)
    weight_path = osj(unique_folder, 'weights.pth') if weight_path is None else (osj(unique_folder, weight_path) if relative_path else weight_path)
    
    gan_class.load_class(net, callbacks, net_path, strict=strict)

    if load_best:
        try:
            net.load_weights_from_path(weight_path, strict=strict)
        except:
            if not supress_warnings: print('Did not find best weights, using original.')
            if raise_errors: raise

    return net