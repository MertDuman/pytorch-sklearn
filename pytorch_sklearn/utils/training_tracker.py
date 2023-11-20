import pandas as pd
import numpy as np
import os
from os.path import join as osj


class TrainingTracker:
    import uuid

    def __init__(self, folder: list, hyperparameters: dict, metrics: dict=None, misc: dict=None, comment: str=None):
        self.folder = folder
        self.hyperparameters = hyperparameters
        self.metrics = metrics if metrics is not None else {}
        self.misc = misc if misc is not None else {}
        self.comment = comment if comment is not None else ""

        # validate parameter names
        if len((self.hyperparameters.keys() | self.metrics.keys() | self.misc.keys()) & set(self.get_hidden_keys())) != 0:
            raise ValueError("Do not use 'id', 'comment' or 'completed' as a parameter name.")

        # check identical parameter names
        if len(self.hyperparameters.keys() & self.metrics.keys()) != 0:
            raise ValueError("Hyperparameters and metrics cannot have an identical key.")
        elif len(self.hyperparameters.keys() & self.misc.keys()) != 0:
            raise ValueError("Hyperparameters and misc cannot have an identical key.")
        elif len(self.metrics.keys() & self.misc.keys()) != 0:
            raise ValueError("Metrics and misc cannot have an identical key.")

        self.hyper_keys = [key for key in self.hyperparameters]
        self.metric_keys = [key for key in self.metrics]
        self.misc_keys = [key for key in self.misc]

        self.training_index = None  # set when initialize_training is called

        # create folder if not exists
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # create the config file if not present, or read it
        try:
            self.config = self.read_config()
        except FileNotFoundError:
            print('writing')
            self.config = pd.DataFrame(columns=self.get_all_keys())
            self.write_config()

        # change config file based on new input
        self.assert_config_changes((self.hyperparameters.keys() | self.metrics.keys() | self.misc.keys()), self.get_config_set_except_hidden(), name="parameters")
        
        self.config = self.config.reindex(columns=self.get_all_keys())

        # set default value for NaN values
        for i, key in enumerate(self.config.columns):
            if self.config[key].isnull().all() and self.config[key].size != 0:
                if key in self.hyperparameters:
                    self.config[key] = self.hyperparameters[key]
                elif key in self.metrics:
                    self.config[key] = self.metrics[key]
                elif key in self.misc:
                    self.config[key] = self.misc[key]

        self.write_config()


    def start_training(self):
        """
        Adds a new empty row to the config, and returns the id.
        """
        self.config = self.read_config()
        new_id = str(self.uuid.uuid4())
        new_row = self.get_empty_row(new_id, [""] * len(self.hyper_keys), [""] * len(self.misc_keys), "")
        self.training_index = len(self.config)
        self.config.loc[self.training_index] = new_row
        self.write_config()
        return new_id


    def initialize_training(self, hyperparameters: dict, misc: dict=None, comment: str=None):
        """
        Adds a new row to the config file using this training's hyperparameters and returns a savename to be used. 
        """
        misc = misc if misc is not None else {}
        comment = comment if comment is not None else ""

        # this ensures any changes are reflected, but if parameters were added/removed/reordered by others, this function will soon raise an error.
        self.config = self.read_config()

        self.assert_diff(hyperparameters.keys(), self.hyperparameters.keys(), name="hyperparameters")
        self.assert_extra(misc.keys(), self.misc.keys(), name="misc")

        # reorders the given dictionaries to match the original
        hyperparameters = self.reorder_dict(hyperparameters, self.hyperparameters)
        misc = self.reorder_dict_relaxed(misc, self.misc)

        param_values = list(hyperparameters.values())
        misc_values = list(misc.values())
        
        # check if we already have an active training
        if self.training_index is not None:
            cur_id = self.config.loc[self.training_index, "id"]
            cur_row = self.get_init_row(cur_id, hyperparameters, misc, comment)
            self.config.loc[self.training_index] = cur_row
        # check if we already have a training with the same config
        elif (query := self.get_query(param_values)).any():
            ans = input("A training with the same config exists. Do you want to override? (y/n)")
            if ans.lower() != "y":
                raise RuntimeError("Interrupted on purpose. Training is not saved.")
            self.training_index = self.config.loc[query].index[0]
            self.config.loc[self.training_index, "completed"] = False
        else:  # no active training and no training with the same config
            self.training_index = len(self.config)
            new_id = str(self.uuid.uuid4())
            new_row = self.get_init_row(new_id, hyperparameters, misc, comment)
            self.config.loc[self.training_index] = new_row

        self.write_config()
        return self.config.loc[self.training_index, "id"]


    def finalize_training(self, metrics: dict=None, misc: dict=None, comment: str=None):
        """
        Updates the config file with the current training.
        """
        metrics = metrics if metrics is not None else {}
        misc = misc if misc is not None else {}

        # this ensures any changes are reflected, but if parameters were added/removed/reordered by others, this function will soon raise an error.
        self.config = self.read_config()

        self.assert_diff(metrics.keys(), self.metrics.keys(), name="metrics")
        self.assert_extra(misc.keys(), self.misc.keys(), name="misc")

        # reorders the given dictionaries to match the original
        metrics = self.reorder_dict(metrics, self.metrics)
        misc = self.reorder_dict_relaxed(misc, self.misc)

        metric_values = list(metrics.values())
        
        # override misc if needed
        misc_values = [misc[k] if k in misc else self.config.loc[self.training_index, k] for k in self.misc]

        # override comment if needed
        comment = self.config.loc[self.training_index, "comment"] if comment is None else comment
        
        self.config.loc[self.training_index, self.metric_keys] = np.array(metric_values, dtype=object)
        self.config.loc[self.training_index, self.misc_keys] = np.array(misc_values, dtype=object)
        self.config.loc[self.training_index, "comment"] = comment
        self.config.loc[self.training_index, "completed"] = True
        self.write_config()


    def set_current_training_from_index(self, index):
        self.config = self.read_config()
        assert index < len(self.config), f"Index out of bounds ({index} >= {len(self.config)}). Training at index {index} does not exist."
        self.training_index = index
        self.config.loc[self.training_index, "completed"] = False
        self.write_config()
        return self.config.loc[self.training_index]["id"]

    
    def set_current_training_completed(self):
        self.config = self.read_config()
        assert self.training_index is not None, "No training is active."
        assert 0 <= self.training_index < len(self.config), f"Index out of bounds ({self.training_index} >= {len(self.config)}). Training at index {self.training_index} does not exist."
        self.config.loc[self.training_index, "completed"] = True
        self.write_config()


    def set_all_trainings_completed(self):
        self.config = self.read_config()
        self.config["completed"] = True
        self.write_config()


    def clean_folder(self):
        """
        Deletes all subfolders within the folder that are not in the config file.
        """
        self.config = self.read_config()
        for id in os.listdir(self.folder):
            # check if id is uuid
            try:
                self.uuid.UUID(id)
            except ValueError:
                continue
            if id not in self.config["id"].values:
                import shutil
                shutil.rmtree(osj(self.folder, id))
        

    def get_files(self, hyperparameters: dict=None, id: str=None):
        """
        Returns all the files inside this training's corresponding folder.
        """
        if hyperparameters is None and id is None:
            raise ValueError("Pass either hyperparameters or id.")

        if id is not None:
            if os.path.exists(osj(self.folder, id)):
                return [osj(self.folder, id, filename) for filename in os.listdir(osj(self.folder, id))]
            else:
                raise FileNotFoundError(f"File {osj(self.folder, id)} does not exist.")
            
        self.assert_diff(hyperparameters.keys(), self.hyperparameters.keys(), name="hyperparameters")
            
        # reorders the given dictionaries to match the original
        hyperparameters = self.reorder_dict(hyperparameters, self.hyperparameters)
        param_values = list(hyperparameters.values())

        query = self.get_query(param_values)
        
        if query.any():
            print(self.config.loc[query, "id"])
            id = self.config.loc[query, "id"].iloc[0]
            if os.path.exists(osj(self.folder, id)):
                return [osj(self.folder, id, filename) for filename in os.listdir(osj(self.folder, id))]
            
        return []

            
    def delete_folder(self):
        import shutil
        ans = input(f"Are you sure you want to delete the folder '{self.folder}' and all its contents? (yes/n)")
        if ans.lower() == "yes":
            shutil.rmtree(self.folder)

                    
    def assert_config_changes(self, keys_new: set, keys_old: set, name: str="hyperparameters"):
        added = keys_new - keys_old
        n_added = len(added)
        removed = keys_old - keys_new
        n_removed = len(removed)
        num_uncompleted = len(self.get_uncompleted())

        if n_removed > 0 or n_added > 0:
            if num_uncompleted > 0:
                raise RuntimeError(f"Cannot add or remove parameters when there are {num_uncompleted} incomplete trainings.")
            ans = input(f"This will remove {n_removed} {name} and add {n_added} new {name}. Are you sure? (y/n)")
            if ans.lower() != "y":
                raise RuntimeError("Interrupted on purpose. Config file was not changed.")
                
        key_order = self.get_all_keys()

        i = 0
        j = 0
        while i < len(key_order) and j < len(self.config.columns):
            if key_order[i] in added:
                i += 1
            elif self.config.columns[j] in removed:
                j += 1
            elif key_order[i] == self.config.columns[j]:
                i += 1
                j += 1
            else:  # key_order[i] != self.config.columns[j]
                if num_uncompleted > 0:
                    raise RuntimeError(f"Cannot reorder parameters ({self.config.columns[j]} <-> {key_order[i]}) when there are {num_uncompleted} incomplete trainings.")
                ans = input(f"This will change the ordering of the parameters. Are you sure? (y/n)")
                if ans.lower() != "y":
                    raise RuntimeError("Interrupted on purpose. Config file was not changed.")
                break

    def assert_diff(self, keys_passed: set, keys_existing: set, name: str="hyperparameters"):
        n_extra = len(keys_passed - keys_existing)
        n_missing = len(keys_existing - keys_passed)
        if n_missing > 0:
            raise ValueError(f"Some {name} are missing. Pass all of: {keys_existing - keys_passed}")
        elif n_extra > 0:
            raise ValueError(f"Extra {name}. Remove: {keys_passed - keys_existing}")

    def assert_extra(self, keys_passed: set, keys_existing: set, name: str="hyperparameters"):
        n_extra = len(keys_passed - keys_existing)
        if n_extra > 0:
            raise ValueError(f"Extra {name}. Remove: {keys_passed - keys_existing}")

    def reorder_dict(self, dict_passed: dict, dict_existing: dict):
        return {k: dict_passed[k] for k in dict_existing.keys()}

    def reorder_dict_relaxed(self, dict_passed: dict, dict_existing: dict):
        return {k: dict_passed[k] for k in dict_existing.keys() if k in dict_passed}

    def read_config(self):
        return pd.read_csv(osj(self.folder, "config.csv"), index_col=0, keep_default_na=False)

    def write_config(self):
        self.config.to_csv(osj(self.folder, "config.csv"))

    def get_query(self, hyperparameter_values: list):
        values = [self.value_to_csv(v) for v in hyperparameter_values]
        return (self.config[self.hyper_keys] == values).all(axis=1)
        
    def value_to_csv(self, value):
        if isinstance(value, list):
            return str(value)
        elif isinstance(value, dict):
            return str(value)
        return value

    def get_config_set_except_hidden(self):
        return set(self.config.columns) - set(self.get_hidden_keys())

    def get_all_keys(self):
        return ["id"] + self.hyper_keys + self.metric_keys + self.misc_keys + ["comment", "completed"]

    def get_empty_row(self, id: str, hyperparameter_values: list, misc_values: list, comment: str):
        return np.array([id] + hyperparameter_values + [""] * len(self.metric_keys) + misc_values + [comment, False], dtype=object)

    def get_init_row(self, id: str, hyperparameters: dict, misc: dict, comment: str):
        # set misc as "" if not passed.
        misc_values = [misc[k] if k in misc else "" for k in self.misc]

        return np.array([
            id,
            *list(hyperparameters.values()),
            *([""] * len(self.metric_keys)),
            *misc_values,
            comment,
            False
        ], dtype=object)

    def get_uncompleted(self):
        return self.config.loc[self.config["completed"] == False, "id"]

    def get_hidden_keys(self):
        return ["id", "comment", "completed"]