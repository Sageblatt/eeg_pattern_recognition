import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# get two datafiles' paths for training - with and without spikes
def load_data(spikes_path: str, no_spikes_path: str = None, 
              not_spikes_number: int = -1, print_info: bool = False) -> pd.DataFrame:
    """
    params: 
        spikes_path:            path to spikes file
        no_spikes_path:         path to no_spikes file, if None don't load 
        not_spikes_number:      number of not spikes that will be used, if less than 0 use all
        print_info:             print info
    """
    data = pd.read_csv(spikes_path, header=None)
    data['y'] = [1] * data.shape[0]
    if no_spikes_path:
        data2 = pd.read_csv(no_spikes_path, header=None)
        if not_spikes_number > 0:
            if not_spikes_number > data2.shape[0]:   Exception('Not spikes number is greater than they in array')
            data2 = data2[:not_spikes_number]
        data2['y'] = [0] * data2.shape[0]

        # if not_spikes_number > 0:
        #     remove_n = data2.shape[0] - not_spikes_number
        #     if remove_n <= 0:   Exception('Not spikes number is greater than they in array')
        # else:
        #     remove_n = int(data2.shape[0] * (1 - not_spike_proportion))
        # data2.drop(np.random.choice(data2.index, remove_n, replace=False), inplace=True)
        if print_info:
            print(f' \n \n spikes len: {data.shape[0]},  not spikes len: {data2.shape[0]}')
        data = pd.concat([data, data2], axis=0, join='inner', ignore_index=True)

    return data


# shuffle the data, cut into signs and answers
# allocate part of the data for training and verification
# data normalization
def fit_data(data: pd.DataFrame, normolize: bool = False, randomize: bool = False) -> tuple[np.array, np.array, np.array, np.array]:
    if randomize:
        data = data.sample(frac=1)

    y = data['y'].astype(np.float32)
    X = data.drop('y', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8,
                                                        random_state=42,
                                                        stratify=y
                                                        )
    # convert the dataframe into an array, for some reason the indexes are saved, so we delete the first element
    X_train = np.array([[element[i] for i in range(len(element)) if i >= 1] for element in X_train.to_numpy()])
    X_test = np.array([[element[i] for i in range(len(element)) if i >= 1] for element in X_test.to_numpy()])
    y_train = y_train.to_numpy().reshape((-1, 1))
    y_test = y_test.to_numpy().reshape((-1, 1))
    
    # normalize
    if normolize:
        X_train = np.array(
            [[(el - element.min()) / (element.max() - element.min()) for el in element] for element in X_train])
        X_test = np.array(
            [[(el - element.min()) / (element.max() - element.min()) for el in element] for element in X_test])
    return (X_train, X_test, y_train, y_test)



class ModelException(Exception):
    '''raise this when there's a problem with model'''


# decorators 
def check_model_fit(function: callable) -> callable:
    def decorator(self, *args, **kwargs):
        if not self.history:
            raise ModelException("model is not fitted")
        elif not self.model:
            raise ModelException("model is not loaded")
        elif self.X_test is None:
            raise ModelException("data is not loaded")
        else:
            function(self, *args, **kwargs)
                
    return decorator

def check_model_predict(function: callable) -> callable:
    def decorator(self, *args, **kwargs):
        if self.y_pred is None:
            raise ModelException("model has not predicted anything yet")
        elif not self.model:
            raise ModelException("model is not loaded")
        elif self.X_test is None:
            raise ModelException("data is not loaded")
        else:
            function(self, *args, **kwargs)
                
    return decorator

def check_model_load(function: callable) -> callable:
    def decorator(self, *args, **kwargs):
        if not self.model:
            raise ModelException("model is not loaded")
        elif self.X_test is None:
            raise ModelException("data is not loaded")
        else:
            function(self, *args, **kwargs)
                
    return decorator


def check_data_load(function: callable) -> callable:
    def decorator(self, *args, **kwargs):
        if self.X_train is None and self.X_test is None and self.y_train is None and self.y_test is None:
            raise ModelException("data is not loaded")
        else:
            function(self, *args, **kwargs)
                
    return decorator