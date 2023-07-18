from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.metrics import Recall, TruePositives, FalseNegatives, Accuracy
from types import FunctionType as function

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import load_data, fit_data, ModelException, check_model_fit, check_model_load, check_data_load, check_model_predict




class Model:
    def __init__(self) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None
        self.signal_size = None
        self.history = None
        self.y_pred, self.y_pred_cat = None, None
        self.threshold = None


    def load_data(self, spikes_path: str, no_spikes_path: str = None, print_info: bool = False, not_spike_proportion: float = 1.0,
                 normalize: bool = False, randomize: bool = False) -> None:
        data = load_data(spikes_path, no_spikes_path, print_info, not_spike_proportion)
        self.X_train, self.X_test, self.y_train, self.y_test = fit_data(data, normalize, randomize)
        self.signal_size = self.X_train.shape[1]


    def load_model(self, model_path: str) -> None:
        self.model = keras.models.load_model(model_path)

    
    def set_model(self, model_architecture: function, **kwargs) -> None:
        try:
            self.model = model_architecture(signal_size=self.signal_size, **kwargs) 
        except:
            raise ModelException('problems with model getter function') 
    

    @check_model_load
    def predict(self, threshold: float = 0.5) -> None:
        self.y_pred = self.model.predict(self.X_test)
        self.change_threshold(threshold)


    @check_model_predict
    def change_threshold(self, threshold: float) -> None:
        self.y_pred_cat = [1 if i > threshold else 0 for i in self.y_pred]
        self.threshold = threshold


    @check_model_load
    def fit(self, batch_size: int = 16, validation_split: float = 0.2, epochs: int = 10, verbose: str = 'auto') -> None:
        self.history = self.model.fit(self.X_train, self.y_train, batch_size=batch_size, 
                                          validation_split=validation_split, epochs=epochs, verbose=verbose)


    @check_model_predict
    def plot_loss_function(self, save_fig: bool = False, file_name: str = 'loss_function.png') -> None:    
        plt.plot(self.history.history['loss'], label='test_split')
        plt.plot(self.history.history['val_loss'], label='valid_split')
        plt.title('loss_function')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.grid(True)
        if save_fig:
            plt.savefig(file_name)
        plt.show()
        plt.close()


    @check_model_predict
    def print_metrics(self) -> None:
        print('RECALL SCORE: ', recall_score(self.y_test, self.y_pred_cat))
        print('PRECISION SCORE: ', precision_score(self.y_test, self.y_pred_cat))
        print('ACCURACY SCORE: ', accuracy_score(self.y_test, self.y_pred_cat))


    @check_model_predict
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred_cat)
        sns.heatmap(cm, annot=True)
        plt.title('confusion matrix')
        plt.xlabel('prediction')
        plt.ylabel('groud truth')
        plt.show()
        plt.close()


    # change threshold and calculate metrics (recall, precision, accuracy)
    @check_model_predict
    def plot_threshold_recall(self, savefig: bool = False, file_name: str = 'threshold_recall.png') -> None:
        recalls = [0] * len(self.y_pred)
        accuracy = [0] * len(self.y_pred)
        precisions = [0] * len(self.y_pred)
        for threshold, i in zip(self.y_pred, [i for i in range(len(self.y_pred))]):
            if 2 > 1:
                y_proba_cat = np.copy(self.y_pred)
                y_proba_cat[y_proba_cat >= threshold] = 1
                y_proba_cat[y_proba_cat < threshold] = 0
                recalls[i] = recall_score(self.y_test, y_proba_cat)
                accuracy[i] = accuracy_score(self.y_test, y_proba_cat)
                precisions[i] = precision_score(self.y_test, y_proba_cat)

        plt.xlabel('threshold')
        plt.ylabel('metrics')
        plt.scatter(self.y_pred, recalls, label='recall')
        plt.scatter(self.y_pred, accuracy, label='accuracy')
        plt.scatter(self.y_pred, precisions, label='precision')
        plt.grid()
        plt.legend()
        if savefig:
            plt.savefig(file_name)
        plt.show()
        plt.close()
    

    # change threshold and calculate TP TF FN FP
    @check_model_predict
    def plot_confusion(self, savefig: bool =False, file_name: str ='confusion.png') -> None:   
        TP = [0] * len(self.y_pred)
        TN = [0] * len(self.y_pred)
        FN = [0] * len(self.y_pred)
        FP = [0] * len(self.y_pred)
        for threshold, i in zip(self.y_pred, [i for i in range(len(self.y_pred))]):
            if 2 > 1:
                y_proba_cat = np.copy(self.y_pred)
                y_proba_cat[y_proba_cat >= threshold] = 1
                y_proba_cat[y_proba_cat < threshold] = 0
                cm = confusion_matrix(self.y_test, y_proba_cat)
                TN[i] = cm[0][0]
                FP[i] = cm[0][1]
                FN[i] = cm[1][0]
                TP[i] = cm[1][1]

        plt.xlabel('threshold')
        plt.ylabel('metrics')
        #plt.scatter(self.y_pred, TN, label='True:0,Pred:0')
        #plt.scatter(self.y_pred, FP, label='True:0,Pred:1')
        plt.scatter(self.y_pred, FN, label='True:1,Pred:0')
        plt.scatter(self.y_pred, TP, label='True:1,Pred:1')
        plt.plot(self.y_pred, [sum(self.y_test)]*len(self.y_pred), label='True:1')
        plt.grid()
        plt.legend()
        if savefig:
            plt.savefig(file_name)
        plt.show()
        plt.close()


    @check_data_load
    def get_selections(self) -> tuple:
        return self.X_train, self.X_test, self.y_train, self.y_test
    

    @check_model_load
    def get_model(self) -> keras.Model:
        return self.model


    @check_model_load
    def save_model(self, model_path: str) -> None:
        self.model.save(f'{model_path}.h5')


