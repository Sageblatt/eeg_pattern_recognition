from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.metrics import Recall, TruePositives, FalseNegatives, Accuracy

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# get two datafiles' paths for training - with and without spikes
def load_data(spikes_path, no_spikes_path=None, not_spike_proportion=1.0):
    data = pd.read_csv(spikes_path, header=None)
    data['y'] = [1 for i in range(data.shape[0])]
    if no_spikes_path:
        data2 = pd.read_csv(no_spikes_path, header=None)
        data2['y'] = [0 for i in range(data2.shape[0])]
        remove_n = int(data2.shape[0] * (1 - not_spike_proportion))
        data2.drop(np.random.choice(data2.index, remove_n, replace=False), inplace=True)
        print('spikes len: ', data.shape[0], ' not spikes len: ', data2.shape[0])
        data = pd.concat([data, data2], axis=0, join='inner', ignore_index=True)

    return data


# shuffle the data, cut into signs and answers
# allocate part of the data for training and verification
# data normalization
def fit_data(data, normolize=False, randomize=False):
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


# example of the simplest neural network
def simple_network(metrics, save_fig=False, batch_size=16, validation_split=0.2, epochs=10,
                   not_spike_proportion=1.0, first_dense_neurons=40, showfig=True,
                   verbose='auto', save=False):
    path = 'data/spikes_beginning.csv'
    path2 = 'data/not_spikes.csv'

    data = load_data(path, path2, not_spike_proportion=not_spike_proportion)
    x_train, x_test, y_train, y_test = fit_data(data)
    signal_size = x_train.shape[1]
    model = keras.Sequential([Flatten(input_shape=(signal_size, 1)),
                              Dense(first_dense_neurons, activation='relu'),
                              Dense(1, activation='sigmoid')
                              ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=metrics
                  )

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)

    # plot loss function
    plt.plot(history.history['loss'], label='test_split')
    plt.plot(history.history['val_loss'], label='valid_split')
    plt.title('loss_function')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)

    # save result
    if save_fig:
        plt.savefig('NN.png')
    if showfig:
        plt.show()

    plt.close()

    print('Training completed, verification results:')
    model.evaluate(x_test, y_test, verbose=verbose)
    y_pred = model.predict(x_test, verbose=verbose)

    y_train_pred = model.predict(x_train, verbose=verbose)

    #save model
    if save:
        model.keras.save('model')

    return y_pred, y_test, y_train_pred, y_train, x_test


# load model
def model_predict(path_to_data):
    path = 'data/spikes_beginning.csv'
    path2 = 'data/not_spikes.csv'

    data = load_data(path, path2)
    x_train, x_test, y_train, y_test = fit_data(data)
    signal_size = x_train.shape[1]
    model = keras.Sequential([Flatten(input_shape=(signal_size, 1)),
                              Dense(40, activation='relu'),
                              Dense(1, activation='sigmoid')
                              ])
    threshold = 0.5
    metrics = [Recall(name='recall', thresholds=[threshold])]
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=metrics
                  )

    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    data = pd.read_csv(path_to_data, header=None)
    data = np.array([[element[i] for i in range(len(element)) if i >= 1] for element in data.to_numpy()])

    y_pred = model.predict(data)
    return y_pred


####################### DATA VISUALISATION #######################

# change threshold and calculate metrics
def plot_threshold_recall(y_proba, y_true, savefig=False, showfig=True):
    recalls = [0] * len(y_proba)
    accuracy = [0] * len(y_proba)
    precisions = [0] * len(y_proba)
    for threshold, i in zip(y_proba, [i for i in range(len(y_proba))]):
        if i % 2 == 1:
            y_proba_cat = np.copy(y_proba)
            y_proba_cat[y_proba_cat >= threshold] = 1
            y_proba_cat[y_proba_cat < threshold] = 0
            recalls[i] = recall_score(y_true, y_proba_cat)
            accuracy[i] = accuracy_score(y_true, y_proba_cat)
            precisions[i] = precision_score(y_true, y_proba_cat)

    plt.xlabel('threshold')
    plt.ylabel('metrics')
    plt.scatter(y_proba, recalls, label='recall')
    plt.scatter(y_proba, accuracy, label='accuracy')
    plt.scatter(y_proba, precisions, label='precision')
    plt.grid()
    plt.legend()

    if savefig:
        plt.savefig('result.png')

    if showfig:
        plt.show()


# change threshold and calculate TP TF FN FP
def plot_confusion(y_proba, y_true, savefig=False):
    TP = [0] * len(y_proba)
    TN = [0] * len(y_proba)
    FN = [0] * len(y_proba)
    FP = [0] * len(y_proba)
    for threshold, i in zip(y_proba, [i for i in range(len(y_proba))]):
        if i % 2 == 1:
            y_proba_cat = np.copy(y_proba)
            y_proba_cat[y_proba_cat >= threshold] = 1
            y_proba_cat[y_proba_cat < threshold] = 0
            cm = confusion_matrix(y_true, y_proba_cat)

            TN[i] = cm[0][0]
            FP[i] = cm[0][1]
            FN[i] = cm[1][0]
            TP[i] = cm[1][1]

    plt.xlabel('threshold')
    plt.ylabel('metrics')
    #plt.scatter(y_proba, TN, label='True:0,Pred:0')
    #plt.scatter(y_proba, FP, label='True:0,Pred:1')
    plt.scatter(y_proba, FN, label='True:1,Pred:0')
    plt.scatter(y_proba, TP, label='True:1,Pred:1')
    plt.plot(y_proba, [sum(y_true)]*len(y_proba), label='True:1')
    plt.grid()
    plt.legend()

    if savefig:
        plt.savefig('result.png')

    else:
        plt.show()


# plot spikes of False Positive error
def plot_spikes_FP(x, y_pred, y_test, number):
    plt.close()
    s = 0
    for spike, pred, true in zip(x, y_pred, y_test):
        if pred == 0 and true == 1:
            s+=1
            plt.plot([i for i in range(len(spike))], spike)
            plt.title('NN thinks it is no spike')
            plt.show()

        if s >=number:
            break



def plot_graphics(y_test, y_pred_cat, print_metrics=True,
                  confusionmatrix=True):
    # print metrics
    if print_metrics:
        print('RECALL SCORE: ', recall_score(y_test, y_pred_cat))
        print('PRECISION SCORE: ', precision_score(y_test, y_pred_cat))
        print('ACCURACY SCORE: ', accuracy_score(y_test, y_pred_cat))

    # plot confusion_matrix
    if confusionmatrix:
        cm = confusion_matrix(y_test, y_pred_cat)
        sns.heatmap(cm, annot=True)
        plt.title('confusion matrix')
        plt.xlabel('prediction')
        plt.ylabel('groud truth')
        plt.show()



####################### TESTING #######################
def test_hyper_parameters(hyperparameters, STEPS_FOR_AVERAGING = 10):
    STEPS_FOR_TESTING = len(hyperparameters)
    accuracy = [0] * STEPS_FOR_TESTING
    recall = [0] * STEPS_FOR_TESTING
    precision = [0] * STEPS_FOR_TESTING

    for hyper_parameter, j in zip(hyperparameters, [i for i in range(len(hyperparameters))]):
        print('STEP ', j, ' / ', STEPS_FOR_TESTING)
        for i in range(STEPS_FOR_AVERAGING):
            y_pred, y_test, y_train_pred, y_train, x_test = simple_network(metrics, batch_size=32,
                                                                   validation_split=0.2, epochs=50,
                                                                   showfig=False, not_spike_proportion=1.0,
                                                                       first_dense_neurons=hyper_parameter,
                                                                   verbose=0)

            # fit probability to logit
            y_pred_cat = np.copy(y_pred)
            y_pred_cat[y_pred_cat >= threshold] = 1
            y_pred_cat[y_pred_cat < threshold] = 0

            # calculate metrics
            accuracy[j] += (accuracy_score(y_test, y_pred_cat) / STEPS_FOR_AVERAGING)
            recall[j] += (recall_score(y_test, y_pred_cat) / STEPS_FOR_AVERAGING)
            precision[j] += (precision_score(y_test, y_pred_cat) / STEPS_FOR_AVERAGING)

    plt.close()
    plt.plot(hyperparameters, accuracy, label='accuracy')
    plt.plot(hyperparameters, recall, label='recall')
    plt.plot(hyperparameters, precision, label='precision')

    plt.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    threshold = 0.5

    metrics = [Recall(name='recall', thresholds=[threshold])]

    print(model_predict('data/spikes_beginning.csv'))




    # plot threshold-recall graph
    # plot_threshold_recall(y_pred, y_test)
    # plot_threshold_recall(y_train_pred, y_train)
    # plot_confusion(y_pred, y_test, savefig=False)
