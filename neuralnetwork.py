from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix
from keras.utils.np_utils import to_categorical
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# get two datafiles' paths for training - with and without spikes
def load_data(spikes_path, no_spikes_path=None):
    data = pd.read_csv(spikes_path)
    data['y'] = [1 for i in range(data.shape[0])]
    if no_spikes_path:
        data2 = pd.read_csv(no_spikes_path)
        data2['y'] = [0 for i in range(data2.shape[0])]
        data = pd.concat([data, data2], axis=0, join='inner', ignore_index=True)

    print(data.shape)
    return data


# shuffle the data, cut into signs and answers
# allocate part of the data for training and verification
# data normalization
def fit_data(data, normolize=False):
    data = data.sample(frac=1)
    y = data['y']
    X = data.drop('y', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8,
                                                        random_state=42,
                                                        stratify=y
                                                        )
    # convert the dataframe into an array, for some reason the indexes are saved, so we delete the first element
    X_train = np.array([[element[i] for i in range(len(element)) if i >= 1] for element in X_train.to_numpy()])
    X_test = np.array([[element[i] for i in range(len(element)) if i >= 1] for element in X_test.to_numpy()])
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # normalize
    if normolize:
        X_train = np.array(
            [[(el - element.min()) / (element.max() - element.min()) for el in element] for element in X_train])
        X_test = np.array(
            [[(el - element.min()) / (element.max() - element.min()) for el in element] for element in X_test])
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    return (X_train, X_test, y_train_cat, y_test_cat)


# example of the simplest neural network
def simple_network(save_fig=False, batch_size=16, validation_split=0.2):
    path = 'spikes.csv'
    path2 = 'not_spikes.csv'
    data = load_data(path, path2)
    x_train, x_test, y_train, y_test = fit_data(data)
    signal_size = x_train.shape[1]
    model = keras.Sequential([Flatten(input_shape=(signal_size, 1)),
                              Dense(500, activation='relu'),
                              Dense(2, activation='softmax')
                              ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.Recall(), 'accuracy']
                  )

    history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_split=validation_split)

    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.grid(True)

    # save result
    if save_fig:
        plt.savefig('NN.png')
    else:
        plt.show()

    print('Training completed, verification results:')
    model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)

    # fit probability to logit
    y_test_cat = y_test.argmax(axis=1)
    y_pred_cat = y_pred.argmax(axis=1)

    # print test metrics
    print('RECALL SCORE: ', recall_score(y_test_cat, y_pred_cat, average='binary'))
    print('PRECISION SCORE: ', precision_score(y_test_cat, y_pred_cat, average='binary'))
    print('ACCURACY SCORE: ', accuracy_score(y_test_cat, y_pred_cat))

    # plot confusion_matrix
    cm = confusion_matrix(y_test_cat, y_pred_cat)
    sns.heatmap(cm, annot=True)
    plt.title('confusion matrix')
    plt.xlabel('prediction')
    plt.ylabel('groud truth')
    plt.show()

    # plot PR curve
    precision,recall,thresholds = precision_recall_curve(y_test[:, 1],y_pred[:,1])
    plt.title('PR curve')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.plot(recall, precision)
    plt.show()


if __name__ == '__main__':
    simple_network()