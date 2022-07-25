from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# get two datafiles' paths for training - with and without spikes
def load_data(spikes_path, no_spikes_path=None):
    data = pd.read_csv(spikes_path)
    data['y'] = [1 for i in range(data.shape[0])]
    if no_spikes_path:
        data2 = pd.read_csv(no_spikes_path)
        data2['y'] = [0 for i in range(data2.shape[0])]
        data = pd.concat([data, data2], axis=0)

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
    X_train = np.array([[element[i] for i in range(len(element)) if i != 0] for element in X_train.to_numpy()])
    X_test = np.array([[element[i] for i in range(len(element)) if i != 0] for element in X_test.to_numpy()])
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # normalize
    if normolize:
        X_train = np.array(
            [[(el - element.min()) / (element.max() - element.min()) for el in element] for element in X_train])
        X_test = np.array(
            [[(el - element.min()) / (element.max() - element.min()) for el in element] for element in X_test])

    return (X_train, X_test, y_train, y_test)


# example of the simplest neural network
def simple_network(save_fig=False, batch_size=16, validation_split=0.2):
    path = 'spikes.csv'
    path2 = 'not_spikes.csv'
    data = load_data(path, path2)
    x_train, x_test, y_train, y_test = fit_data(data)
    signal_size = x_train.shape[1]
    model = keras.Sequential([Flatten(input_shape=(signal_size, 1)),
                              Dense(500, activation='relu'),
                              Dense(1, activation='softmax')
                              ])
    #myAdam = keras.optimizers.Adam(learning_rate=0.01)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy']
                  )

    history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_split=validation_split)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.grid(True)

    if save_fig:
        plt.savefig('NN.png')

    else:
        plt.show()

    print('Training completed, verification results:')
    model.evaluate(x_test, y_test)


if __name__ == '__main__':
    simple_network(save_fig=False)

