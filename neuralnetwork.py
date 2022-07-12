from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


#получаем данные для обучения два файла - со спайками и без
def load_data(spikes_path, no_spikes_path = None):
    data = pd.read_csv(spikes_path)
    data['y'] = [1 for i in range(data.shape[0])]

    if no_spikes_path:
        data2 = pd.read_csv(spikes_path)
        data2['y'] = [0 for i in range(data2.shape[0])]
        data = pd.concat([data, data2], axis=0)

    return data



#перемешиваем данные, разрезаем на признаки и ответы
#выделяем часть данных для обучения и проверки
#нормировка данных
def fit_data(data):
    data = data.sample(frac=1)
    y = data['y']
    X = data.drop('y', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.9,
                                                        random_state=42,
                                                        stratify=y
                                                        )
    #конвертируем датафрейм в массив, почему-то сохраняются индексы, поэтому удаляем первый элемент
    X_train = np.array([[element[i] for i in range(len(element)) if i != 0] for element in X_train.to_numpy()])
    X_test = np.array([[element[i] for i in range(len(element)) if i != 0] for element in X_test.to_numpy()])
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    #нормировка признаков
    X_train = np.array([[(el - element.min()) / (element.max() - element.min()) for el in element] for element in X_train])
    X_test = np.array([[(el - element.min()) / (element.max() - element.min()) for el in element] for element in X_test])

    return (X_train, X_test, y_train, y_test)



#пример простейшей нейросети
def simple_network():
    path = 'spikes/3_0.csv'
    data = load_data(path)
    x_train, x_test, y_train, y_test = fit_data(data)

    signal_size = x_train.shape[1]
    model = keras.Sequential([Flatten(input_shape=(signal_size, 1)),
                             Dense(100, activation='relu'),
                             Dense(1, activation='softmax')
                             ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy']
                  )


    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)
    print('ура я работаю!')


if __name__ == '__main__':
    simple_network()


