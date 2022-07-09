from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten
import  numpy as np


#получаем данные для обучения
def load_data(path):
    pass


#перемешиваем данные, разрезаем на признаки и ответы
#выделяем часть данных для обучения и проверки
def fit_data(data):
    pass



signal_size = 50
model = keras.Sequential([Flatten(input_shape=(signal_size, 1)),
                         Dense(100, activation='relu'),
                         Dense(1, activation='softmax')
                         ])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
              )


'''path = ''
data = load_data(path)
x_train, y_train, x_test, y_test = fit_data(data)


model.fit(x_train, y_train, epochs=5, validatiion_split= 0.2)
model.evaluate(x_test, y_test)'''

