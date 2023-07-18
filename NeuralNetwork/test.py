from model import Model
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.metrics import Recall, TruePositives, FalseNegatives, Accuracy


def simple_network(signal_size: int, threshold: float = 0.5) -> keras.Model:
    model = keras.Sequential([Flatten(input_shape=(signal_size, 1)),
                              Dense(40, activation='relu'),
                              Dense(1, activation='sigmoid')
                              ])
    metrics = [Recall(name='recall', thresholds=[threshold])]
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=metrics
                  )
    return model



if __name__ == '__main__':
    model = Model()
    model.load_data('data/learning 24h data/spikes/channel_3.csv', 'data/learning 24h data/not_spikes/channel_3.csv', print_info=True)
    #model.set_model(simple_network, threshold=0.5)
    #model.fit()
    model.load_model('NeuralNetwork/models/model_old.h5')
    model.predict()

    model.print_metrics()
    model.plot_confusion_matrix()

    #model.save_model('NeuralNetwork/models/model_big_data')

    """
    FUNCTIONALITY 
    - to load model: 

    model.load_model('NeuralNetwork/models/model_old.h5')
    model.predict()
    
    - to make new model:
    model.set_model(simple_network)
    model.fit()
    model.predict()

    - available methods for analysis:

    model.plot_loss_function()
    model.print_metrics()
    model.plot_confusion_matrix()
    model.plot_threshold_recall()
    model.plot_confusion()

    model.save_model('NeuralNetwork/models/model_big_data')
    """
