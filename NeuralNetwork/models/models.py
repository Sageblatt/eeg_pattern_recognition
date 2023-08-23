from models.template import ModelArchitecture
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.metrics import Recall, TruePositives, FalseNegatives, Accuracy


class Simple_Network(ModelArchitecture):
    def get_model(self, signal_size: int) -> keras.Model:
        model = keras.Sequential([Flatten(input_shape=(signal_size, 1)),
                                    Dense(40, activation='relu'),
                                    Dense(1, activation='sigmoid')
                                    ])
        metrics = [Recall(name='recall', thresholds=[self.threshold])]
        model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=metrics
                        )
        return model
    

class Custom_Model(ModelArchitecture):
    """
        first layer: 25-45 optimum neurons
    """
    def get_model(self, signal_size: int) -> keras.Model:
        model = keras.Sequential([Flatten(input_shape=(signal_size, 1)),
                                    Dense(32, activation='relu'),
                                    Dense(16, activation='tanh'),
                                    Dense(8, activation='relu'),
                                    # Dropout(0.5),
                                    Dense(1, activation='sigmoid')
                                    ])
        metrics = [Recall(name='recall', thresholds=[self.threshold])]
        model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=metrics
                        )
        return model