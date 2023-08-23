from model import Model
from models.template import ModelArchitecture
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.metrics import Recall, TruePositives, FalseNegatives, Accuracy
import models.models as models



def test_simple_network() -> None:
    model = Model()
    model.load_data(spikes_path = 'data/learning 24h data/spikes/channel_3.csv',
                    no_spikes_path = 'data/learning 24h data/not_spikes/channel_3.csv',
                    not_spikes_number=20000, print_info=True)
    model.set_model(models.Simple_Network())
    model.fit(epochs=20)
    model.plot_loss_function()
    model.predict()
    model.print_selection_info()
    model.print_metrics()
    model.plot_confusion_matrix()


def test_loaded_model(model_path: str = 'NeuralNetwork/models/model_old.h5') -> None:
    model = Model()
    model.load_data(spikes_path = 'data/learning 24h data/spikes/channel_3.csv',
                    no_spikes_path = 'data/learning 24h data/not_spikes/channel_3.csv',
                    not_spikes_number=100000, print_info=True)

    model.load_model(model_path)
    model.predict()
    model.print_selection_info()
    model.print_metrics()
    model.plot_confusion_matrix()


def test_custom_model(model_architecture: ModelArchitecture, epochs = 50, not_spikes_number = 20000) -> None:
    model = Model()
    model.load_data(spikes_path = 'data/learning 24h data/spikes/channel_3.csv',
                    no_spikes_path = 'data/learning 24h data/not_spikes/channel_3.csv',
                    not_spikes_number=20000, normalize=True, print_info=True)
    model.set_model(model_architecture)
    model.fit(epochs=epochs)
    model.predict() 

    model.print_selection_info()
    model.print_metrics()

    # model.plot_threshold_recall()
    model.plot_loss_function()
    model.plot_confusion_matrix()


if __name__ == '__main__':
    test_custom_model(model_architecture = models.Custom_Model_large(threshold=0.5))


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
