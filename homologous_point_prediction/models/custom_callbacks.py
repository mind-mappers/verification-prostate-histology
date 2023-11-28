import tensorflow as tf
import builtins
from homologous_point_prediction.evaluate import show_in_out


class SaveOnEpoch(tf.keras.callbacks.Callback):

    def __init__(self, log_dir, log_interval):
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.repeats = 0

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            self.repeats += 1
        if epoch != 0 and (epoch % self.log_interval) == 0:
            builtins.in_training_session = False
            show_in_out(self.model, self.log_dir, str(epoch)+"_"+str(self.repeats))
        if epoch >= 30:
            builtins.in_training_session = True
