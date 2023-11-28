from homologous_point_prediction.models.helpers import save_model, load_model, set_model_trainability, nested_summary
from homologous_point_prediction.data_processing.data_loader_multipoint import MultiPointDataLoader
from homologous_point_prediction.models.model import get_regression_model
from homologous_point_prediction.models.custom_layers import loss
from homologous_point_prediction.models.custom_callbacks import SaveOnEpoch
from homologous_point_prediction.evaluate import evaluate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.backend import learning_phase_scope
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import builtins
import sys
import os


divide = "".join(["+"] * 100)
require_scale_validation = True

def train(log_dir):
    # Training Config
    batch_size = 4
    point_per_sample = 75
    n_epochs = 350
    learning_rate = 0.00007 #0001
    #learning_rate = 0.05
    input_shape=(512, 512, 1)
    pair_ratio = 0.5
    edge_rate = 0.7

    # Select Model

    model_type = "OVERLAY_TRANSFORMER"



    # Set Up Callbacks
    checkpoint_callback = model_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(log_dir, "{epoch:02d}-{loss:.2f}"), save_weights_only=False, save_freq='epoch', period=25, save_best_only=False)
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=5, patience=100)
    reducelearningrate_callback = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=20, min_lr=0.0000001)
    save_visualizations_callbacck = SaveOnEpoch(log_dir, 25)

    # Model Config
    model = get_regression_model(model_type, input_shape, points_per_input=point_per_sample)
    print("\n\nModel Summary")
    print("6 layers 0.7 edge")
    nested_summary(model, print)

    # Log all training hyperparams
    print("Learning Rate: {0}".format(learning_rate))
    print("Pair Ratio: {0}".format(pair_ratio))
    print("Training Regression")



    # Data Config
    training_data_config = "./homologous_point_prediction/data_processing/metadata/{0}".format("multipoint_train_config.json")
    validation_data_config = "./homologous_point_prediction/data_processing/metadata/{0}".format("multipoint_validation_config.json")

    data_loader = MultiPointDataLoader(training_data_config, batch_size=batch_size, num_points=point_per_sample, warped_pair_rate=pair_ratio, edge_rate=edge_rate)
    validation_data_loader = MultiPointDataLoader(validation_data_config, batch_size=batch_size, num_points=point_per_sample)


    data_loader.save_sample_batch(log_dir)
    data_loader.summarize()

    model.compile(optimizer=Adam(lr=learning_rate, beta_1=0.9), loss=['mae'])
    model.fit(x=data_loader, epochs=n_epochs, callbacks=[reducelearningrate_callback, checkpoint_callback, save_visualizations_callbacck], validation_data=validation_data_loader, verbose=2)

    set_model_trainability(model, True)
    # Save results
    save_model(model, os.path.join(logging_dir, "model"))
    return model


# Run train sending all outputs to script_outputs
logging_dir = sys.argv[1] if len(sys.argv) > 1 else "."

with open(logging_dir + "/script_outputs.txt", "w") as f:
    with redirect_stdout(f):
        builtins.in_training_session = False
        model = train(logging_dir)
        builtins.in_training_session = False
        evaluate(model, logging_dir, requires_scaling=require_scale_validation)
