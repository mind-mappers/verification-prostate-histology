from homologous_point_prediction.models import helpers
from homologous_point_prediction.models import att_13_overlay_transformer

import os

MODEL_MAPPINGS = {
    "OVERLAY_TRANSFORMER": att_13_overlay_transformer
}

def get_regression_model(name=list(MODEL_MAPPINGS.keys())[0], image_shape=(512,512,1), learning_rate=0.001, points_per_input=10):
    # If resuming from previous checkpoint
    #if name.startswith("saved_"):
    #    saved_folder = name[len("saved_"):]
    #    print("Loading Saved Model from", saved_folder)
    #    model = helpers.load_model(os.path.join("./homologous_point_prediction/outputs", saved_folder, "model"))
    #    return model

    assert name in MODEL_MAPPINGS
    print("Loading Model: ", name)
    model_package = MODEL_MAPPINGS[name]
    return model_package.get_model(image_shape, learning_rate, points_per_input=points_per_input)