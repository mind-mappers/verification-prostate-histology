from homologous_point_prediction.models.custom_layers import AddOnes, ImagePatchEncoder3, Softmax, SquareBased, DotProductCorrelation, L2Norm, ModelToggle, loss, HardCodedPositions, ActivePointMaskMultiplication, EdgePointAugmentation, RangeOut, ImagePatchEncoder2, ActivePointMask, GatherPatches, TiledCorrelation3D, PointTranslation, Correlation3D, GrayscaleToRGB, ProjectPoints, Patches, PositionEncoder, ModalityEncoder, PointPatchEncoder, PointAttentionMask, ImagePatchEncoder, ExpandToBatch, ReduceAttentionScores
from homologous_point_prediction.models.multihead_attention import EinsumDense, MultiHeadAttention, Softmax, gelu
from homologous_point_prediction.models.model import get_regression_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def save_model(model, path):
    model.save_weights(path+".h5")
    return tf.keras.models.save_model(model, path)

def load_model(path, use_weights=True):
    if use_weights:
        model = get_regression_model(points_per_input=75)
        model.load_weights(path+".h5")
        return model
    result =  tf.keras.models.load_model(path, 
        custom_objects={
            'PointTranslation': PointTranslation,
            'TiledCorrelation3D': TiledCorrelation3D,
            'Correlation3D': Correlation3D,
            'GrayscaleToRGB': GrayscaleToRGB,
            'MultiHeadAttention': MultiHeadAttention,
            'ProjectPoints': ProjectPoints,
            'Patches': Patches,
            'PositionEncoder': PositionEncoder,
            'ModalityEncoder': ModalityEncoder,
            'PointPatchEncoder': PointPatchEncoder,
            'ImagePatchEncoder': ImagePatchEncoder,
            'PointAttentionMask': PointAttentionMask,
            'ExpandToBatch': ExpandToBatch,
            'ReduceAttentionScores': ReduceAttentionScores,
            'GatherPatches': GatherPatches,
            'ActivePointMask': ActivePointMask,
            'ModalityEncoder': ModalityEncoder,
            'ImagePatchEncoder2': ImagePatchEncoder2,
            'RangeOut': RangeOut,
            'ActivePointMaskMultiplication': ActivePointMaskMultiplication,
            'gelu': gelu,
            'HardCodedPositions': HardCodedPositions,
            'EdgePointAugmentation': EdgePointAugmentation,
            'custom_loss': loss,
            'ModelToggle': ModelToggle,
            'L2Norm': L2Norm,
            'DotProductCorrelation': DotProductCorrelation,
            'SquareBased': SquareBased,
            'Softmax': Softmax,
            'ImagePatchEncoder3': ImagePatchEncoder3,
            'AddOnes': AddOnes
            }, compile=False)
    return result

def get_start_layers(model):
    """
    Return all layers with _start prefix(including nested)
    """
    start_layers = []
    try:
        for layer in model.layers:
            if layer.name.startswith("start_"):
                start_layers.append(layer)
            else:
                start_layers.extend(get_start_layers(layer))
    except:
        return start_layers


def set_model_trainability(model, t):
    model.trainable = t
    for layer in model.layers:
        layer.trainable = t




def freeze_start_layers(model):
    """
    Set the trainability of every layer with _start as a prefix to false
    """
    start_layers = get_start_layers(model)
    for start_layer in start_layers:
        print("Freezing layer", start_layer.name)
        start_layer.trainable = False
    model.compile(loss=['mse'], optimizer=Adam(lr=0.0002, beta_1=0.5))
    return model

def nested_summary(model, print_fn):
    try:
        model.summary(print_fn=print_fn)
        for layer in model.layers:
            nested_summary(layer, print_fn)
    except:
        pass